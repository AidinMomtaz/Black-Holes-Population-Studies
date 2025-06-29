import numpy as np

def train(info= False, return_info = False):
    #This is the NF training function
    import torch
    from models.flows import MAF,BatchNormFlow
    from tqdm import tqdm
    from time import time as t_time
    from tools.constants import defaults
    from os.path import exists, join
    from os import mkdir

    #Varibles ----------------------------------------------------
    # Pass user settings from info dictionary see defaults
    if info is not False:
        defaults.update(info)# info is highest protity data, we update the defults with info
    
    info =  defaults 
    
    #Automatic Time Labelling 
    if info['time'] is not False: info['label'] = info['label']+'_'+str(info['time'])+'min' 
    info['num_cond_inputs'] = len(info['pop_cols']) # number of population parameters, e.g. num_cond_inputs = 2 for (alpha, beta): two power-law index
    info['num_inputs'] = len(info['sample_cols']) # number of event parameters, e.g. num_inputs =3 for (m1,q,z) }
    info['d_score_rate'] =  []
    info['epoch_time'] = []
    print('num_cond_inputs = ',info['num_cond_inputs'],', num_inputs =', info['num_inputs'])

    #Training Config------------------------------

    # MAF for training p(θ|λ)
    # θ: event parameters ( parameters for single GW event), 
    # λ: population parameters (parameters for population model)
    # config for normalizing flow MAF
    info['MAFconfig'] = {'num_cond_inputs' : info['num_cond_inputs'], # number of population parameters, e.g. num_cond_inputs = 2 for (alpha, beta): two power-law index 
                        'act' : info['A'], # activation function for NF model
                        'num_blocks' : info['blocks'],# number of blocks in the model
                        'num_hidden' : info['hidden'],                     
                        'num_inputs' : info['num_inputs']} # number of event parameters, e.g. num_inputs =3 for (m1,q,z) }   

    if not exists(info['outdir']): mkdir(info['outdir'])
    info['best_model_path'] = join(info['outdir'],info['label']+'1low.pt')
    device = torch.device(info['device'])
    model = MAF(**info['MAFconfig']).to(device) # initialize the MAF model
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4) 
    status = ', '
    status+='Early stopping is on.' if info['early_stop'] else ', Early stopping is off.'
    status+='DataLoader is on.' if info['dataloader'] else ', DataLoader is off.'
    print('Using', device, status)

    def data_transform(path, sample_cols = info['sample_cols'], pop_cols = info['pop_cols']):
        from pandas import read_parquet
        #from pathos.multiprocessing import ProcessingPool as Pool
        # If we have `Nsim` set of population parameters 
        # and each simulation with 'Nevent' binary merger events
        # and each event with `Ndim` event parameters,
        # Then samples and pop_parameters should come in with shape (Nsim*Nevent,Ndim)
        df = read_parquet(path)
        s=np.array([df[sample_cols]])
        c=np.array([df[pop_cols]])

        Nsim, Nevent, Ndim = s.shape 
        Nsim_c, Nevent_c, Ndim_c = c.shape
        
        # convert to tensor form and to the device being used ( cpu or gpu )
        torch_s = torch.from_numpy(s).float()
        torch_c = torch.from_numpy(c).float()

        #if not info['dataloader']:
        #    torch_s = torch.to(device) 
        #    torch_c = torch.to(device)
        return Nsim, Nevent, Ndim, torch_s.reshape(Nsim*Nevent, Ndim), torch_c.reshape(Nsim_c*Nevent_c, Ndim_c)

    # This way we all into RAM and then split in dataloader (Faster on our hardware)
    Nsim_train, Nevent_train, Ndim, samples, pop_parameters = data_transform(info['training_file']) 
    Nsim_valid, Nevent_valid, Ndim, valid_samples, valid_pop_parameters = data_transform(info['validation_file'])

    if info['dataloader']:
        from torch.utils.data import DataLoader# Only import if used
        class mk_Dataset():# Requires dataframe
            def __init__(self, path):
                self.x_train = samples
                self.y_train = pop_parameters

            def __len__(self):
                return len(self.y_train)

            def __getitem__(self, idx):
                return self.x_train[idx], self.y_train[idx]
        
        train_ds = mk_Dataset(info['training_file'])
        train_loader = DataLoader(train_ds, 
                       batch_size=info['batch_size'],
                       num_workers=info['workers'],
                       pin_memory=info['PIN_MEM'],
                       shuffle=False)
        """
        valid_ds = mk_Dataset(info['training_file'])
        valid_loader = DataLoader(valid_ds, 
                batch_size=info['batch_size'],
                num_workers=info['workers'],
                pin_memory=info['PIN_MEM'],
                shuffle=False)
        """
        
    if not info['dataloader']:
        print('Training set: {} simulation, each with {} events and each event with {} parameters'.format(Nsim_train, Nevent_train, Ndim))
        print('Validation set: {} simulation, each with {} events and each event with {} parameters'.format(Nsim_valid, Nevent_valid, Ndim))

    train_loss = [] # list to store training loss
    valid_loss = [] # list to store validation loss

    def loss_function(ps,cp):
        # We are aiming to find the mimumium for loss function in ML
        # so we add a negative sign to the llh to be our loss function
        likelihood = model.log_probs(ps,cp)
        return -torch.mean(likelihood)

    # define function to save the model 
    def save_checkpoint(model, optimizer, epoch, save_path):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }, save_path)

    # Define a large enough loss so that it will be replaced after first few runs.
    # Indicator for saving models during the training
    temp_loss1 = 1e12 
    temp_loss2 = 1e10

    ########################## Training ##################################
    start = t_time()
    batch_val_loss= np.array([])
    stop =False
    threshold = 0.0000001

    model.train() 
    for i in tqdm(range(info['epochs'])):
        if not stop:
            if not info['dataloader']:  
                optimizer.zero_grad()
                loss = loss_function(samples, pop_parameters)
                loss.backward()
                optimizer.step()

                for module in model.modules():
                    if isinstance(module, BatchNormFlow):module.momentum = 0

                with torch.no_grad():
                    model(samples, pop_parameters)
                    
                for module in model.modules():
                    if isinstance(module, BatchNormFlow): module.momentum = 1
                            
                # calculate validation loss
                with torch.no_grad():
                    loss2 = loss_function(valid_samples, valid_pop_parameters)
                
                train_loss.append(loss.cpu().detach().numpy())
                valid_loss.append(loss2.cpu().detach().numpy())
                
            else: 
                for train_batch in train_loader:
                    train_batch_samples = train_batch[0].to(device)
                    train_batch_pop = train_batch[1].to(device)

                    optimizer.zero_grad()
                    loss = torch.mean(loss_function(train_batch_samples, train_batch_pop))
                    loss.backward() 
                    optimizer.step()

                    for module in model.modules():
                        if isinstance(module, BatchNormFlow): module.momentum = 0

                    with torch.no_grad():
                        model(train_batch_samples, train_batch_pop)
                        
                    for module in model.modules():
                        if isinstance(module, BatchNormFlow): module.momentum = 1
                    
                    # calculate validation loss
                    with torch.no_grad():
                        loss2 = torch.mean(loss_function(valid_samples.to(device), valid_pop_parameters.to(device)))
                    
                    batch_val_loss=np.append(batch_val_loss, loss2.cpu().detach().numpy())

                    """
                    #Validation dataloader (slower)
                    validation_batch_loss = np.array([])
                    with torch.no_grad():
                        for val_batch in valid_loader:
                            val_batch_samples = val_batch[0].to(device)
                            val_batch_pop = val_batch[1].to(device)
                            validation_batch_loss = np.append(validation_batch_loss, torch.mean(loss_function(val_batch_samples.to(device), val_batch_pop.to(device))).cpu().detach().numpy())
                    loss2 = np.mean(validation_batch_loss)
                    batch_val_loss=np.append(batch_val_loss, loss2)"""
                         
                train_loss.append(loss.cpu().detach().numpy())
                valid_loss.append(np.mean(batch_val_loss))

            # save training loss and validation loss
            # save 2 models with lowest validation loss 
            if valid_loss[-1] < temp_loss1:
                if valid_loss[-1] < temp_loss2:
                    temp_loss2 = valid_loss[-1]
                    save_checkpoint(model, optimizer, i, info['best_model_path'])
                    info['best_loss'] = valid_loss[-1]

                else:
                    temp_loss1 = valid_loss[-1]
                    save_checkpoint(model, optimizer, i, join(info['outdir'],info['label']+'2low.pt'))
            
            info['epoch_time'].append(t_time()-start)
            
            if len(info['epoch_time'])>1:
                info['d_score_rate'].append((float(valid_loss[-1])-float(valid_loss[-2]))/(info['epoch_time'][-1]- info['epoch_time'][-2]))
                if info['early_stop']:
                    if abs(info['d_score_rate'][-1]) < threshold:
                        print('stopped_early')
                        stop =True
                        info['stopped_early']= True
                        info['epochs'] = i 
        
        if info['time']:
            if t_time()-start>float(info['time'])*60:
                info['stopped_early']= True
                info['epochs'] = i
                break

    # save the model after the whole training (probably with lowest training loss)
    save_checkpoint(model, optimizer, i, join(info['outdir'],info['label']+'_tloss1low.pt'))

    loss_path =  join(info['outdir'],info['label']+'_loss.npz')
    ########################## saving training loss and validation loss ##################################
    np.savez(loss_path,train_loss=train_loss,valid_loss=valid_loss, config=info['MAFconfig'])

    runtime = t_time()-start
    info['final_loss'] = valid_loss[-1]
    info['loss_path'] = loss_path
    info['runtime'] = runtime
    
    for key in info.keys():
        if isinstance(info[key], (np.ndarray, torch.Tensor)):
            info[key] = info[key].tolist()
        elif isinstance(info[key], float):
            info[key] = str(info[key])  

    from json import dump as jdump
    from pathlib import Path
    json_path = Path(info['outdir'], f"{info['label']}.json")
    print(info)
    with open (json_path, 'w') as f:
        pass
        jdump(info, f)
    
    print('Time used for training:{} s'.format(runtime))
    print('run information saved to', json_path)

    if return_info is not False:
        return info

def train_from_prev(path, params):
    #Systematically check the set of conditions have been trained before
    #If so, load the model and train from the last epoch
    from tools.tuning import get_run, get_matches
    
    default_vals = get_run('trained_model/default.json')
    #need to add final epoch
    N = default_vals['epochs']
    del default_vals['epochs']
    
    if 'epochs' in params:
        N = params['epochs']
        del params['epochs']
        
    match_critera =  ["training_file", "validation_file", "A", "blocks", "hidden", "log", "early_stop", "MAFconfig"]
    default_trim = {key : default_vals[key] for key in default_vals if key in match_critera}
    print(default_trim)
    param_test = default_trim.update(params)
    
    print(param_test)

    matches = get_matches(path, param_test)
    print(matches)
    prev_epoch_dist=9999
    for m in matches:
        match_dict=get_run(m)
        match_epoch = match_dict['epochs']
        if stop_epoch in match_dict:
            match_epoch = match_dict['stop_epoch']
        
        if match_epoch < N:
            epoch_dist = N - match_epoch
            if epoch_dist < prev_epoch_dist:
                closest_match = match_dict
                prev_epoch_dist = epoch_dist 
        
    if prev_epoch_dist!= 9999:
        print('closest_match ', epoch_dist, 'epochs away')
        print(match_dict)
    
    return match_dict

def prep_model(info, device = 'cuda', lr = 1e-3, weight_decay=1e-4):
    #cpu more stable for sampling

    import torch
    from tools.flows import MAF
    device = torch.device(device)
    model = MAF(**info['MAFconfig']).to(device) # initialize the MAF model
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    checkpoint = torch.load(info['best_model_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, device, optimizer

def sample_run(info, hyperparams, ref_data = 'test.pq', n_samples = 10000):
    from tools.tuning import get_run
    from tools.constants import data_dir
    from os.path import join
    import numpy as np
    import torch
    import pandas as pd
    from figaro.transform import transform_from_probit

    if isinstance(info, str):info=get_run(info)

    model, device, optimizer = prep_model(info)
    model.eval() # evaluate mode
    
    pop_para = np.array([np.ones(int(n_samples))*hyperparams[0],np.ones(int(n_samples))*hyperparams[1]]).T
                              
    #Needs Initialization 
    event = np.repeat([[10,0.3]], len(pop_para), axis=0)

    a = torch.from_numpy(event).float().to(device)# event what
    b = torch.from_numpy(pop_para).float().to(device)
    model.log_probs(a, b)

    # sample from NF model
    s2 = transform_from_probit(model.sample(num_samples=len(pop_para), cond_inputs=b).detach().cpu().numpy(), np.array([[0, 100], [0, 1.01]]))
    #s2 = model.sample(num_samples=len(pop_para), cond_inputs=b).detach().cpu().numpy()

    
    df_NF=df_NF = pd.DataFrame(data=np.column_stack([s2, b.detach().cpu().numpy()]), index=None, columns=['Mass_0','q','Z','alpha'])

    return df_NF 

def eval_likelihood(info, samples, pop_parameters):
    import numpy as np
    import torch
    from torch import nn
    from tools.flows import MAF,BatchNormFlow
    from tqdm import tqdm
    import pandas as pd 

    

    if isinstance(info, str):info=get_run(info)

    def array_transform(samples, pop_parameters):
        #lighter verison for just loading in data already loaded
        s = samples
        c = pop_parameters

        Nsim, Nevent, Ndim = s.shape
        # convert to tensor form and to the device being used ( cpu or gpu )
        torch_s = torch.from_numpy(s).float()
        torch_c = torch.from_numpy(c).float()
        return Nsim, Nevent, Ndim, torch_s.reshape(Nsim*Nevent, Ndim), torch_c.reshape(Nsim*Nevent, Ndim)

    Nsim_train, Nevent_train, Ndim, samples, pop_parameters = array_transform(samples, pop_parameters) 

    model, device, optimizer = prep_model(info)
    model.eval() 
    optimizer.zero_grad()

    with torch.no_grad():model(samples, pop_parameters)
        
    for module in model.modules():
        if isinstance(module, BatchNormFlow):
            module.momentum = 1
                    
    # calculate validation loss
    with torch.no_grad():
        likelihood_log = model.log_probs(samples,pop_parameters)

    return np.exp(likelihood_log.cpu().detach().numpy())

def eval_likelihood_test(model, samples, pop_parameters):
    import numpy as np
    import torch
    from torch import nn
    from tools.flows import MAF,BatchNormFlow
    from tqdm import tqdm
    import pandas as pd 

    

    #if isinstance(info, str):info=get_run(info)

    def array_transform(samples, pop_parameters):
        #lighter verison for just loading in data already loaded
        s = samples
        c = pop_parameters

        Nsim, Nevent, Ndim = s.shape
        # convert to tensor form and to the device being used ( cpu or gpu )
        torch_s = torch.from_numpy(s).float()
        torch_c = torch.from_numpy(c).float()
        return Nsim, Nevent, Ndim, torch_s.reshape(Nsim*Nevent, Ndim), torch_c.reshape(Nsim*Nevent, Ndim)

    Nsim_train, Nevent_train, Ndim, torch_s, torch_c = array_transform(samples, pop_parameters)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    

    with torch.no_grad():model(samples, pop_parameters)
        
    for module in model.modules():
        if isinstance(module, BatchNormFlow):
            module.momentum = 1
                    
    # calculate validation loss
    with torch.no_grad():
        likelihood_log = model.log_probs(samples,pop_parameters)

    return np.exp(likelihood_log.cpu().detach().numpy())



def eval_likelihood_inference(model, samples, pop_parameters):
    import numpy as np
    import torch
    from tools.flows import BatchNormFlow
    from figaro.transform import transform_to_probit

   
    

    # Define bounds for each parameter (same order as input dims)
    sample_bounds = np.array([[0.5, 100.0], [0.01, 1.]])        # [mass_1, mass_ratio]
    pop_bounds    = np.array([[0.5, 5.0], [0.0001, 0.03]])       # [alpha, Z]

    def clip_parameters(params, bounds, epsilon=1e-6):
    # Clip slightly inside the bounds
        return np.clip(params, bounds[:, 0] + epsilon, bounds[:, 1] - epsilon)

    pop_parameters = clip_parameters(pop_parameters, pop_bounds)
    samples = clip_parameters(samples, sample_bounds)

    

    # Transform samples to probit space
    samples = transform_to_probit(samples, sample_bounds)

    


   

    
   

    # Flatten and convert to torch
    Nsim, Nevent, Ndim = samples.shape
    NsimNevent = Nsim * Nevent
    torch_s = torch.from_numpy(samples).float().reshape(NsimNevent, Ndim)
    torch_c = torch.from_numpy(pop_parameters).float().reshape(NsimNevent, Ndim)

    


   

    # Send to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch_s = torch_s.to(device)
    torch_c = torch_c.to(device)

    

    # Ensure proper eval mode
    model.eval()
    for module in model.modules():
        if isinstance(module, BatchNormFlow):
            module.eval()
            module.momentum = 1.0

    # (Optional) dummy forward call
    with torch.no_grad():
        _ = model(torch_s, torch_c)

   #check for -inf values for both torch tensors
    if torch.isinf(torch_s).any() or torch.isinf(torch_c).any():
        raise ValueError("Infinite values detected in input tensors!")
    #check for NaNs for both torch tensors
    if torch.isnan(torch_s).any() or torch.isnan(torch_c).any():
        raise ValueError("NaN values detected in input tensors!")
    print(torch_s.shape, torch_c.shape)
        

    # Evaluate log-likelihood
    with torch.no_grad():
        likelihood_log = model.log_probs(torch_s, cond_inputs = torch_c)
        if torch.isnan(likelihood_log).any():
            print("⚠️ NaNs detected in log_probs!")
        print("likelihood_log stats — min:", likelihood_log.min().item(),
              "max:", likelihood_log.max().item(),
              "mean:", likelihood_log.mean().item())

    # Convert to density
    result = torch.exp(likelihood_log).detach().cpu().numpy()
    return result






def eval_likelihood_inference_1(model, samples, pop_parameters, device=None):
    import numpy as np
    import torch
    from torch.cuda.amp import autocast
    import time

    # Choose device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)

    # Only set batchnorm momentum once (assuming this func called multiple times)
    for module in model.modules():
        if hasattr(module, "momentum"):
            module.momentum = 1

    # Transform numpy arrays to flattened torch tensors
    def array_transform(samples_np, pop_params_np):
        Nsim, Nevent, Ndim = samples_np.shape
        s = torch.from_numpy(samples_np).float().reshape(Nsim * Nevent, Ndim)
        c = torch.from_numpy(pop_params_np).float().reshape(Nsim * Nevent, Ndim)
        return s.to(device), c.to(device)

    start_total = time.time()

    start_transform = time.time()
    samples_tensor, pop_parameters_tensor = array_transform(samples, pop_parameters)
    end_transform = time.time()
    print(f"[Transform] {end_transform - start_transform:.4f}s")

    # Use autocast for mixed precision
    start_forward = time.time()
    with torch.no_grad(), autocast():
        model(samples_tensor, pop_parameters_tensor)
    end_forward = time.time()
    print(f"[Forward pass] {end_forward - start_forward:.4f}s")

    start_logprobs = time.time()
    with torch.no_grad(), autocast():
        log_probs = model.log_probs(samples_tensor, pop_parameters_tensor)
    end_logprobs = time.time()
    print(f"[Log probs] {end_logprobs - start_logprobs:.4f}s")

    start_result = time.time()
    result = torch.exp(log_probs).cpu().numpy()
    end_result = time.time()
    print(f"[To numpy] {end_result - start_result:.4f}s")

    print(f"[Total time] {end_result - start_total:.4f}s")
    return result



def test_likelihood(info, hyperparams = [0.0001, 0.5], m_range = (0, 80)):
    m = np.linspace(m_range[0],m_range[1], num=300)
    M1, M2 = np.meshgrid(m, m)
    data = np.vstack((M1.flatten(), M2.flatten())).T
    hyperparams = np.repeat([hyperparams], len(data), axis =0)
    likelihood=eval_likelihood(info, np.array([data]), np.array([hyperparams]))
    return hyperparams, data[:, 0], data[:, 1], likelihood


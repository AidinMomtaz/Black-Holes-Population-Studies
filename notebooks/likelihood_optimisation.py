import numpy as np
from figaro.load import load_data, load_selection_function
from figaro.cosmology import Planck15, dVdz_approx_planck15
from scipy.optimize import minimize
from tools.model import eval_likelihood_inference
from tools.model import eval_likelihood_test
import pandas as pd
from tools.tuning import get_run
from os.path import join
from tools.constants import root_dir, data_path #full_data_path
from tools.model import prep_model
from figaro.transform import transform_to_probit
import numpy as np
import torch
from numba import njit


run_path=join(root_dir, 'tuning', 'test','blocks_20_hidden_128_A_relu_N_2600.json')

info = get_run(run_path)
print(info)
model, device, optimizer = prep_model(info, device='cuda')
model.eval() 
optimizer.zero_grad()

zmax   = 2.3 # LVK paper
z_norm = np.linspace(0,zmax,1000)
dz     = z_norm[1]-z_norm[0]

@njit
def _unnorm_powerlaw_redshift(z, k):
    reg_const = (1+zmax)/dVdz_approx_planck15(zmax)
    return (1+z)**k * dVdz_approx_planck15(z)/(1+z) * reg_const

@njit
def powerlaw_redshift(z, k):
    norm = np.sum(_unnorm_powerlaw_redshift(z_norm,k)*dz)
    return _unnorm_powerlaw_redshift(z, k)/norm


def normalising_flow_pdf(M1, q, alpha, Z):
    """
    Normalising flow pdf evaluation with probit-transformed inputs.
    Evaluates probability densities for primary mass M1 and mass ratio q
    conditioned on alpha and Z using a normalizing flow model.
    """

    
    # Flatten M1 and q for batched evaluation
    M1_flat = M1.flatten()
    q_flat = q.flatten()




    # Create dataset dictionary compatible with eval_likelihood_inference_1
    dataset = {
        "mass_1": M1_flat,
        "mass_ratio": q_flat
    }
    

    # Evaluate probability density using the existing utility function
    prob = eval_likelihood_inference(
        model,
        np.atleast_3d([dataset["mass_1"], dataset["mass_ratio"]]).T,
        np.repeat([[alpha, Z]], len(dataset["mass_1"]), axis=0)
    )

    # Reshape the result back to the original shape
    prob = prob.reshape(M1.shape)
   
    return prob
    



def _log_likelihood(x, events, pe_prior, injections, inj_pdf, n_total_inj):
    """
    Likelihood following Mandel, Farr & Gair (
    """
    # Unpack data
    M1 = events[:,:,0]
    q  = events[:,:,1]
    z  = events[:,:,2]
    # Unpack injections
    M1_inj = injections[:,0]
    q_inj  = injections[:,1]
    z_inj  = injections[:,2]
   
    # Evaluate pdf on events
    redshift_pdf = powerlaw_redshift(z, x[0])
    mass_pdf     = normalising_flow_pdf(M1, q, x[1], x[2])
    # Individual events likelihood
    like_events = mass_pdf*redshift_pdf
    logL_events = np.log(np.mean(like_events/pe_prior, axis = 1))
    if not np.isfinite(logL_events).all():
        return -np.inf
    # Evaluate pdf on injections
    redshift_inj_pdf = powerlaw_redshift(z_inj, x[0])
    mass_inj_pdf     = normalising_flow_pdf(M1_inj, q_inj, 0.5, 0.001)
    # Injections likelihood
    like_inj = mass_inj_pdf*redshift_inj_pdf
    logL_inj = np.log(np.sum(like_inj/inj_pdf)) - np.log(n_total_inj)
    if not np.isfinite(logL_inj):
        return -np.inf
    

    return np.sum(logL_events) - len(events)*logL_inj
    
def log_likelihood(x, events, pe_prior, injections, inj_pdf, n_total_inj):
    """
    Change sign of likelihood for optimisers (i.e. minimise the function value, but we actually want the maximum)
    """
    return -_log_likelihood(x, events, pe_prior, injections, inj_pdf, n_total_inj)

# Convert dict to vector and back
def pack_params(x):
    return [x['kappa'], x['alpha'], x['Z']]

def unpack_params(vec):
    return {'kappa': vec[0], 'alpha': vec[1], 'Z': vec[2]}

# Wrapper to make log_likelihood compatible with minimize
def wrapped_logL(vec):
    x = unpack_params(vec)
    return log_likelihood(x, events, pe_prior, injections, inj_pdf, n_total_inj)


if __name__ == '__main__':

    # folder with PE samples
    # Replace with your path to folder with LVK samples
    samples_folder = 'D:\gwpopulation\examples\Data 1'
    # File with injections 
    # Replace with your path to file with LVK injections
    injections_file = 'D:\gwpopulation\examples\endo3_bbhpop-LIGO-T2100113-v12.hdf5'

    # Load PE samples using FIGARO loader
    events, names = load_data(samples_folder, par = ['m1','q','z'], likelihood = False)
    events = [ev[ev[:,0]<100] for ev in events] 
    # Convert events list to array (number of samples is variable, needs to be the same for all)
    nmin          = np.min([len(ev) for ev in events])
    events        = np.array([ev[:nmin] for ev in events])
    # PE priors
    vol      = Planck15.ComovingVolume(zmax)/1e9       # Scaling factor (Gpc)
    dVdz     = dVdz_approx_planck15(events[:,:,2])/vol # redshift prior
    M1       = events[:,:,0]                           # Conversion from (M1, M2) to (M1, q)
    #print M1 and q
    # (1+z)^-1 from time dilation times (1+z)^2 for detector-frame component masses
    oneplusz = (1.+events[:,:,2])
    # Parameter estimation prior
    pe_prior = dVdz*oneplusz*M1
    # Load injection samples and relative sampling PDF (for selection effects)
    injections, inj_pdf, n_total_inj, duration = load_selection_function(injections_file, par = ['m1','q','z'])
    # Call your optimising function
    # Suggested bounds for kappa: [-4,8]
    # Expected value: kappa = 3~4
    x0 = {'kappa': 3.0, 'alpha': 1.0, 'Z': 0.001}  # Example


    
    opt_pars = minimize(log_likelihood, [3.0, 1. , 1e-3], method='powell', bounds=[(-4, 8), (0.5, 4.6), (0.0001, 0.1)], args=(events, pe_prior, injections, inj_pdf, n_total_inj))
    print(opt_pars)


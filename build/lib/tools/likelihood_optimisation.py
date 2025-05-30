import numpy as np
from figaro.load import load_data, load_selection_function
from figaro.cosmology import Planck15, dVdz_approx_planck15

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
    Normalising flow pdf evaluation goes here
    """
    pass
    
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
    redshift_pdf = powerlaw_redshift(z, x['kappa'])
    mass_pdf     = normalising_flow_pdf(M1, q, x['alpha'], x['Z'])
    # Individual events likelihood
    like_events = mass_pdf*redshift_pdf
    logL_events = np.log(np.mean(like_events/pe_prior, axis = 1))
    if not np.isfinite(logL_events).all():
        return -np.inf
    # Evaluate pdf on injections
    redshift_inj_pdf = powerlaw_redshift(z_inj, x['kappa'])
    mass_inj_pdf     = normalising_flow_pdf(M1_inj, q_inj, x['alpha'], x['Z'])
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


if __name__ == '__main__':

    # folder with PE samples
    # Replace with your path to folder with LVK samples
    samples_folder = '/Users/rinaldi/Documents/GWTC3/o3pop_bbh'
    # File with injections
    # Replace with your path to file with LVK injections
    injections_file = '/Users/rinaldi/Documents/projects/astro_dist/selection_function/o1+o2+o3_bbhpop_real+semianalytic-LIGO-T2100377-v2.hdf5'

    # Load PE samples using FIGARO loader
    events, names = load_data(samples_folder, par = ['m1','q','z'], likelihood = False)
    # Convert events list to array (number of samples is variable, needs to be the same for all)
    nmin          = np.min([len(ev) for ev in events])
    events        = np.array([ev[:nmin] for ev in events])
    # PE priors
    vol      = Planck15.ComovingVolume(zmax)/1e9       # Scaling factor (Gpc)
    dVdz     = dVdz_approx_planck15(events[:,:,2])/vol # redshift prior
    M1       = events[:,:,0]                           # Conversion from (M1, M2) to (M1, q)
    # (1+z)^-1 from time dilation times (1+z)^2 for detector-frame component masses
    oneplusz = (1.+events[:,:,2])
    # Parameter estimation prior
    pe_prior = dVdz*oneplusz*M1
    # Load injection samples and relative sampling PDF (for selection effects)
    injections, inj_pdf, n_total_inj, duration = load_selection_function(injections_file, par = ['m1','q','z'])
    # Call your optimising function
    # Suggested bounds for kappa: [-4,8]
    # Expected value: kappa = 3~4
    opt_pars = optimiser_of_your_choice(log_likelihood, other_parameters)

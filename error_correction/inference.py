import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import seaborn as sns
from error_correction.model import *

def emcee_biased_fit(data, nwalkers=50, nsteps=2000):
    """
    inputs
        data : SyntheticData or GenerateData-like object
            data : Pandas dataframe
            params : dictionary with parameter values
        nwalkers : number of MCMC walkers, int
        nsteps : number of MCMC steps, int
    output
        sampler : emcee results
    """
    # unpack parameters
    n_chrom = data.params['n_chrom']
    n_cells = data.params['n_cells']
    dNk = np.array(data.data['dNk'])
    ndim = 2
    # set up starting positions
    gaussian_ball = 1.e-3 * np.random.randn(nwalkers, ndim)
    starting_positions = (1 + gaussian_ball) * [0.05,0.25]
    # run sampler
    sampler = emcee.EnsembleSampler
    first_argument = nwalkers
    sampler = sampler(first_argument, ndim, logPostBiasedDelta, args=(dNk, n_chrom, n_cells))
    sampler.run_mcmc(starting_positions, nsteps)
    
    #df = pd.DataFrame(np.vstack(sampler.chain))
    #df.index = pd.MultiIndex.from_product([range(nwalkers), range(nsteps)], names=['walker', 'step'])
    #df.columns = ['p_misseg', 'p_left']
    return sampler



def burnInPlotAffine(nTheta, sampler):
    '''
    function that outputs a plot of all of the simulation parameters
    in time
    
    Inputs
    __________
    
    nTheta : int-like
        number of dimensions of the mcmc
    sampler : emcee-EnsembleSampler-like
        emcee output from sampling
        
        
    
    Outputs
    __________
    
    '''
    fig, axs = plt.subplots(nTheta)
    xVals = range(len(sampler.chain[0]))
    for i in range(len(sampler.chain)):
        for j in range(nTheta):
            sns.lineplot(xVals, sampler.chain[i,:,j], ax=axs[j])


def delete_burn_in(sampler, n_pts, n_dim):
    """
    deletes burn in samples from emcee sampler
    inputs
        sampler: emcee sampler object
        n_pts : number of points to cut off from beginning, int
        n_dim : number of parameters, int
    output
        df : dataframe containing burned in results
    """
    samples = sampler.chain[:,n_pts:,:]
    traces = samples.reshape(-1, n_dim).T
    df = pd.DataFrame({'p_misseg': traces[0], 'p_left': traces[1]})
    return df
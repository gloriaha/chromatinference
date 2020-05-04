import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import seaborn as sns
from error_correction.model import *

def emcee_fit(data, pos0, model, nwalkers=50, nsteps=2000):
    """Runs ensemble MCMC on biased independent segregation model.

    Parameters
    ----------
    data : SyntheticData or GenerateData-like object
        data : Pandas dataframe
        params : dictionary with parameter values
    pos0 : list 
      initial position
    model : str
        model name (unbiased, biased, unbiased_noisy, biased_noisy)
    nwalkers : int
        number of MCMC walkers
    nsteps : int
        number of MCMC steps

    Returns
    -------
    emcee sampler
        sampler emcee results
    """
    
    # unpack parameters and data
    n_chrom = data.params['n_chrom']
    dNk = data.data['dNk'].values
    N1s = data.data['N_1_w_noise'].values
    N2s = data.data['N_2_w_noise'].values
    ndim = len(pos0)
    
    # set up dictionary for model posteriors and arguments
    model_dict = {'unbiased':[logPostUnbiasedDelta, (dNk, n_chrom)], 
                  'biased':[logPostBiasedDelta, (dNk, n_chrom)], 
                  'unbiased_noisy':[logPostUnbiasedNoisy, (N1s, N2s, n_chrom)], 
                  'biased_noisy':[logPostBiasedNoisy, (N1s, N2s, n_chrom)]}
    
    # make sure model name is valid
    if model not in model_dict.keys():
        raise ValueError('model must be unbiased, biased, unbiased_noisy, or biased_noisy')
        
    # set up starting positions
    gaussian_ball = 1.e-3 * np.random.randn(nwalkers, ndim)
    starting_positions = (1 + gaussian_ball) * pos0
    
    # run sampler
    sampler = emcee.EnsembleSampler
    first_argument = nwalkers
    sampler = sampler(first_argument, ndim, model_dict[model][0],
                      args=model_dict[model][1])
    # print progress
    for i, result in enumerate(sampler.sample(starting_positions, iterations=nsteps)):
        if (i + 1) % 100 == 0:
            print("{0:5%}".format(float(i + 1) / nsteps))

    return sampler


def burnInPlotAffine(nTheta, sampler, param_names):
    """Function that outputs a plot of all of the simulation parameters
    in time

    Parameters
    ----------
    nTheta : int
        number of dimensions of the mcmc
    sampler : emcee-EnsembleSampler-like
        emcee output from sampling
    param_names : list
        parameter names, strings
        
    Returns
    -------
    axs : axes
        axes of plots

    """
    fig, axs = plt.subplots(nTheta)
    xVals = range(len(sampler.chain[0]))
    for i in range(len(sampler.chain)):
        for j in range(nTheta):
            sns.lineplot(xVals, sampler.chain[i,:,j], ax=axs[j])
    for j in range(nTheta):
        axs[j].set_ylabel(param_names[j])
        axs[j].set_xlabel('steps')
    return axs

def delete_burn_in(sampler, n_pts, n_dim, param_names):
    """Deletes burn in samples from emcee sampler

    Parameters
    ----------
    sampler : emcee-EnsembleSampler-like
        emcee sampler object
    n_pts : int
        number of points to cut off from beginning
    n_dim : int
        number of parameters
    param_names : list
        parameter names, strings

    Returns
    -------
    Numpy DataFrame
        dataframe containing burned in results

    """
    samples = sampler.chain[:, n_pts:, :]
    traces = samples.reshape(-1, n_dim).T
    sample_dict = {name:trace for name, trace in zip(param_names, traces)}
    df = pd.DataFrame(sample_dict)
    return df

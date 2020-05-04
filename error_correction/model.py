import numpy as np
import pandas as pd
from scipy import special
from scipy.stats import binom

# A bunch of functions that will help us construct likelihoods

def probN1(p, alpha, N1, N):
    '''
    Compute the probability of N1 chromatids in daughter cell 1
    given N chromosomes, bias alpha, and missegregation probability p

    Parameters
    __________

    p : float-like
        probability of missegregation
    alpha : float-like
        bias
    N1 : ndarray
        number of chromatids in daughter cell 1
    N : int-like
        number of chromosomes

        

    Returns
    __________

    pN1 : ndarray
        probabilities of N1 chromosomes
    '''
    # Array of number of possible missegregations, corresponding meshgrid
    m = np.arange(N+1)
    mm, nn = np.meshgrid(m, N1)
    
    # Groups of binomial probabilities
    binoms1 = binom.pmf(mm, N, p)
    binoms2 = binom.pmf((nn-N+mm)/2., mm, alpha)
    
    # multiply and sum everything up
    pN1 = np.sum(binoms1*binoms2, axis=1)

    return pN1


def probN1N2Noisy(p, alpha, pd, N1s, N2s, N):
    '''
    Function that returns the probability of N1 chromatids in cell 1
    and N2 chromatids in cell 2

    Parameters
    __________
    
    p : float-like
        probability of missegregation
    alpha : float-like
        chromosome missegregation bias
    pd : float-like
        kinetochore detection probability
    N1 : ndarray
        number of kinetochores in daughter cell 1
    N2 : ndarray
        number of kinetochores in daughter cell 2
    N : int-like
        number of chromosomes

    Returns
    __________

    probN1N2 : ndarray
        probabilities of each N1 and N2 pair
    '''
    # set up all possible values of N1Tilde
    N1_min = np.min(N1s)
    N1T = np.arange(N1_min, 2*N+1-np.min(N2s))

    # calculate the probabilities of these N1T
    PN1T = probN1(p, alpha, N1T, N)

    # calculate all possible binomial values
    nn1, nn1t = np.meshgrid(N1s, N1T)
    nn2, nn2t = np.meshgrid(N2s, N1T)
    binom1 = binom.pmf(nn1, nn1t, pd)
    binom2 = binom.pmf(nn2, 2*N-nn2t, pd)
    binomPdt = binom1*binom2
    
    # calculate all possible products of two probabilities
    pn1tPdt = binomPdt*PN1T[:, np.newaxis]
    
    # mask out actual slices
    mask = np.zeros(pn1tPdt.shape)
    for i in range(len(N1s)):
        mask[N1s[i]-N1_min:2*N-N2s[i]+1-N1_min, i] = 1
        
    # compute probabilities for each N1, N2 pair
    probN1N2 = np.sum(pn1tPdt*mask, axis=0)

    return probN1N2


# Write down likelihood functions

def logLikeBiasedDelta(params, deltas, N):
    """Compute the likelihood of the difference in chromosomes
    within the biased missegregation model

    Parameters
    ----------
    params : list-like
        params = [p, alpha]
        p : probability of chromosome missegregation
        alpha : probability chromosome ends up in daughter cell 1
    deltas : ndarray
        kinetochore differences, from dNk column of data
    N : int
        number of chromosomes, from 'n_chrom' parameter in params file

    Returns
    -------
    float-like
        log likelihood function of the above

    """
    # Note the model parameters
    p, alpha = params

    # Calculate the likelihoods for N1 and N2
    likes = probN1(p, alpha, N+deltas/2, N)+probN1(p, alpha, N-deltas/2, N)
    
    # correct for overcounting for delta=0 cases
    likes -= likes/2*(deltas==0)

    # Take the logs, then sum, then return the log likelihood
    logLikelihood = np.sum(np.log(likes))

    return logLikelihood


def logLikeUnbiasedDelta(params, deltas, N):
    """Compute the likelihood of the difference in chromosomes
    within the biased missegregation model

    Parameters
    ----------
    params : list-like
        params = [p, alpha]
        p : probability of chromosome missegregation
    deltas : ndarray
        kinetochore differences, from dNk column of data
    N : int
        number of chromosomes, from 'n_chrom' parameter in params file

    Returns
    -------
    float-like
        log likelihood function of the above

    """
    # Note the model parameters
    p, = params

    # Repackage the model parameters
    params2 = [p, 0.5]
    logLikelihood = logLikeBiasedDelta(params2, deltas, N)

    return logLikelihood



def logLikeBiasedNoisy(params, N1s, N2s, N):
    '''
    Compute the likelihood of the difference in chromosomes
    within the biased missegregation model

    Parameters
    ----------
    params : list-like
        params = [p, alpha, pd]
        p : probability of chromosome missegregation
        alpha : probability chromosome ends up in daughter cell 1
        pd : probability of chomatid detection
    N1s : ndarray
        array with number of kinetochores in cell 1
    N2s : ndarray
        array with number of kinetochores in cell 2
    N : int
        number of chromosomes, from 'n_chrom' parameter in params file

    Returns
    -------
    float-like
        log likelihood function of the above
    '''
    
    # Note the model parameters
    p, alpha, pd = params

    # Calculate the likelihood for alpha and 1-alpha (symmetry)
    likes1 = probN1N2Noisy(p, alpha, pd, N1s, N2s, N)
    likes2 = probN1N2Noisy(p, 1.-alpha, pd, N1s, N2s, N)
    likes = (likes1+likes2)/2.

    # Take the logs, then sum, then return the log likelihood
    logLikelihood = np.sum(np.log(likes))

    return logLikelihood


    return logLikelihood
def logLikeUnbiasedNoisy(params, N1s, N2s, N):
    '''
    Compute the likelihood of the difference in chromosomes
    within the unbiased missegregation model

    Parameters
    ----------
    params : list-like
        params = [p, pd]
        p : probability of chromosome missegregation
        pd : probability of chomatid detection
    N1s : ndarray
        array with number of kinetochores in cell 1
    N2s : ndarray
        array with number of kinetochores in cell 2
    N : int
        number of chromosomes, from 'n_chrom' parameter in params file

    Returns
    -------
    float-like
        log likelihood function of the above
    '''
    # Note the model parameters
    p, pd = params
    #params2 = [p, 0.5, pd]

    # Call the biased likelihood function but with alpha = 0.5
    #logLikelihood = logLikeBiasedNoisy(params2, N1s, N2s, N)
    likes = probN1N2Noisy(p, 0.5, pd, N1s, N2s, N)
    logLikelihood = np.sum(np.log(likes))

    return logLikelihood


# Write down priors
def logPriorBiasedDelta(params):
    """Compute prior for biased model

    Parameters
    ----------
    params : list
        parameters

    Returns
    -------
    float
        log prior (unnormalized)

    """
    # unpack the model parameters
    p, alpha = params

    # make sure parameters are within bounds
    if p < 0. or p > 1. or alpha < 0. or alpha > 0.5:
        return -np.inf
    else:
        return 0.

def logPriorUnbiasedDelta(params):
    """Compute prior for unbiased model

    Parameters
    ----------
    params : list
        parameters

    Returns
    -------
    float
        log prior (unnormalized)

    """
    # unpack the model parameters
    p, = params
    params2 = [p, 0.5]

    return logPriorBiasedDelta(params2)



def logPriorBiasedNoisy(params):
    """Compute prior for biased noisy model

    Parameters
    ----------
    params : list
        parameters

    Returns
    -------
    float
        log prior (unnormalized)

    """
    # unpack the model parameters
    p, alpha, pd = params

    # make sure parameters are within bounds
    if p < 0. or p > 1. or alpha < 0. or alpha > 0.5 or pd < 0. or pd > 1.:
        return -np.inf
    else:
        return 0


def logPriorUnbiasedNoisy(params):
    """Compute prior for unbiased noisy model.

    Parameters
    ----------
    params : list
        parameters

    Returns
    -------
    float
        log prior (unnormalized)

    """
    # unpack the model parameters
    p, pd = params
    params2 = [p, 0.5, pd]

    return logPriorBiasedNoisy(params2)


# Write down posteriors


def logPostBiasedDelta(params, deltas, N):
    """Compute the posterior of the difference in chromosomes
    within the biased missegregation model

    Parameters
    ----------
    params : list-like
        params = [p, alpha]
        p : probability of chromosome missegregation
        alpha : probability chromosome ends up in daughter cell 1
    deltas : ndarray
        kinetochore differences, from dNk column of data
    N : int
        number of chromosomes, from 'n_chrom' parameter in params file

    Returns
    -------
    float-like
        log posterior function of the above

    """
    # check that prior is finite
    if logPriorBiasedDelta(params) == -np.inf:
        return -np.inf
    else:
        logPost = logLikeBiasedDelta(params, deltas, N) + logPriorBiasedDelta(params)
        return logPost



def logPostUnbiasedDelta(params, deltas, N):
    """Compute the posterior of the difference in chromosomes
    within the biased missegregation model

    Parameters
    ----------
    params : list-like
        params = [p]
        p : probability of chromosome missegregation
    deltas : ndarray
        kinetochore differences, from dNk column of data
    N : int
        number of chromosomes, from 'n_chrom' parameter in params file

    Returns
    -------
    float-like
        log posterior function of the above

    """
    # check that prior is finite
    if logPriorUnbiasedDelta(params) == -np.inf:
        return -np.inf
    else:
        logPost = logLikeUnbiasedDelta(params, deltas, N) + logPriorUnbiasedDelta(params)
        return logPost



def logPostBiasedNoisy(params, N1s, N2s, N):
    '''
    Compute the posterior of the difference in chromosomes
    within the biased missegregation model

    Parameters
    ----------
    params : list-like
        params = [p, alpha]
        p : probability of chromosome missegregation
        alpha : probability chromosome ends up in daughter cell 1
        pd : probability of chromatid detection
    N1s : ndarray
        array with number of kinetochores in cell 1
    N2s : ndarray
        array with number of kinetochores in cell 2
    N : int
        number of chromosomes, from 'n_chrom' parameter in params file

    Returns
    -------
    float-like
        log posterior function of the above
    '''
    # check that prior is finite
    if logPriorBiasedNoisy(params) == -np.inf:
        return -np.inf
    else:
        logPost = logLikeBiasedNoisy(params, N1s, N2s, N) + logPriorBiasedNoisy(params)
        return logPost



def logPostUnbiasedNoisy(params, N1s, N2s, N):
    '''
    Compute the posterior of the difference in chromosomes
    within the biased missegregation model

    Parameters
    ----------
    params : list-like
        params = [p, alpha]
        p : probability of chromosome missegregation
        pd : probability of chromatid detection
    N1s : ndarray
        array with number of kinetochores in cell 1
    N2s : ndarray
        array with number of kinetochores in cell 2
    N : int
        number of chromosomes, from 'n_chrom' parameter in params file

    Returns
    -------
    float-like
        log posterior function of the above
    '''
    # check that prior is finite
    if logPriorUnbiasedNoisy(params) == -np.inf:
        return -np.inf
    else:
        logPost = logLikeUnbiasedNoisy(params, N1s, N2s, N) + logPriorUnbiasedNoisy(params)
        return logPost
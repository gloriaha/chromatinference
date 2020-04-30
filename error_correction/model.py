import numpy as np
import pandas as pd
from scipy import special
from scipy.stats import binom







# A bunch of functions that will help us construct likelihoods
# ___________________________________________________________________
# ___________________________________________________________________
# ___________________________________________________________________
# ___________________________________________________________________
# ___________________________________________________________________



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
    N1 : int-like
        number of chromatids in daughter cell 1
    N : int-like
        number of chromosomes

        

    Returns
    __________

    pN1 : float-like
        probability of N1 chromosomes
    '''
    # Array of number of possible missegregations
    m = np.arange(N+1)

    # Groups of binomial probabilities
    binoms1 = binom.pmf(m, N, p)
    binoms2 = binom.pmf((N1-N+m)/2., m, alpha)

    pN1 = sum(binoms1*binoms2)

    return pN1



def probDelta(p, alpha, delta, N):
    '''
    Compute the probability of N1 chromatids in daughter cell 1
    given N chromosomes, bias alpha, and missegregation probability p

    Parameters
    __________

    p : float-like
        probability of missegregation
    alpha : float-like
        bias
    N : int-like
        number of chromosomes
    delta : int-like
        difference in chromatids between daughter cells

        

    Returns
    __________

    pDelta : float-like
        probability of delta difference of chromosomes
    '''
    # Note that N1 = N(+/-)delta/2
    if delta == 0:
        pDelta = probN1(p, alpha, N, N)
    if delta != 0:
        pDelta = probN1(p, alpha, N+delta/2, N)+probN1(p, alpha, N-delta/2, N)

    return pDelta



def binomialProbsNoisy(pd, N1, N2, N):
    '''
    Function that returns the binomial probabilities needed to sum
    over in order to get the likelihood functions with experimental
    noise (false negatives in kinetochore detection)

    Parameters
    __________

    pd : float-like
        kinetochore detection probability
    N : int-like
        number of chromosomes
    Nd : array-like
        Nd = np.array([N1, N2])
        N1 : number of kinetochores in daughter cell 1
        N2 : number of kinetochores in daughter cell 2

    Returns
    __________

    binomArray : array-like
        array of binomial probabilities
    '''
    
    # Write down the allowed values of N1Tilde
    N1TildeVals = np.arange(N1, 2*N-N2+1)

    # Write down the binomial probabilities
    binom1 = binom.pmf(N1, N1TildeVals, pd)
    binom2 = binom.pmf(N2, 2*N-N1TildeVals, pd)

    binomArray = binom1*binom2

    return binomArray



def probN1N2Noisy(p, alpha, pd, N1, N2, N):
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
    N1 : array-like
        number of kinetochores in daughter cell 1
    N2 : array-like
        number of kinetochores in daughter cell 2
    N : int-like
        number of chromosomes

    Returns
    __________

    probN1N2 : float-like
        probability of N1 and N2
    '''

    # Calculate the allowed values of N1Tilde
    N1T = np.arange(N1, 2*N-N2+1)

    # Calculate the probabilities of these values
    vectorizedPN1 = np.vectorize(probN1)
    PN1T = vectorizedPN1(p, alpha, N1T, N)

    # Calculate the appropriate binomial values
    binomVals = binomialProbsNoisy(pd, N1, N2, N)

    # Compute the sum of the product of the above
    probN1N2 = sum(binomVals*PN1T)

    return probN1N2







# Write down likelihood functions
# ___________________________________________________________________
# ___________________________________________________________________
# ___________________________________________________________________
# ___________________________________________________________________
# ___________________________________________________________________



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

    # Calculate the likelihood
    # First, vectorize it
    vectorizedLikelihood = np.vectorize(probDelta)
    likes = vectorizedLikelihood(p, alpha, deltas, N)

    # Take the logs, then sum, then return the log likelihood
    logLikelihood = sum(np.log(likes))

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

    # Calculate the likelihood
    # First, vectorize it
    vectorizedLikelihood = np.vectorize(probN1N2Noisy)
    likes1 = vectorizedLikelihood(p, alpha, pd, N1s, N2s, N)
    likes2 = vectorizedLikelihood(p, 1.-alpha, pd, N1s, N2s, N)
    likes = (likes1+likes2)/2.

    # Take the logs, then sum, then return the log likelihood
    logLikelihood = sum(np.log(likes))

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
    params2 = [p, 0.5, pd]

    # Call the biased likelihood function but with alpha = 0.5
    logLikelihood = logLikeBiasedNoisy(params2, N1s, N2s, N)

    return logLikelihood







# Write down priors
# ___________________________________________________________________
# ___________________________________________________________________
# ___________________________________________________________________
# ___________________________________________________________________
# ___________________________________________________________________



def logPriorBiasedDelta(params):
    """Short summary.

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
    """Short summary.

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
    """Short summary.

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
    """Short summary.

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
# ___________________________________________________________________
# ___________________________________________________________________
# ___________________________________________________________________
# ___________________________________________________________________
# ___________________________________________________________________



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
        params = [p, alpha]
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



def logPostBiasedNoisy(params, deltas, N):
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



def logPostUnbiasedNoisy(params, deltas, N):
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







# Legacy Code
# ___________________________________________________________________
# ___________________________________________________________________
# ___________________________________________________________________
# ___________________________________________________________________
# ___________________________________________________________________



##def truncateBinom(ns, ms):
##    """Compute the binomial coefficient n choose m, but return
##    zero if the output isn't integer (i.e. if n or m isn't
##    integer)
##
##    Parameters
##    ----------
##    ns : array-like
##        values of n
##    ms : array-like
##        values of m
##
##    Returns
##    -------
##    array-like
##        truncateBinomCoeff - integer-only binomial coefficients n choose m
##
##    """
##    # Calculate binomials
##    binomCoeff = special.binom(ns, ms)
##
##    # Calculate the truncation
##    digitalArray = np.mod(binomCoeff, 1.) == 0.
##
##    # Calculate the product of the above
##    truncateBinomCoeff = binomCoeff * digitalArray
##
##    return truncateBinomCoeff



##def logLikeIndDelta(data, params):
##    """Compute the likelihood of the difference in chromosomes
##    within the independent segregation model
##
##    Parameters
##    ----------
##    data : SyntheticData-like
##        class containing .params, .data, and .load_data with
##        .params a dictionary containing n_cells, n_chrom, p_misseg;
##        .data a dataframe containing errors, dNk, and dNk_w_noise
##    params : list-like
##        model parameters; here, only p -- the probability of missegregation
##
##    Returns
##    -------
##    float-like
##        log likelihood function of the above
##
##    """
##    # Note the model parameters
##    p, = params
##
##    # Extract the parameters from the data.params object
##    nCells = data.params['n_cells']
##    N = data.params['n_chrom']
##
##    # Extract the difference in chromatids from data.data
##    deltas = data.data['dNk'].values
##
##    # Calculate the likelihood
##    mVals = np.arange(N + 1)
##    mValsNew = np.repeat(mVals, len(deltas))
##    deltasNew = np.tile(deltas, len(mVals))
##
##    binom1 = truncateBinom(N, mValsNew)
##    binom2 = truncateBinom(mValsNew, (mValsNew - deltasNew / 2.) / 2.)
##    binom3 = truncateBinom(mValsNew, (mValsNew + deltasNew / 2.) / 2.)
##    likes1 = binom1 * binom2 * p**mValsNew * \
##        (1. - p)**(N - mValsNew) * 2.**(-mValsNew)
##    likes2 = binom1 * binom3 * p**mValsNew * \
##        (1. - p)**(N - mValsNew) * 2.**(-mValsNew)
##
##    likes = (likes1 + likes2).reshape([len(mVals), len(deltas)])
##    likesSum = likes.sum(axis=0)
##    logLikelihood = np.sum(np.log(likesSum))
##
##    return logLikelihood


##def logLikeBiasedDelta(params, deltas, N, nCells):
##    """Compute the likelihood of the difference in chromosomes
##    within the biased missegregation model
##
##    Parameters
##    ----------
##    params : list-like
##        params = [p, alpha]
##        p : probability of chromosome missegregation
##        alpha : probability chromosome ends up in daughter cell 1
##    deltas : ndarray
##        kinetochore differences, from dNk column of data
##    N : int
##        number of chromosomes, from 'n_chrom' parameter in params file
##    nCells : int
##        number of cells, from 'n_cells' parameter in params file
##
##    Returns
##    -------
##    float-like
##        log likelihood function of the above
##
##    """
##    # Write down the model parameters
##    p, alpha = params
##
##    # Calculate the likelihood
##    mVals = np.arange(N + 1)
##    mValsNew = np.repeat(mVals, len(deltas))
##    deltasNew = np.tile(deltas, len(mVals))
##
##    binom1 = truncateBinom(N, mValsNew)
##    binom2 = truncateBinom(mValsNew, (mValsNew - deltasNew / 2.) / 2.)
##    binom3 = truncateBinom(mValsNew, (mValsNew + deltasNew / 2.) / 2.)
##
##    likes1 = binom1 * binom2 * p**mValsNew * (1. - p)**(N - mValsNew) * alpha **\
##        ((mValsNew - deltasNew / 2.) / 2.) * \
##        (1. - alpha)**((mValsNew + deltasNew / 2.) / 2.)
##    likes2 = binom1 * binom3 * p**mValsNew * (1. - p)**(N - mValsNew) * alpha **\
##        ((mValsNew + deltasNew / 2.) / 2.) * \
##        (1. - alpha)**((mValsNew - deltasNew / 2.) / 2.)
##
##    likes = (likes1 + likes2).reshape([len(mVals), len(deltas)])
##    likesSum = likes.sum(axis=0)
##    logLikelihood = np.sum(np.log(likesSum))
##
##    return logLikelihood
##
##
##def logPostBiasedDelta(params, deltas, N, nCells):
##    """Compute the posterior of the difference in chromosomes
##    within the biased missegregation model
##
##    Parameters
##    ----------
##    params : list-like
##        params = [p, alpha]
##        p : probability of chromosome missegregation
##        alpha : probability chromosome ends up in daughter cell 1
##    deltas : ndarray
##        kinetochore differences, from dNk column of data
##    N : int
##        number of chromosomes, from 'n_chrom' parameter in params file
##    nCells : int
##        number of cells, from 'n_cells' parameter in params file
##
##    Returns
##    -------
##    float-like
##        log posterior function of the above
##
##    """
##    # check that prior is finite
##    if logPriorBiasedDelta(params) == -np.inf:
##        return -np.inf
##    logPost = logLikeBiasedDelta(
##        params, deltas, N, nCells) + logPriorBiasedDelta(params)
##    return logPost


##def logLikeGivenCatDelta(data, params):
##    """Compute the likelihood of the difference in chromosomes
##    within the catastrophe model given catastrophe
##
##    Parameters
##    ----------
##    data : SyntheticData-like
##        class containing .params, .data, and .load_data with
##        .params a dictionary containing n_cells, n_chrom, p_misseg;
##        .data a dataframe containing errors, dNk, and dNk_w_noise
##    params : list-like
##        params = [p, C]
##        p : probability of chromosome missegregation
##        C : number of catastrophic chromosomes
##
##    Returns
##    -------
##    float-like
##        log likelihood function of the above
##
##    """
##
##    # Note the model parameters
##    p, C = params
##
##    # Extract the parameters from the data.params object
##    nCells = data.params['n_cells']
##    N = data.params['n_chrom']
##
##    # Extract the difference in chromatids from data.data
##    deltas = data.data['dNk'].values
##
##    # Calculate the likelihood
##    mVals = np.arange(N - C + 1)
##    mValsNew = np.repeat(mVals, len(deltas))
##    deltasNew = np.tile(deltas, len(mVals))
##
##    binom1 = truncateBinom(N, mValsNew)
##    binom2 = truncateBinom(mValsNew, (mValsNew + C - deltasNew / 2.) / 2.)
##    binom3 = truncateBinom(mValsNew, (mValsNew + C + deltasNew / 2.) / 2.)
##
##    likes1 = binom1 * binom2 * p**mValsNew * \
##        (1. - p)**(N - C - mValsNew) * 2.**(-mValsNew - C)
##    likes2 = binom1 * binom3 * p**mValsNew * \
##        (1. - p)**(N - C - mValsNew) * 2.**(-mValsNew - C)
##
##    likes = (likes1 + likes2).reshape([len(mVals), len(deltas)])
##    likesSum = likes.sum(axis=0)
##    logLikelihood = np.sum(np.log(likesSum))
##
##    return logLikelihood
##
##
##def logLikeCatDelta(data, params):
##    """Compute the likelihood of the difference in chromosomes
##    within the catastrophe model
##
##    Parameters
##    ----------
##    data : SyntheticData-like
##        class containing .params, .data, and .load_data with
##        .params a dictionary containing n_cells, n_chrom, p_misseg;
##        .data a dataframe containing errors, dNk, and dNk_w_noise
##    params : list
##        params = [p, C, pCat]
##        p : probability of chromosome missegregation
##        C : number of catastrophic chromosomes
##        pCat : probability of catastrophe
##
##    Returns
##    -------
##    float-like
##        log likelihood function of the above
##
##    """
##    # Note the model parameters
##    p, C, pCat = params
##
##    # Write down the log likelihood given catastrophe
##    logLikeGivenCat = logLikeGivenCatDelta(data, [p, C])
##
##    # Write down the log likelihood with no catastrophe
##    logLikeNoCat = logLikeIndDelta(data, [p])
##
##    # Write down the full likelihood function
##    expLikeRat = np.exp(logLikeGivenCat - logLikeNoCat)
##    logLikelihood = logLikeNoCat + np.log(pCat * expLikeRat + 1 - pCat)
##
##    return logLikelihood
##
##
##def logLikeGivenCatBiasedDelta(data, params):
##    """
##    Compute the likelihood of the difference in chromosomes
##    within the biased missegregation model given catastrophe
##
##    Parameters
##    ----------
##    data : SyntheticData-like
##        class containing .params, .data, and .load_data with
##        .params a dictionary containing n_cells, n_chrom, p_misseg;
##        .data a dataframe containing errors, dNk, and dNk_w_noise
##    params : list-like
##        params = [p, C, alpha]
##        p : probability of chromosome missegregation
##        C : number of catastrophic chromosomes
##        alpha : probability chromosome ends up in daughter cell 1
##
##    Returns
##    -------
##    float-like
##        log likelihood function of the above
##
##    """
##    # Note the model parameters
##    p, C, alpha = params
##
##    # Extract the parameters from the data.params object
##    nCells = data.params['n_cells']
##    N = data.params['n_chrom']
##
##    # Extract the difference in chromatids from data.data
##    deltas = data.data['dNk'].values
##
##    # Calculate the likelihood
##    mVals = np.arange(N - C + 1)
##    mValsNew = np.repeat(mVals, len(deltas))
##    deltasNew = np.tile(deltas, len(mVals))
##
##    binom1 = truncateBinom(N, mValsNew)
##    binom2 = truncateBinom(mValsNew, (mValsNew + C - deltasNew / 2.) / 2.)
##    binom3 = truncateBinom(mValsNew, (mValsNew + C + deltasNew / 2.) / 2.)
##
##    likes1 = binom1 * binom2 * p**mValsNew * (1. - p)**(N - C - mValsNew) * alpha **\
##        ((mValsNew + C - deltasNew / 2.) / 2.) * \
##        (1. - alpha)**((mValsNew - C + deltasNew / 2.) / 2.)
##    likes2 = binom1 * binom3 * p**mValsNew * (1. - p)**(N - C - mValsNew) * alpha **\
##        ((mValsNew + C + deltasNew / 2.) / 2.) * \
##        (1. - alpha)**((mValsNew - C - deltasNew / 2.) / 2.)
##
##    likes = (likes1 + likes2).reshape([len(mVals), len(deltas)])
##    likesSum = likes.sum(axis=0)
##    logLikelihood = np.sum(np.log(likesSum))
##
##    return logLikelihood
##
##
##def logLikeCatBiasedDelta(data, params):
##    """Compute the likelihood of the difference in chromosomes
##    within the catastrophe and biased missegregation model
##
##    Parameters
##    ----------
##    data : SyntheticData-like
##        class containing .params, .data, and .load_data with
##        .params a dictionary containing n_cells, n_chrom, p_misseg;
##        .data a dataframe containing errors, dNk, and dNk_w_noise
##
##    params : list
##        params = [p, C, pCat, alpha]
##        p : probability of chromosome missegregation
##        C : number of catastrophic chromosomes
##        pCat : probability of catastrophe
##        alpha : missegregation bias (0.5 for unbiased)
##
##    Returns
##    -------
##    float-like
##        log likelihood function of the above
##
##    """
##    # Note the model parameters
##    p, C, pCat, alpha = params
##
##    # Write down the log likelihood given catastrophe
##    logLikeGivenCat = logLikeGivenCatBiasedDelta(data, [p, C, alpha])
##
##    # Write down the log likelihood with no catastrophe
##    logLikeNoCat = logLikeBiasedDelta(
##        [p, 0.5], data.data['dNk'], data.params['n_chrom'], data.params['n_cells'])
##
##    # Write down the full likelihood function
##    expLikeRat = np.exp(logLikeGivenCat - logLikeNoCat)
##    logLikelihood = logLikeNoCat + np.log(pCat * expLikeRat + 1 - pCat)
##
##    return logLikelihood

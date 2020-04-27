import numpy as np
import pandas as pd
from scipy import special


def logPriorBiasedDelta(params):
    """Computes log prior for biased segregation model.

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
    if p < 0. or p > 1.:
        return -np.inf
    elif alpha < 0. or alpha > 0.5:
        return -np.inf
    return 0


def truncateBinom(ns, ms):
    """Compute the binomial coefficient n choose m, but return
    zero if the output isn't integer (i.e. if n or m isn't
    integer)

    Parameters
    ----------
    ns : array-like
        values of n
    ms : array-like
        values of m

    Returns
    -------
    array-like
        truncateBinomCoeff - integer-only binomial coefficients n choose m

    """
    # Calculate binomials
    binomCoeff = special.binom(ns, ms)

    # Calculate the truncation
    digitalArray = np.mod(binomCoeff, 1.) == 0.

    # Calculate the product of the above
    truncateBinomCoeff = binomCoeff * digitalArray

    return truncateBinomCoeff


def logLikeIndDelta(data, params):
    """Compute the likelihood of the difference in chromosomes
    within the independent segregation model

    Parameters
    ----------
    data : SyntheticData-like
        class containing .params, .data, and .load_data with
        .params a dictionary containing n_cells, n_chrom, p_misseg;
        .data a dataframe containing errors, dNk, and dNk_w_noise
    params : list-like
        model parameters; here, only p -- the probability of missegregation

    Returns
    -------
    float-like
        log likelihood function of the above

    """
    # Note the model parameters
    p, = params

    # Extract the parameters from the data.params object
    nCells = data.params['n_cells']
    N = data.params['n_chrom']

    # Extract the difference in chromatids from data.data
    deltas = data.data['dNk'].values

    # Calculate the likelihood
    mVals = np.arange(N + 1)
    mValsNew = np.repeat(mVals, len(deltas))
    deltasNew = np.tile(deltas, len(mVals))

    binom1 = truncateBinom(N, mValsNew)
    binom2 = truncateBinom(mValsNew, (mValsNew - deltasNew / 2.) / 2.)
    binom3 = truncateBinom(mValsNew, (mValsNew + deltasNew / 2.) / 2.)
    likes1 = binom1 * binom2 * p**mValsNew * \
        (1. - p)**(N - mValsNew) * 2.**(-mValsNew)
    likes2 = binom1 * binom3 * p**mValsNew * \
        (1. - p)**(N - mValsNew) * 2.**(-mValsNew)

    likes = (likes1 + likes2).reshape([len(mVals), len(deltas)])
    likesSum = likes.sum(axis=0)
    logLikelihood = np.sum(np.log(likesSum))

    return logLikelihood


def logLikeBiasedDelta(params, deltas, N, nCells):
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
    nCells : int
        number of cells, from 'n_cells' parameter in params file

    Returns
    -------
    float-like
        log likelihood function of the above

    """
    # Write down the model parameters
    p, alpha = params

    # Calculate the likelihood
    mVals = np.arange(N + 1)
    mValsNew = np.repeat(mVals, len(deltas))
    deltasNew = np.tile(deltas, len(mVals))

    binom1 = truncateBinom(N, mValsNew)
    binom2 = truncateBinom(mValsNew, (mValsNew - deltasNew / 2.) / 2.)
    binom3 = truncateBinom(mValsNew, (mValsNew + deltasNew / 2.) / 2.)

    likes1 = binom1 * binom2 * p**mValsNew * (1. - p)**(N - mValsNew) * alpha **\
        ((mValsNew - deltasNew / 2.) / 2.) * \
        (1. - alpha)**((mValsNew + deltasNew / 2.) / 2.)
    likes2 = binom1 * binom3 * p**mValsNew * (1. - p)**(N - mValsNew) * alpha **\
        ((mValsNew + deltasNew / 2.) / 2.) * \
        (1. - alpha)**((mValsNew - deltasNew / 2.) / 2.)

    likes = (likes1 + likes2).reshape([len(mVals), len(deltas)])
    likesSum = likes.sum(axis=0)
    logLikelihood = np.sum(np.log(likesSum))

    return logLikelihood


def logPostBiasedDelta(params, deltas, N, nCells):
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
    nCells : int
        number of cells, from 'n_cells' parameter in params file

    Returns
    -------
    float-like
        log posterior function of the above

    """
    # check that prior is finite
    if logPriorBiasedDelta(params) == -np.inf:
        return -np.inf
    logPost = logLikeBiasedDelta(
        params, deltas, N, nCells) + logPriorBiasedDelta(params)
    return logPost


def logLikeGivenCatDelta(data, params):
    """Compute the likelihood of the difference in chromosomes
    within the catastrophe model given catastrophe

    Parameters
    ----------
    data : SyntheticData-like
        class containing .params, .data, and .load_data with
        .params a dictionary containing n_cells, n_chrom, p_misseg;
        .data a dataframe containing errors, dNk, and dNk_w_noise
    params : list-like
        params = [p, C]
        p : probability of chromosome missegregation
        C : number of catastrophic chromosomes

    Returns
    -------
    float-like
        log likelihood function of the above

    """

    # Note the model parameters
    p, C = params

    # Extract the parameters from the data.params object
    nCells = data.params['n_cells']
    N = data.params['n_chrom']

    # Extract the difference in chromatids from data.data
    deltas = data.data['dNk'].values

    # Calculate the likelihood
    mVals = np.arange(N - C + 1)
    mValsNew = np.repeat(mVals, len(deltas))
    deltasNew = np.tile(deltas, len(mVals))

    binom1 = truncateBinom(N, mValsNew)
    binom2 = truncateBinom(mValsNew, (mValsNew + C - deltasNew / 2.) / 2.)
    binom3 = truncateBinom(mValsNew, (mValsNew + C + deltasNew / 2.) / 2.)

    likes1 = binom1 * binom2 * p**mValsNew * \
        (1. - p)**(N - C - mValsNew) * 2.**(-mValsNew - C)
    likes2 = binom1 * binom3 * p**mValsNew * \
        (1. - p)**(N - C - mValsNew) * 2.**(-mValsNew - C)

    likes = (likes1 + likes2).reshape([len(mVals), len(deltas)])
    likesSum = likes.sum(axis=0)
    logLikelihood = np.sum(np.log(likesSum))

    return logLikelihood


def logLikeCatDelta(data, params):
    """Compute the likelihood of the difference in chromosomes
    within the catastrophe model

    Parameters
    ----------
    data : SyntheticData-like
        class containing .params, .data, and .load_data with
        .params a dictionary containing n_cells, n_chrom, p_misseg;
        .data a dataframe containing errors, dNk, and dNk_w_noise
    params : list
        params = [p, C, pCat]
        p : probability of chromosome missegregation
        C : number of catastrophic chromosomes
        pCat : probability of catastrophe

    Returns
    -------
    float-like
        log likelihood function of the above

    """
    # Note the model parameters
    p, C, pCat = params

    # Write down the log likelihood given catastrophe
    logLikeGivenCat = logLikeGivenCatDelta(data, [p, C])

    # Write down the log likelihood with no catastrophe
    logLikeNoCat = logLikeIndDelta(data, [p])

    # Write down the full likelihood function
    expLikeRat = np.exp(logLikeGivenCat - logLikeNoCat)
    logLikelihood = logLikeNoCat + np.log(pCat * expLikeRat + 1 - pCat)

    return logLikelihood


def logLikeGivenCatBiasedDelta(data, params):
    """
    Compute the likelihood of the difference in chromosomes
    within the biased missegregation model given catastrophe

    Parameters
    ----------
    data : SyntheticData-like
        class containing .params, .data, and .load_data with
        .params a dictionary containing n_cells, n_chrom, p_misseg;
        .data a dataframe containing errors, dNk, and dNk_w_noise
    params : list-like
        params = [p, C, alpha]
        p : probability of chromosome missegregation
        C : number of catastrophic chromosomes
        alpha : probability chromosome ends up in daughter cell 1

    Returns
    -------
    float-like
        log likelihood function of the above

    """
    # Note the model parameters
    p, C, alpha = params

    # Extract the parameters from the data.params object
    nCells = data.params['n_cells']
    N = data.params['n_chrom']

    # Extract the difference in chromatids from data.data
    deltas = data.data['dNk'].values

    # Calculate the likelihood
    mVals = np.arange(N - C + 1)
    mValsNew = np.repeat(mVals, len(deltas))
    deltasNew = np.tile(deltas, len(mVals))

    binom1 = truncateBinom(N, mValsNew)
    binom2 = truncateBinom(mValsNew, (mValsNew + C - deltasNew / 2.) / 2.)
    binom3 = truncateBinom(mValsNew, (mValsNew + C + deltasNew / 2.) / 2.)

    likes1 = binom1 * binom2 * p**mValsNew * (1. - p)**(N - C - mValsNew) * alpha **\
        ((mValsNew + C - deltasNew / 2.) / 2.) * \
        (1. - alpha)**((mValsNew - C + deltasNew / 2.) / 2.)
    likes2 = binom1 * binom3 * p**mValsNew * (1. - p)**(N - C - mValsNew) * alpha **\
        ((mValsNew + C + deltasNew / 2.) / 2.) * \
        (1. - alpha)**((mValsNew - C - deltasNew / 2.) / 2.)

    likes = (likes1 + likes2).reshape([len(mVals), len(deltas)])
    likesSum = likes.sum(axis=0)
    logLikelihood = np.sum(np.log(likesSum))

    return logLikelihood


def logLikeCatBiasedDelta(data, params):
    """Compute the likelihood of the difference in chromosomes
    within the catastrophe and biased missegregation model

    Parameters
    ----------
    data : SyntheticData-like
        class containing .params, .data, and .load_data with
        .params a dictionary containing n_cells, n_chrom, p_misseg;
        .data a dataframe containing errors, dNk, and dNk_w_noise

    params : list
        params = [p, C, pCat, alpha]
        p : probability of chromosome missegregation
        C : number of catastrophic chromosomes
        pCat : probability of catastrophe
        alpha : missegregation bias (0.5 for unbiased)

    Returns
    -------
    float-like
        log likelihood function of the above

    """
    # Note the model parameters
    p, C, pCat, alpha = params

    # Write down the log likelihood given catastrophe
    logLikeGivenCat = logLikeGivenCatBiasedDelta(data, [p, C, alpha])

    # Write down the log likelihood with no catastrophe
    logLikeNoCat = logLikeBiasedDelta(
        [p, 0.5], data.data['dNk'], data.params['n_chrom'], data.params['n_cells'])

    # Write down the full likelihood function
    expLikeRat = np.exp(logLikeGivenCat - logLikeNoCat)
    logLikelihood = logLikeNoCat + np.log(pCat * expLikeRat + 1 - pCat)

    return logLikelihood

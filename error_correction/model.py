import numpy as np
import pandas as pd
from scipy import special



def truncateBinom(ns,ms):
    '''
    Compute the binomial coefficient n choose m, but return
    zero if the output isn't integer (i.e. if n or m isn't
    integer)

    Inputs
    __________

    ns : array-like
        values of n
    ms : array-like
        values of m


    Outputs
    __________

    truncateBinomCoeff : array-like
        integer-only binomial coefficients n choose m
    '''
    # Calculate binomials
    binomCoeff = special.binom(ns, ms)

    # Calculate the truncation
    digitalArray = np.mod(binomCoeff,1.)==0.

    # Calculate the product of the above
    truncateBinomCoeff = binomCoeff*digitalArray

    return truncateBinomCoeff
    


def logLikeIndDelta(data, params):
    '''
    Compute the likelihood of the difference in chromosomes
    within the independent segregation model
    
    Inputs
    __________
    
    data : SyntheticData-like
        class containing .params, .data, and .load_data with
        .params a dictionary containing n_cells, n_chrom, p_misseg;
        .data a dataframe containing errors, dNk, and dNk_w_noise
        
    params : list-like
        model parameters; here, only p -- the probability of missegregation
        
        
    Outputs
    __________
    
    logLikelihood : float-like
        log likelihood function of the above
    '''
    # Note the model parameters
    p, = params
    
    # Extract the parameters from the data.params object
    nCells = data.params['n_cells']
    N = data.params['n_chrom']
    
    # Extract the difference in chromatids from data.data
    deltas = data.data['dNk'].values
    
    # Calculate the likelihood
    mVals = np.arange(N+1)
    mValsNew = np.repeat(mVals,len(deltas))
    deltasNew = np.tile(deltas,len(mVals))
    
    binom1 = truncateBinom(N,mValsNew)
    binom2 = truncateBinom(mValsNew,(mValsNew-deltasNew/2.)/2.)
    binom3 = truncateBinom(mValsNew,(mValsNew+deltasNew/2.)/2.)
    
    likes1 = binom1*binom2*p**mValsNew*(1.-p)**(N-mValsNew)*2.**(-mValsNew)
    likes2 = binom1*binom3*p**mValsNew*(1.-p)**(N-mValsNew)*2.**(-mValsNew)
    
    likes = (likes1+likes2).reshape([len(mVals),len(deltas)])
    likesSum = likes.sum(axis=0)
    logLikelihood = np.sum(np.log(likesSum))
    
    return logLikelihood



def logLikeBiasedDelta(data, params):
    '''
    Compute the likelihood of the difference in chromosomes
    within the independent segregation model
    
    Inputs
    __________
    
    data : SyntheticData-like
        class containing .params, .data, and .load_data with
        .params a dictionary containing n_cells, n_chrom, p_misseg;
        .data a dataframe containing errors, dNk, and dNk_w_noise
        
    params : list-like
        params = [p, alpha]
        p : probability of chromosome missegregation
        alpha : probability chromosome ends up in daughter cell 1
        
        
    Outputs
    __________
    
    logLikelihood : float-like
        log likelihood function of the above
    '''
    
    # Write down the model parameters
    p, alpha = params
    
    # Extract the parameters from the data.params object
    nCells = data.params['n_cells']
    N = data.params['n_chrom']
    
    # Extract the difference in chromatids from data.data
    deltas = data.data['dNk'].values
    
    # Calculate the likelihood
    mVals = np.arange(N+1)
    mValsNew = np.repeat(mVals,len(deltas))
    deltasNew = np.tile(deltas,len(mVals))

    binom1 = truncateBinom(N,mValsNew)
    binom2 = truncateBinom(mValsNew,(mValsNew-deltasNew/2.)/2.)
    binom3 = truncateBinom(mValsNew,(mValsNew+deltasNew/2.)/2.)
    
    likes1 = binom1*binom2*p**mValsNew*(1.-p)**(N-mValsNew)*alpha**\
                    ((mValsNew-deltasNew/2.)/2.)*(1.-alpha)**((mValsNew+deltasNew/2.)/2.)
    likes2 = binom1*binom3*p**mValsNew*(1.-p)**(N-mValsNew)*alpha**\
                    ((mValsNew+deltasNew/2.)/2.)*(1.-alpha)**((mValsNew-deltasNew/2.)/2.)
    
    likes = (likes1+likes2).reshape([len(mVals),len(deltas)])
    likesSum = likes.sum(axis=0)
    logLikelihood = np.sum(np.log(likesSum))
    
    return logLikelihood


    
def logLikeCatDelta(data, params):
    '''
    Compute the likelihood of the difference in chromosomes
    within the independent segregation model
    
    Inputs
    __________
    
    data : SyntheticData-like
        class containing .params, .data, and .load_data with
        .params a dictionary containing n_cells, n_chrom, p_misseg;
        .data a dataframe containing errors, dNk, and dNk_w_noise
        
    params : list-like
        params = [p, alpha]
        p : probability of chromosome missegregation
        C : number of catastrophic chromosomes
        
        
    Outputs
    __________
    
    logLikelihood : float-like
        log likelihood function of the above
    '''
    
    # Note the model parameters
    p, C = params
    
    # Extract the parameters from the data.params object
    nCells = data.params['n_cells']
    N = data.params['n_chrom']
    
    # Extract the difference in chromatids from data.data
    deltas = data.data['dNk'].values
    
    # Calculate the likelihood
    mVals = np.arange(N-C+1)
    mValsNew = np.repeat(mVals,len(deltas))
    deltasNew = np.tile(deltas,len(mVals))

    binom1 = truncateBinom(N,mValsNew)
    binom2 = truncateBinom(mValsNew,(mValsNew+C-deltasNew/2.)/2.)
    binom3 = truncateBinom(mValsNew,(mValsNew+C+deltasNew/2.)/2.)
    
    likes1 = binom1*binom2*p**mValsNew*(1.-p)**(N-C-mValsNew)*2.**(-mValsNew-C)
    likes2 = binom1*binom3*p**mValsNew*(1.-p)**(N-C-mValsNew)*2.**(-mValsNew-C)
    
    likes = (likes1+likes2).reshape([len(mVals),len(deltas)])
    likesSum = likes.sum(axis=0)
    logLikelihood = np.sum(np.log(likesSum))
    
    return logLikelihood



def logLikeCatBiasedDelta(data, params):

    '''
    Compute the likelihood of the difference in chromosomes
    within the independent segregation model
    
    Inputs
    __________
    
    data : SyntheticData-like
        class containing .params, .data, and .load_data with
        .params a dictionary containing n_cells, n_chrom, p_misseg;
        .data a dataframe containing errors, dNk, and dNk_w_noise
        
    params : list-like
        params = [p, alpha]
        p : probability of chromosome missegregation
        C : number of catastrophic chromosomes
        alpha : probability chromosome ends up in daughter cell 1
        
        
    Outputs
    __________
    
    logLikelihood : float-like
        log likelihood function of the above
    '''
    
    # Note the model parameters
    p, C, alpha = params
    
    # Extract the parameters from the data.params object
    nCells = data.params['n_cells']
    N = data.params['n_chrom']
    
    # Extract the difference in chromatids from data.data
    deltas = data.data['dNk'].values
    
    # Calculate the likelihood
    mVals = np.arange(N-C+1)
    mValsNew = np.repeat(mVals,len(deltas))
    deltasNew = np.tile(deltas,len(mVals))

    binom1 = truncateBinom(N,mValsNew)
    binom2 = truncateBinom(mValsNew,(mValsNew+C-deltasNew/2.)/2.)
    binom3 = truncateBinom(mValsNew,(mValsNew+C+deltasNew/2.)/2.)

    likes1 = binom1*binom2*p**mValsNew*(1.-p)**(N-C-mValsNew)*alpha**\
                    ((mValsNew+C-deltasNew/2.)/2.)*(1.-alpha)**((mValsNew-C+deltasNew/2.)/2.)
    likes2 = binom1*binom3*p**mValsNew*(1.-p)**(N-C-mValsNew)*alpha**\
                    ((mValsNew+C+deltasNew/2.)/2.)*(1.-alpha)**((mValsNew-C-deltasNew/2.)/2.)
    
    likes = (likes1+likes2).reshape([len(mVals),len(deltas)])
    likesSum = likes.sum(axis=0)
    logLikelihood = np.sum(np.log(likesSum))
    
    return logLikelihood









    

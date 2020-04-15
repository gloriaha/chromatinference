import numpy as np
import pandas as pd
from scipy import special

def logLikeIndDelta(data, p):
    '''
    Compute the likelihood of the difference in chromosomes
    within the independent segregation model
    
    Inputs
    __________
    
    data : SyntheticData-like
        class containing .params, .data, and .load_data with
        .params a dictionary containing n_cells, n_chrom, p_misseg;
        .data a dataframe containing errors, dNk, and dNk_w_noise
        
    p : float-like
        probability of missegregation
        
        
    Outputs
    __________
    
    logLikelihood : float-like
        log likelihood function of the above
    '''
    
    # Extract the parameters from the data.params object
    nCells = data.params['n_cells']
    N = data.params['n_chrom']
    
    # Extract the difference in chromatids from data.data
    deltas = data.data['dNk'].values
    
    # Calculate the likelihood
    mVals = np.arange(N+1)
    mValsNew = np.repeat(mVals,len(deltas))
    deltasNew = np.tile(deltas,len(mVals))
    
    binom1 = special.binom(N,mValsNew)
    binom1 = (np.mod(binom1,1.)==0.)*binom1
    binom2 = special.binom(mValsNew,(mValsNew-deltasNew/2.)/2.)
    binom2 = (np.mod(binom2,1.)==0.)*binom2
    binom3 = special.binom(mValsNew,(mValsNew+deltasNew/2.)/2.)
    binom3 = (np.mod(binom3,1.)==0.)*binom3
    
    likes1 = binom1*binom2*p**mValsNew*(1.-p)**(N-mValsNew)*2.**(-mValsNew)
    likes2 = binom1*binom3*p**mValsNew*(1.-p)**(N-mValsNew)*2.**(-mValsNew)
    
    likes = (likes1+likes2).reshape([len(mVals),len(deltas)])
    likesSum = likes.sum(axis=0)
    logLikelihood = np.sum(np.log(likesSum))
    
    return logLikelihood

import numpy as np
import scipy.stats as st
import pandas as pd

def dNk_from_errors(errors, p_left=0.5):
    """
    generates random kinetochore count differences
    inputs
        errors : error counts for each cell, ndarray
        p_left : probability of missegregating into left cell, float
    output
        dN_k : random kinetochore count differences, ndarray
    """
    # choose random numbers to determine if missegregations go left or right
    random_choice = [np.random.random(int(n)) for n in errors]
    # convert into kinetochore count diferences
    dN_k = [2*np.abs(np.sum(r>p_left)-np.sum(r<=p_left)) for r in random_choice]
    return np.array(dN_k)

def generate_independent_data(params, p_left=0.5):
    """
    generates true errors and kinetochore count data for independent model
    inputs
        p_misseg : probability of independent chromosome missegregation, float
        n_cells : number of cells to sample, int
        n_chrom : number of chromosomes per cell, int
        p_left : probability of missegregating into left cell, float
    output
        Pandas dataframe containing
            errors : true error counts for each cell
            dNk : random kinetochore count differences
            dNk_w_noise : kinetochore count differences with Poisson noise added
    """
    # unpack parameters
    p_misseg, n_cells, n_chrom = params
    # randomly choose number of errors for each cell
    errors = st.binom.rvs(n_chrom, size=n_cells, p=p_misseg)
    # convert to kinetochore count difference
    dNk = dNk_from_errors(errors, p_left)
    # add Poisson noise
    dNk_w_noise = dNk + st.poisson.rvs(size=n_cells, mu=1)
    # write to dataframe
    df = pd.DataFrame({'errors' : errors,
                      'dNk' : dNk,
                      'dNk_w_noise' : dNk_w_noise})
    return df
    
def generate_catastrophe_data(params, p_left=0.5):
    """
    generates true errors and kinetochore count data for catastrophe model
    inputs
        p_misseg : probability of independent chromosome missegregation, float
        n_cells : number of cells to sample, int
        n_chrom : number of chromosomes per cell, int
        p_cat : probability of cell going into catastrophe, float
        C : fixed number of chromosomes that missegregate in catastrophe cells, int
        p_left : probability of missegregating into left cell, float
    output
        Pandas dataframe containing
            errors : true error counts for each cell
            dNk : random kinetochore count differences
            dNk_w_noise : kinetochore count differences with Poisson noise added
    """
    # unpack parameters
    p_misseg, n_cells, n_chrom, p_cat, C = params
    # randomly choose number of catastrophe cells
    n_cat = st.binom.rvs(n_cells, p=p_cat)
    # randomly choose number of errors for each cell
    errors = np.zeros(n_cells)
    errors[:n_cat]= st.binom.rvs(n_chrom, size=n_cat, p=p_misseg)+C
    errors[n_cat:] = st.binom.rvs(n_chrom, size=n_cells-n_cat, p=p_misseg)
    # convert to kinetochore count differences
    dNk = dNk_from_errors(errors, p_left)
    # add Poisson noise
    dNk_w_noise = dNk + st.poisson.rvs(size=n_cells, mu=1)
    # write to dataframe
    df = pd.DataFrame({'errors' : errors.astype(int),
                      'dNk' : dNk,
                      'dNk_w_noise' : dNk_w_noise})
    return df
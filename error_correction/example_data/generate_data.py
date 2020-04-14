import numpy as np
import scipy.stats as st

def dNk_from_errors(errors, p_left=0.5):
    """
    generates random kinetochore count differences
    inputs
        errors : error counts for each cell, ndarray
        p_left : probability of missegregating into left cell, float
    output
        dN_k : random kinetochore count differences, ndarray
    """
    random_choice = [np.random.random(int(n)) for n in errors]
    dN_k = [2*np.abs(np.sum(r>p_left)-np.sum(r<=p_left)) for r in random_choice]
    return np.array(dN_k)

def generate_independent_data(p_misseg, n_cells, n_chrom, p_left=0.5):
    """
    generates true errors and kinetochore count data for independent model
    inputs
        p_misseg : probability of independent chromosome missegregation, float
        n_cells : number of cells to sample, int
        n_chrom : number of chromosomes per cell, int
        p_left : probability of missegregating into left cell, float
    output
        errors : true error counts for each cell, ndarray
        dNk : random kinetochore count differences, ndarray
        dNk_w_noise : kinetochore count differences with Poisson noise added, ndarray
    """
    # randomly choose number of errors for each cell
    errors = st.binom.rvs(n_chrom, size=n_cells, p=p_misseg)
    # convert to kinetochore count difference
    dNk = dNk_from_errors(errors, p_left)
    # add Poisson noise
    dNk_w_noise = dNk + st.poisson.rvs(size=n_cells, mu=1)
    return errors, dNk, dNk_w_noise
    
def generate_catastrophe_data(p_misseg, n_cells, n_chrom, p_cat, C, p_left=0.5):
    """
    generates true errors and kinetochore count data for catastrophe model
    inputs
        p_misseg : probability of independent chromosome missegregation, float
        n_cells : number of cells to sample, int
        n_chrom : number of chromosomes per cell, int
        p_left : probability of missegregating into left cell, float
        p_cat : probability of cell going into catastrophe, float
        C : fixed number of chromosomes that missegregate in catastrophe cells, int
    output
        errors : true error counts for each cell, ndarray
        dNk : random kinetochore count differences, ndarray
        dNk_w_noise : kinetochore count differences with Poisson noise added, ndarray
    """
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
    return errors, dNk, dNk_w_noise
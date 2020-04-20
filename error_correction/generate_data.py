import numpy as np
import scipy.stats as st
import pandas as pd
from error_correction.data_io import get_example_data_file_path
import yaml

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
    # store cell 1, cell 2 kinetochore numbers
    N_1 = np.array([2*np.sum(r>p_left) for r in random_choice])
    N_2 = np.array([2*np.sum(r<=p_left) for r in random_choice])
    # convert into kinetochore count diferences
    dN_k = np.abs(N_1-N_2)
    return N_1, N_2, dN_k

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
    N_1, N_2, dNk = dNk_from_errors(errors, p_left)
    # add Poisson noise
    dNk_w_noise = dNk + st.poisson.rvs(size=n_cells, mu=1)
    # write to dataframe
    df = pd.DataFrame({'errors' : errors,
                       'N_1' : N_1,
                       'N_2' : N_2,
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
    N_1, N_2, dNk = dNk_from_errors(errors, p_left)
    # add Poisson noise
    dNk_w_noise = dNk + st.poisson.rvs(size=n_cells, mu=1)
    # write to dataframe
    df = pd.DataFrame({'errors' : errors.astype(int),
                       'N_1' : N_1,
                       'N_2' : N_2,
                      'dNk' : dNk,
                      'dNk_w_noise' : dNk_w_noise})
    return df

class GenerateData:
    def __init__(self, model, params, name, data_dir):
        self.data, self.params = self.make_data(model, params, name, data_dir)
    def make_data(self, model, params, name, data_dir):
        # store parameter names
        param_names = ['p_misseg', 'n_cells', 'n_chrom', 'p_cat', 'C']
        # set up file path
        data_path = get_example_data_file_path('data_'+str(name)+'.txt', data_dir)
        param_path = get_example_data_file_path('params_'+str(name)+'.yml', data_dir)
        # generate and write data files
        if model == 'independent':
            param_dict = {name:param for name, param in zip(param_names[:3], params)}
            df = generate_independent_data(params)
            with open(data_path, 'w') as f:
                f.write('# independent model\n')
                f.write(df.to_string(index=None))
            with open(param_path, 'w') as f:
                yaml.dump(param_dict, f, default_flow_style=False)
            return df, param_dict
        elif model == 'catastrophe':
            param_dict = {name:param for name, param in zip(param_names, params)}
            df = generate_catastrophe_data(params)
            with open(data_path, 'w') as f:
                f.write('# catastrophe model\n')
                f.write(df.to_string(index=None))
            with open(param_path, 'w') as f:
                yaml.dump(param_dict, f, default_flow_style=False)
            return df, param_dict
        else:
            print('Model name should be independent or catastrophe')
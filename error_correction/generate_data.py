import numpy as np
import scipy.stats as st
import pandas as pd
from error_correction.data_io import get_example_data_file_path
import yaml

def generate_errors(n_chrom, n_cells, p_misseg):
    """Generates number of random independent missegregation events
    
    Parameters
    ----------
    n_chrom : int
        number of chromosomes in cell
    n_cells : int
        number of cells
    p_misseg : float
        probability of missegregation
        
    Returns
    -------
    errors : ndarray
        error counts for each cell
    """
    errors = st.binom.rvs(n_chrom, size=n_cells, p=p_misseg)
    return errors

def dN_k_from_errors(n_chrom, errors, p_left):
    """Generates random kinetochore count differences
    
    Parameters
    ----------
    n_chrom : int
        number of chromosomes in cell
    n_cells : int
        number of cells
    p_left : float
        probability of missegregating into left cell, float
    
    Returns
    ----------
    N_1 : ndarray
        true chromatids in cell 1
    N_2 : ndarray
        true chromatids in cell 2
    dN_k : ndarray
        random kinetochore count differences, ndarray
    """
    # choose random numbers to determine if missegregations go left or right
    random_choice = [np.random.random(int(n)) for n in errors]
    # store cell 1, cell 2 kinetochore numbers
    N_1 = n_chrom - errors + np.array([2*np.sum(r<p_left) for r in random_choice])
    N_2 = n_chrom - errors + np.array([2*np.sum(r>=p_left) for r in random_choice])
    # convert into kinetochore count diferences
    dN_k = np.abs(N_1-N_2)
    return N_1, N_2, dN_k

def add_noise(N_1, N_2, p_fn):
    """Adds counting noise to true chromatid counts
    
    Parameters
    ----------
    N_1 : ndarray
        true chromatids in cell 1
    N_2 : ndarray
        true chromatids in cell 2
    p_fn : float
        probability of false negative in counting chromatids
    
    Returns
    ----------
    N_1_noise : ndarray
        measured number of chromatids in cell 1
    N_2_noise : ndarray
        measured number of chromatids in cell 2
    dN_k_noise : ndarray
        measured kinetochore count differences, ndarray
    """
    N_1_noise = np.array([st.binom.rvs(int(n), 1-p_fn) for n in N_1])
    N_2_noise = np.array([st.binom.rvs(int(n), 1-p_fn) for n in N_2])
    dN_k_noise = np.abs(N_1_noise-N_2_noise)
    return N_1_noise, N_2_noise, dN_k_noise

def generate_independent_data(params):
    """Generates true errors and kinetochore count data for independent model
    
    Parameters
    ----------
    p_misseg : float
        probability of missegregation
    n_cells : int
        number of cells
    n_chrom : int
        number of chromosomes in cell
    p_left : float
        probability of missegregating into left cell, float
    p_fn : float
        probability of false negative in counting chromatids
        
    Returns
    -------
    df : Pandas dataframe
        errors : true error counts for each cell
        N_1 : true chromatids in cell 1
        N_2 : true chromatids in cell 2
        dNk : true kinetochore count differences
        N_1_w_noise : measured number of chromatids in cell 1
        N_2_w_noise : measured number of chromatids in cell 2
        dNk_w_noise : measured kinetochore count differences
    """
    # unpack parameters
    p_misseg, n_cells, n_chrom, p_left, p_fn = params
    # randomly choose number of errors for each cell
    errors = generate_errors(n_chrom, n_cells, p_misseg)
    # convert to kinetochore count difference
    N_1, N_2, dN_k = dN_k_from_errors(n_chrom, errors, p_left)
    # add counting noise
    N_1_noise, N_2_noise, dN_k_noise = add_noise(N_1, N_2, p_fn)
    # write to dataframe
    df = pd.DataFrame({'errors' : errors,
                       'N_1' : N_1,
                       'N_2' : N_2,
                       'dNk' : dN_k,
                       'N_1_w_noise' : N_1_noise,
                       'N_2_w_noise' : N_2_noise,
                       'dNk_w_noise' : dN_k_noise})
    return df
    
def generate_catastrophe_data(params):
    """Generates true errors and kinetochore count data for catastrophe model
    
    Parameters
    ----------
    p_misseg : float
        probability of missegregation
    n_cells : int
        number of cells
    n_chrom : int
        number of chromosomes in cell
    p_left : float
        probability of missegregating into left cell, float
    p_fn : float
        probability of false negative in counting chromatids
    p_cat : float
        probability of cell going into catastrophe
    C : int
        fixed number of chromosomes that missegregate in catastrophe cells
        
    Returns
    -------
    df : Pandas dataframe
        errors : true error counts for each cell
        N_1 : true chromatids in cell 1
        N_2 : true chromatids in cell 2
        dNk : true kinetochore count differences
        N_1_w_noise : measured number of chromatids in cell 1
        N_2_w_noise : measured number of chromatids in cell 2
        dNk_w_noise : measured kinetochore count differences
    """
    # unpack parameters
    p_misseg, n_cells, n_chrom, p_left, p_fn, p_cat, C = params
    # randomly choose number of catastrophe cells
    n_cat = st.binom.rvs(n_cells, p=p_cat)
    # randomly choose number of errors for each cell
    errors = np.zeros(n_cells)
    errors[:n_cat]= generate_errors(n_chrom, n_cat, p_misseg)+C
    errors[n_cat:] = generate_errors(n_chrom, n_cells-n_cat, p_misseg)
    errors = errors.astype(int)
    # convert to kinetochore count difference
    N_1, N_2, dN_k = dN_k_from_errors(n_chrom, errors, p_left)
    # add counting noise
    N_1_noise, N_2_noise, dN_k_noise = add_noise(N_1, N_2, p_fn)
    # write to dataframe
    df = pd.DataFrame({'errors' : errors,
                       'N_1' : N_1,
                       'N_2' : N_2,
                       'dNk' : dN_k,
                       'N_1_w_noise' : N_1_noise,
                       'N_2_w_noise' : N_2_noise,
                       'dNk_w_noise' : dN_k_noise})
    return df

class GenerateData:
    def __init__(self, model, params, name, data_dir):
        """Generate synthetic data.

        Parameters
        ----------
        model : str
            name of model, 'independent' or 'catastrophe'
        params : list
            parameters of model
        name : str
            name of data file (ex: 'high_ind')
        data_dir : str
            directory to store data file

        Attributes
        ----------
        params : dict
            parameter names and values
        data : Pandas dataframe
            resulting data
        """
        self.data, self.params = self.make_data(model, params, name, data_dir)
    def make_data(self, model, params, name, data_dir):
        """Generate synthetic data.

        Parameters
        ----------
        model : str
            name of model, 'independent' or 'catastrophe'
        params : list
            parameters of model
        name : str
            name of data file (ex: 'high_ind')
        data_dir : str
            directory to store data file

        Returns
        ----------
        df : Pandas dataframe
            resulting data
        param_dict : dict
            parameter names and values
        """
        # store parameter names
        param_names = ['p_misseg', 'n_cells', 'n_chrom', 'p_left', 'p_fn', 'p_cat', 'C']
        # set up file path if name is not None
        if name:
            data_path = get_example_data_file_path('data_'+str(name)+'.txt', data_dir)
            param_path = get_example_data_file_path('params_'+str(name)+'.yml', data_dir)
        # generate and write data files
        if model == 'independent':
            param_dict = {name:param for name, param in zip(param_names[:5], params)}
            df = generate_independent_data(params)
            if name:
                # save file if name is not None
                with open(data_path, 'w') as f:
                    f.write('# independent model\n')
                    f.write(df.to_string(index=None))
                with open(param_path, 'w') as f:
                    yaml.dump(param_dict, f, default_flow_style=False)
            return df, param_dict
        elif model == 'catastrophe':
            param_dict = {name:param for name, param in zip(param_names, params)}
            df = generate_catastrophe_data(params)
            if name:
                # save file if name is not None
                with open(data_path, 'w') as f:
                    f.write('# catastrophe model\n')
                    f.write(df.to_string(index=None))
                with open(param_path, 'w') as f:
                    yaml.dump(param_dict, f, default_flow_style=False)
            return df, param_dict
        else:
            print('Model name should be independent or catastrophe')
from unittest import TestCase
from error_correction.data_io import SyntheticData
from error_correction.model import *
from error_correction.inference import *
from error_correction import generate_data
import pandas as  pd
import numpy as np
from math import inf


data_dir='tests/'

class TestEmceeFit(TestCase):
    def test_emcee_walker(self):
        np.random.seed(2)
        params = [0, 10, 10, 0.5, 0]
        data = generate_data.GenerateData('independent', params, name=None, data_dir=None)
        sampler = emcee_fit(data, [0.05, 0.1], 'biased', nwalkers=6, nsteps=2)
        # check that sampler took 2 steps
        assert len(sampler.chain[0]) == 2
        # check that steps are not identical
        assert (np.array_equal(sampler.chain[0][0], sampler.chain[0][1]) == False)
    def test_burn_in_plot(self):
        params = [0, 10, 10, 0.5, 0]
        data = generate_data.GenerateData('independent', params, name=None, data_dir=None)
        sampler = emcee_fit(data, [0.05, 0.1], 'biased', nwalkers=6, nsteps=2)
        ax = burnInPlotAffine(2, sampler, ['p_misseg', 'p_left'])
        # test that there are 2 plots, 1 for each parameter
        assert len(ax) == 2
    def test_delete_burn_in(self):
        params = [0, 10, 10, 0.5, 0]
        data = generate_data.GenerateData('independent', params, name=None, data_dir=None)
        sampler = emcee_fit(data, [0.05, 0.1], 'biased', nwalkers=6, nsteps=2)
        trimmed = delete_burn_in(sampler, 1, 2, ['p_misseg', 'p_left'])
        assert isinstance(trimmed, pd.DataFrame)
        assert len(trimmed['p_misseg']==1)
        assert len(trimmed['p_left']==1)
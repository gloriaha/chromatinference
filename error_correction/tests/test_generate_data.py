from unittest import TestCase
from error_correction.data_io import SyntheticData
from error_correction import generate_data
import pandas as  pd
import numpy as np
from math import inf


data_dir='tests/'

class TestErrorConstruction(TestCase):
    def test_zero_error(self):
        test_errors = generate_data.generate_errors(10, 10, 0)
        assert len(test_errors) == 10
        assert np.array_equal(test_errors, np.zeros(10))
    def test_100_error(self):
        test_errors = generate_data.generate_errors(10, 10, 1)
        assert np.array_equal(test_errors, 10*np.ones(10))
    def test_error_range(self):
        test_errors = generate_data.generate_errors(10, 1, 0.1)
        assert test_errors in range(10)
        
class TestDeltaConstruction(TestCase):
    def test_delta_range(self):
        test_N1, test_N2, test_delta = generate_data.dN_k_from_errors(10, 
                                                                      np.array([10,0]), 
                                                                      0.5)
        # check that 10 errors is in correct range
        assert test_N1[0] in range(20)
        assert test_N2[0] in range(20)
        assert test_delta[0] in range(20)
        # check that 0 errors gives expected results
        assert test_N1[1] == 10
        assert test_N2[1] == 10
        assert test_delta[1] == 0
    def test_delta_dependence(self):
        test_N1, test_N2, test_delta = generate_data.dN_k_from_errors(10, 
                                                                      np.array([10,0]), 
                                                                      0.5)
        # check that delta is the absolute difference between N1 and N2
        assert np.abs(test_N1[0]-test_N2[0]) == test_delta[0]

class TestNoiseConstruction(TestCase):
    def test_zero_noise(self):
        # check that no noise doesn't change results
        test_N1_noise, test_N2_noise, test_delta_noise = generate_data.add_noise(np.array([20,10]), 
                                                                                 np.array([0,10]), 
                                                                                 0)
        assert np.array_equal(test_N1_noise, np.array([20,10]))
        assert np.array_equal(test_N2_noise, np.array([0,10]))
        assert np.array_equal(test_delta_noise, np.array([20,0]))
    def test_100_noise(self):
        # check that 100% false negative rate yields expected results
        test_N1_noise, test_N2_noise, test_delta_noise = generate_data.add_noise(np.array([20,10]), 
                                                                                 np.array([0,10]), 
                                                                                 1)
        assert np.array_equal(test_N1_noise, np.array([0,0]))
        assert np.array_equal(test_N2_noise, np.array([0,0]))
        assert np.array_equal(test_delta_noise, np.array([0,0]))
    def test_add_noise(self):
        # check that intermediate noise is in correct range
        test_N1_noise, test_N2_noise, test_delta_noise = generate_data.add_noise(np.array([20,10]), 
                                                                                 np.array([0,10]), 
                                                                                 0.5)
        assert test_N1_noise[0] in range(20)
        assert test_N2_noise[0] == 0
        assert test_delta_noise[0] in range(20)
        assert test_N1_noise[1] in range(10)
        assert test_N2_noise[1] in range(10)
        assert test_delta_noise[1] in range(10)

class TestDataConstruction(TestCase):
    def test_independent_data_zero_error(self):
        # check that no errors occur when p_misseg=0
        params = [0, 10, 10, 0.5, 0]
        df = generate_data.generate_independent_data(params)
        assert np.array_equal(df['dNk'].values, np.zeros(10))
        # also check that no noise occurs when p_fn=0
        assert np.array_equal(df['dNk'].values, df['dNk_w_noise'].values)
    def test_independent_data_range(self):
        # check that results are in expected range
        params = [0.01, 10, 10, 0.5, 0]
        df = generate_data.generate_independent_data(params)
        assert np.all(df['dNk'].values >= 0)
        assert np.all(df['dNk'].values <= 20)
        # also check that no noise occurs when p_fn=0
        assert np.array_equal(df['dNk'].values, df['dNk_w_noise'].values)
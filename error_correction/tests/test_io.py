from unittest import TestCase
from error_correction.data_io import SyntheticData
from error_correction import generate_data
from error_correction.model import *
import pandas as  pd
import numpy as np
from math import inf


data_dir='tests/'

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

class TestIo(TestCase):
    def test_data_io(self):
        data = SyntheticData('params_test.yml',
                             'data_test.txt',
                             data_dir)
        assert data.data['errors'][0] == 1

class TestLike1(TestCase):
    def test_data_io(self):
        data = SyntheticData('params_test.yml',
                             'data_test.txt',
                            data_dir)
        pee = np.random.rand()
        logp = np.log(pee)
        logLike = logLikeIndDelta(data,[pee])
        assert logp == logLike

class TestLike2(TestCase):
    def test_data_io(self):
        data = SyntheticData('params_test.yml',
                             'data_test.txt',
                            data_dir)
        pee = np.random.rand()
        logLike1 = logLikeIndDelta(data,[pee])
        logLike2 = logLikeCatDelta(data,[pee,0.])
        assert logLike1 == logLike2

class TestLike3(TestCase):
    def test_data_io(self):
        data = SyntheticData('params_test.yml',
                             'data_test.txt',
                            data_dir)
        pee = np.random.rand()
        logLike1 = logLikeIndDelta(data,[pee])
        logLike2 = logLikeBiasedDelta(data,[pee,0.5])
        assert logLike1 == logLike2

class TestLike4(TestCase):
    def test_data_io(self):
        data = SyntheticData('params_test.yml',
                             'data_test.txt',
                            data_dir)
        pee = np.random.rand()
        logLike1 = logLikeIndDelta(data,[pee])
        logLike2 = logLikeCatBiasedDelta(data,[pee,0.,0.5])
        assert logLike1 == logLike2

class TestLike5(TestCase):
    def test_data_io(self):
        data = SyntheticData('params_test.yml',
                             'data_test_2.txt',
                            data_dir)
        pee = np.random.rand()
        logLike1 = logLikeIndDelta(data,[pee])
        assert logLike1 == -inf

class TestBinom1(TestCase):
    def test_data_io(self):
        tB1 = truncateBinom(1.,0.)
        assert tB1 == 1.

class TestBinom2(TestCase):
    def test_data_io(self):
        tB1 = truncateBinom(1.,0.5)
        assert tB1 == 0.

class TestBinom3(TestCase):
    def test_data_io(self):
        tB1 = truncateBinom([1.,1.],[0.5,1.])
        assert tB1.all() == np.array([0.,1.]).all()

from unittest import TestCase
from error_correction.data_io import SyntheticData
from error_correction.model import *
import pandas as  pd
import numpy as np
from math import inf


data_dir='tests/'

# Test the probN1 function

class TestProbN1_1(TestCase):
    def test_data_io(self):
        # The test should succeed if probN1(0., 0.5, 10, 10)
        # is one, since p=0 gives no missegregations
        assert probN1(0., 0.5, 10, 10) == 1.

class TestProbN1_2(TestCase):
    def test_data_io(self):
        # The test should succeed if probN1(1., 0., 10, 20)
        # is one, since p=1 gives 10 missegregations and
        # all end up in one daughter cell, giving N1=20
        assert probN1(1., 1., 10, 20) == 1.

class TestProbN1_3(TestCase):
    def test_data_io(self):
        # Same as TestProbN1_2 but opposite limit
        assert probN1(1., 0., 10, 0) == 1.



# Test the probDelta function

class TestProbDelta_1(TestCase):
    def test_data_io(self):
        # The test should succeed if probDelta(0., 0.5, 10, 0)
        # is one, since p=0 gives no missegregations
        assert probDelta(0., 0.5, 10, 0) == 1.

class TestProbDelta_2(TestCase):
    def test_data_io(self):
        # The test should succeed if probDelta(0., 0.5, 10, 0)
        # is one, since p=1 gives 10 missegregations and thus
        # delta is 20
        assert probDelta(1., 0., 10, 20) == 1.

class TestProbDelta_3(TestCase):
    def test_data_io(self):
        # Same as TestProbDelta_2 but opposite limit
        assert probDelta(1., 1., 10, 20) == 1.



# Test the biased log likelihood function

class TestUnbiasedLike_1(TestCase):
    def test_data_io(self):
        data = SyntheticData('params_test.yml',
                             'data_test.txt',
                            data_dir)
        pee = np.random.rand()
        logp = np.log(pee)
        logLike = logLikeUnbiasedDelta([pee], data.data['dNk'], data.params['n_chrom'])
        assert logp == logLike

##class TestLike2(TestCase):
##    def test_data_io(self):
##        data = SyntheticData('params_test.yml',
##                             'data_test.txt',
##                            data_dir)
##        pee = np.random.rand()
##        logLike1 = logLikeIndDelta(data,[pee])
##        logLike2 = logLikeGivenCatDelta(data,[pee,0.])
##        assert logLike1 == logLike2

class TestLike7(TestCase):
    def test_data_io(self):
        data = SyntheticData('params_test.yml',
                             'data_test.txt',
                            data_dir)
        pee = np.random.rand()
        logLike1 = logLikeUnbiasedDelta([pee], data.data['dNk'], data.params['n_chrom'])
        logLike2 = logLikeBiasedDelta([pee,0.5], data.data['dNk'], data.params['n_chrom'])
        assert logLike1 == logLike2

##class TestLike4(TestCase):
##    def test_data_io(self):
##        data = SyntheticData('params_test.yml',
##                             'data_test.txt',
##                            data_dir)
##        pee = np.random.rand()
##        logLike1 = logLikeIndDelta(data,[pee])
##        logLike2 = logLikeGivenCatBiasedDelta(data,[pee,0.,0.5])
##        assert logLike1 == logLike2

class TestLike8(TestCase):
    def test_data_io(self):
        data = SyntheticData('params_test.yml',
                             'data_test_2.txt',
                            data_dir)
        pee = np.random.rand()
        logLike1 = logLikeUnbiasedDelta([pee], data.data['dNk'], data.params['n_chrom'])
        assert logLike1 == -inf

##class TestLike6(TestCase):
##    def test_data_io(self):
##        data = SyntheticData('params_test.yml',
##                             'data_test.txt',
##                            data_dir)
##        pee = np.random.rand()
##        logLike1 = logLikeCatDelta(data,[pee,0.,0.])
##        logLike2 = logLikeGivenCatDelta(data,[pee,0.])
##        assert logLike1 == logLike2

##class TestLike7(TestCase):
##    def test_data_io(self):
##        data = SyntheticData('params_test.yml',
##                             'data_test.txt',
##                            data_dir)
##        pee = np.random.rand()
##        logLike1 = logLikeCatBiasedDelta(data,[pee,0.,0.,0.5])
##        logLike2 = logLikeGivenCatBiasedDelta(data,[pee,0.,0.5])
##        assert logLike1 == logLike2

class TestPriors(TestCase):
    def test_biased_prior(self):
        prior = logPriorBiasedDelta([0,0])
        assert prior == 0
    def test_biased_prior_range(self):
        prior = logPriorBiasedDelta([-10, 0])
        prior2 = logPriorBiasedDelta([0,-10])
        assert prior == -np.inf
        assert prior2 == -np.inf

class TestPost(TestCase):
    def test_biased_post(self):
        post = logPostBiasedDelta([0,0], np.array([0, 0]), 10)
        assert post != -np.inf

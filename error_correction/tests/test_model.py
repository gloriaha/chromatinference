from unittest import TestCase
from error_correction.data_io import SyntheticData
from error_correction.model import *
import pandas as  pd
import numpy as np
from math import inf


data_dir='tests/'

# Test the probN1 function

class TestProbN1(TestCase):
    def test_zero_error_probN1(self):
        # The test should succeed if probN1(0., 0.5, 10, 10)
        # is one, since p=0 gives no missegregations
        assert probN1(0., 0.5, 10, 10) == 1.

    def test_100_error_100_bias_probN1(self):
        # The test should succeed if probN1(1., 0., 10, 20)
        # is one, since p=1 gives 10 missegregations and
        # all end up in one daughter cell, giving N1=20
        assert probN1(1., 1., 20, 10) == 1.

    def test_100_error_zero_bias_probN1(self):
        # Same as test_100_error_100_bias_probN1 but opposite limit (symmetry)
        assert probN1(1., 0., 0, 10) == 1.

# Test the probN1N2Noisy function

class TestProbN1N2Noisy(TestCase):
    def test_zero_noise_zero_error_probN1N2Noisy(self):
        # Test that perfect detection and no missegregations
        # gives probability 1 of N1, N2 = N
        en = 10
        en1 = np.array([en])
        en2 = en1
        testThing = probN1N2Noisy(0., 0.5, 1., en1, en2, en)
        assert testThing == 1.

    def test_zero_noise_zero_error_N1N2difference_impossible(self):
        # Test that perfect detection and no missegregations
        # gives probability 0 of N1 = N-1, N2 = N+1
        en = 10
        en1 = np.array([en])-1
        en2 = np.array([en])+1
        testThing = probN1N2Noisy(0., 0.5, 1., en1, en2, en)
        assert testThing == 0.

    def test_zero_noise_100_error_100_bias_probN1N2Noisy(self):
        # Test that perfect detection and 100% chance of missegregation
        # and bias against cell 1 gives N1 = 0 and N2 = 2
        en = 1
        en1 = np.array([en])-1
        en2 = np.array([en])+1
        testThing = probN1N2Noisy(1., 0., 1., en1, en2, en)
        assert testThing == 1.

    def test_zero_noise_100_error_zero_bias_probN1N2Noisy(self):
        # Test the above but with opposite bias
        en = 1
        en1 = np.array([en])+1
        en2 = np.array([en])-1
        testThing = probN1N2Noisy(1., 1., 1., en1, en2, en)
        assert testThing == 1.

    def test_finite_noise_probN1N2Noisy(self):
        # Check that with non-unit detection efficiency, we get a sen-
        # sible probability
        pdet = np.random.uniform(0., 1.)
        en = 10
        en1 = np.array([en])
        en2 = en1
        testThing = probN1N2Noisy(0., 0.5, pdet, en1, en2, en)
        testProb = pdet**(2*en)
        assert '%.5f'%testThing == '%.5f'%testProb



# Test the biased log likelihood function

class TestLogLikeBiasedDelta(TestCase):
    def test_known_data_biased_loglike(self):
        data = SyntheticData('params_test.yml',
                             'data_test.txt',
                            data_dir)
        pee = np.random.rand()
        logp = np.log(pee)
        logLike = logLikeUnbiasedDelta([pee], data.data['dNk'], data.params['n_chrom'])
        assert logp == logLike



# Test the unbiased likelihood function

class TestLogLikeUnbiasedDelta(TestCase):
    def test_known_data_unbiased_loglike(self):
        data = SyntheticData('params_test.yml',
                             'data_test.txt',
                            data_dir)
        pee = np.random.rand()
        logLike1 = logLikeUnbiasedDelta([pee], data.data['dNk'], data.params['n_chrom'])
        logLike2 = logLikeBiasedDelta([pee,0.5], data.data['dNk'], data.params['n_chrom'])
        assert logLike1 == logLike2

    def test_impossible_data_unbiased_loglike(self):
        data = SyntheticData('params_test.yml',
                             'data_test_2.txt',
                            data_dir)
        pee = np.random.rand()
        logLike1 = logLikeUnbiasedDelta([pee], data.data['dNk'], data.params['n_chrom'])
        assert logLike1 == -inf



# Test the biased, noisy likelihood function

class TestLogLikeBiasedNoisy(TestCase):
    def test_zero_noise_zero_error_only_possible_case(self):
        # Test that perfect detection and no missegregations
        # gives probability 1 of N1, N2 = N
        en = 10
        testThing = logLikeBiasedNoisy([0., 0.5, 1.], np.array([en]), np.array([en]), en)
        assert testThing == 0.

    def test_zero_noise_zero_error_impossible_case(self):
        # Test that perfect detection and no missegregations
        # gives probability 0 of N1 = N-1, N2 = N+1
        en = 10
        testThing = logLikeBiasedNoisy([0., 0.5, 1.], np.array([en-1]), np.array([en+1]), en)
        assert testThing == -inf

    def test_zero_noise_100_error_equal_prob(self):
        # Test that perfect detection and 100% chance of missegregation
        # and bias against cell 1 gives N1 = 0 and N2 = 2
        en = 1
        testThing = logLikeBiasedNoisy([1., 0., 1.], np.array([en-1]), np.array([en+1]), en)
        assert testThing == -np.log(2)

    def test_zero_noise_100_error_equal_prob_symmetry(self):
        # Test the above but with opposite bias
        en = 1
        testThing = logLikeBiasedNoisy([1., 1., 1.], np.array([en+1]), np.array([en-1]), en)
        assert testThing == -np.log(2)

    def test_finite_noise_logLikeBiasedNoisy(self):
        # Check that with non-unit detection efficiency, we get a sen-
        # sible probability
        pdet = np.random.uniform(0., 1.)
        en = 10
        testThing = logLikeBiasedNoisy([0., 0.5, pdet], np.array([en]), np.array([en]), en)
        testProb = pdet**(2*en)
        assert '%.5f'%testThing == '%.5f'%np.log(testProb)



# Test the unbiased, noisy likelihood function

class TestLogLikeUnbiasedNoisy(TestCase):
    def test_biased_unbiased_loglike_equality(self):
        data = SyntheticData('params_test.yml',
                             'data_test.txt',
                            data_dir)
        pee = np.random.rand()
        pd = np.random.rand()
        en = 10
        logLike1 = logLikeUnbiasedNoisy([pee, pd], np.array([en]), np.array([en]), en)
        logLike2 = logLikeBiasedNoisy([pee, 0.5, pd], np.array([en]), np.array([en]), en)
        assert logLike1 == logLike2


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
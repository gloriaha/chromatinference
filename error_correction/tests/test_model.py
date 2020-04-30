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
        assert probN1(1., 1., 20, 10) == 1.

class TestProbN1_3(TestCase):
    def test_data_io(self):
        # Same as TestProbN1_2 but opposite limit
        assert probN1(1., 0., 0, 10) == 1.



# Test the probDelta function

class TestProbDelta_1(TestCase):
    def test_data_io(self):
        # The test should succeed if probDelta(0., 0.5, 10, 0)
        # is one, since p=0 gives no missegregations
        assert probDelta(0., 0.5, 0, 10) == 1.

class TestProbDelta_2(TestCase):
    def test_data_io(self):
        # The test should succeed if probDelta(0., 0.5, 10, 0)
        # is one, since p=1 gives 10 missegregations and thus
        # delta is 20
        assert probDelta(1., 0., 20, 10) == 1.

class TestProbDelta_3(TestCase):
    def test_data_io(self):
        # Same as TestProbDelta_2 but opposite limit
        assert probDelta(1., 1., 20, 10) == 1.



# Test the binomialProbsNoisy function

class TestBinomialProbsNoisy_1(TestCase):
    def test_data_io(self):
        # Test that perfect detection gives probability 1 of
        # N1Tilde = N1 and that subsequent values are 0
        testThing = binomialProbsNoisy(1., 10, 10, 10)
        assert testThing == np.array([1.])



# Test the probN1N2Noisy function

class TestProbN1N2Noisy_1(TestCase):
    def test_data_io(self):
        # Test that perfect detection and no missegregations
        # gives probability 1 of N1, N2 = N
        en = 10
        en1 = en
        en2 = en
        testThing = probN1N2Noisy(0., 0.5, 1., en1, en2, en)
        assert testThing == 1.

class TestProbN1N2Noisy_2(TestCase):
    def test_data_io(self):
        # Test that perfect detection and no missegregations
        # gives probability 0 of N1 = N-1, N2 = N+1
        en = 10
        en1 = en-1
        en2 = en+1
        testThing = probN1N2Noisy(0., 0.5, 1., en1, en2, en)
        assert testThing == 0.

class TestProbN1N2Noisy_3(TestCase):
    def test_data_io(self):
        # Test that perfect detection and 100% chance of missegregation
        # and bias against cell 1 gives N1 = 0 and N2 = 2
        en = 1
        en1 = en-1
        en2 = en+1
        testThing = probN1N2Noisy(1., 0., 1., en1, en2, en)
        assert testThing == 1.

class TestProbN1N2Noisy_4(TestCase):
    def test_data_io(self):
        # Test the above but with opposite bias
        en = 1
        en1 = en+1
        en2 = en-1
        testThing = probN1N2Noisy(1., 1., 1., en1, en2, en)
        assert testThing == 1.

class TestProbN1N2Noisy_5(TestCase):
    def test_data_io(self):
        # Check that with non-unit detection efficiency, we get a sen-
        # sible probability
        pdet = np.random.uniform(0., 1.)
        en = 10
        en1 = en
        en2 = en
        testThing = probN1N2Noisy(0., 0.5, pdet, en1, en2, en) 
        testProb = pdet**(2*en)
        assert '%.5f'%testThing == '%.5f'%testProb



# Test the biased log likelihood function

class TestLogLikeBiasedDelta_1(TestCase):
    def test_data_io(self):
        data = SyntheticData('params_test.yml',
                             'data_test.txt',
                            data_dir)
        pee = np.random.rand()
        logp = np.log(pee)
        logLike = logLikeUnbiasedDelta([pee], data.data['dNk'], data.params['n_chrom'])
        assert logp == logLike



# Test the unbiased likelihood function

class TestLogLikeUnbiasedDelta_1(TestCase):
    def test_data_io(self):
        data = SyntheticData('params_test.yml',
                             'data_test.txt',
                            data_dir)
        pee = np.random.rand()
        logLike1 = logLikeUnbiasedDelta([pee], data.data['dNk'], data.params['n_chrom'])
        logLike2 = logLikeBiasedDelta([pee,0.5], data.data['dNk'], data.params['n_chrom'])
        assert logLike1 == logLike2

class TestLogLikeUnbiasedDelta_2(TestCase):
    def test_data_io(self):
        data = SyntheticData('params_test.yml',
                             'data_test_2.txt',
                            data_dir)
        pee = np.random.rand()
        logLike1 = logLikeUnbiasedDelta([pee], data.data['dNk'], data.params['n_chrom'])
        assert logLike1 == -inf



# Test the biased, noisy likelihood function

class TestLogLikeBiasedNoisy_1(TestCase):
    def test_data_io(self):
        # Test that perfect detection and no missegregations
        # gives probability 1 of N1, N2 = N
        en = 10
        testThing = logLikeBiasedNoisy([0., 0.5, 1.], np.array([en]), np.array([en]), en)
        assert testThing == 0.

class TestLogLikeBiasedNoisy_2(TestCase):
    def test_data_io(self):
        # Test that perfect detection and no missegregations
        # gives probability 0 of N1 = N-1, N2 = N+1
        en = 10
        testThing = logLikeBiasedNoisy([0., 0.5, 1.], np.array([en-1]), np.array([en+1]), en)
        assert testThing == -inf

class TestLogLikeBiasedNoisy_3(TestCase):
    def test_data_io(self):
        # Test that perfect detection and 100% chance of missegregation
        # and bias against cell 1 gives N1 = 0 and N2 = 2
        en = 1
        testThing = logLikeBiasedNoisy([1., 0., 1.], np.array([en-1]), np.array([en+1]), en)
        assert testThing == -np.log(2)

class TestLogLikeBiasedNoisy_4(TestCase):
    def test_data_io(self):
        # Test the above but with opposite bias
        en = 1
        testThing = logLikeBiasedNoisy([1., 1., 1.], np.array([en+1]), np.array([en-1]), en)
        assert testThing == -np.log(2)

class TestLogLikeBiasedNoisy_5(TestCase):
    def test_data_io(self):
        # Check that with non-unit detection efficiency, we get a sen-
        # sible probability
        pdet = np.random.uniform(0., 1.)
        en = 10
        testThing = logLikeBiasedNoisy([0., 0.5, pdet], np.array([en]), np.array([en]), en)
        testProb = pdet**(2*en)
        assert '%.5f'%testThing == '%.5f'%np.log(testProb)



# Test the unbiased, noisy likelihood function

class TestLogLikeUnbiasedNoisy1(TestCase):
    def test_data_io(self):
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



# Old, deprecated tests
# __________________________________________________
# __________________________________________________
# __________________________________________________
# __________________________________________________
# __________________________________________________



##class TestLike2(TestCase):
##    def test_data_io(self):
##        data = SyntheticData('params_test.yml',
##                             'data_test.txt',
##                            data_dir)
##        pee = np.random.rand()
##        logLike1 = logLikeIndDelta(data,[pee])
##        logLike2 = logLikeGivenCatDelta(data,[pee,0.])
##        assert logLike1 == logLike2




##class TestLike4(TestCase):
##    def test_data_io(self):
##        data = SyntheticData('params_test.yml',
##                             'data_test.txt',
##                            data_dir)
##        pee = np.random.rand()
##        logLike1 = logLikeIndDelta(data,[pee])
##        logLike2 = logLikeGivenCatBiasedDelta(data,[pee,0.,0.5])
##        assert logLike1 == logLike2



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

from unittest import TestCase
from error_correction.data_io import SyntheticData
from error_correction.model import *
import pandas as  pd
import numpy as np
from math import inf


data_dir='tests/'

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
        logLike2 = logLikeGivenCatDelta(data,[pee,0.])
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
        logLike2 = logLikeGivenCatBiasedDelta(data,[pee,0.,0.5])
        assert logLike1 == logLike2

class TestLike5(TestCase):
    def test_data_io(self):
        data = SyntheticData('params_test.yml',
                             'data_test_2.txt',
                            data_dir)
        pee = np.random.rand()
        logLike1 = logLikeIndDelta(data,[pee])
        assert logLike1 == -inf

class TestLike6(TestCase):
    def test_data_io(self):
        data = SyntheticData('params_test.yml',
                             'data_test.txt',
                            data_dir)
        pee = np.random.rand()
        logLike1 = logLikeCatDelta(data,[pee,0.,0.])
        logLike2 = logLikeGivenCatDelta(data,[pee,0.])
        assert logLike1 == logLike2

class TestLike7(TestCase):
    def test_data_io(self):
        data = SyntheticData('params_test.yml',
                             'data_test.txt',
                            data_dir)
        pee = np.random.rand()
        logLike1 = logLikeCatBiasedDelta(data,[pee,0.,0.,0.5])
        logLike2 = logLikeGivenCatBiasedDelta(data,[pee,0.,0.5])
        assert logLike1 == logLike2

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

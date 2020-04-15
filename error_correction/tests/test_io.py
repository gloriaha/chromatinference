from unittest import TestCase
from error_correction.data_io import SyntheticData
from error_correction.model import logLikeIndDelta
import pandas as  pd
import numpy as np


dir=''

class TestIo(TestCase):
    def test_data_io(self):
        data = SyntheticData('params_test.yml',
                             'data_test.txt',
                             dir)
        assert data.data['errors'][0] == 1

class TestIo2(TestCase):
    def test_data_io(self):
        data = SyntheticData('params_test.yml',
                             'data_test.txt',
                            dir)
        pee = np.random.rand()
        logp = np.log(pee)
        logLike = logLikeIndDelta(data,pee)
        assert logp == logLike

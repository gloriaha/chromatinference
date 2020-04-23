from unittest import TestCase
from error_correction.data_io import SyntheticData
import pandas as  pd
import numpy as np
from math import inf


data_dir='tests/'

class TestIo(TestCase):
    def test_data_io(self):
        # test data loading
        data = SyntheticData('params_test.yml',
                             'data_test.txt',
                             data_dir)
        assert data.data['errors'][0] == 1
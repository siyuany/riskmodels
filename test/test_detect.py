# -*- encoding: utf-8 -*-
import numpy as np
import pandas as pd
import unittest

from addpath import addpath

try:
    addpath()
    from sy_riskmodels.detector import detect
except ModuleNotFoundError as e:
    raise e


class TestDetect(unittest.TestCase):

    def test_detect(self):
        arr = np.random.rand(50, 100)
        df = pd.DataFrame(arr, columns=['V' + str(i) for i in range(100)])
        res1 = detect(df, n_cores=1)
        res2 = detect(df, n_cores=None)
        res2 = detect(df, n_cores=None)


if __name__ == '__main__':
    unittest.main()
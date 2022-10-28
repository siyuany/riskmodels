# -*- encoding: utf-8 -*-

import numpy as np
import pandas as pd
import unittest

from addpath import addpath

try:
    addpath()
    from riskmodels.scorecard import WOEBinFactory
    from riskmodels.scorecard import WOEBin
except ModuleNotFoundError as e:
    raise e


class TestWOEBin(unittest.TestCase):

    def setUp(self) -> None:
        self.df = pd.read_csv('germancredit.csv')

    def test_split_vec_to_df(self):
        x = ['b%,%d', 'a', 'c%,%e']
        df = WOEBin.split_vec_to_df(x)
        print(df)


    def test_woebin_cat_vars(self):
        tmp_df = pd.DataFrame({
            'variable': 'property',
            'value': self.df['property'],
            'y': np.where(self.df['creditability'] == 'good', 0, 1)
        })
        woe_bin_method = WOEBinFactory.build(['quantile', 'tree'])
        binning_result = woe_bin_method(tmp_df)
        print(binning_result)


if __name__ == '__main__':
    unittest.main()

# -*- encoding: utf-8 -*-

import unittest

import numpy as np
import pandas as pd

from addpath import addpath

try:
    addpath()
    from sy_riskmodels.scorecard import WOEBinFactory
    from sy_riskmodels.scorecard import WOEBin
    from sy_riskmodels.contrib.build_scorecard import build_scorecard
except ModuleNotFoundError as e:
    raise e


class TestWOEBin(unittest.TestCase):

    def setUp(self) -> None:
        self.df = pd.read_hdf('dataset.h5', key='germancredit')
        self.df2 = pd.read_hdf('dataset.h5', key='creditcard')

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

    def test_chi2_woebin(self):
        binner = WOEBinFactory.build(['quantile', 'chi2'])
        tmp_df = pd.DataFrame({
            'variable': 'V3',
            'value': self.df2['V3'],
            'y': self.df2['Class']
        })
        binning_result = binner(tmp_df)
        print(binning_result)

    def test_build_scorecard(self):
        features = self.df2.columns.tolist()[1:-1]
        print(np.quantile(self.df2['Time'], q=0.8))
        build_scorecard(self.df2,
                        features=features,
                        target='Class',
                        train_filter=lambda x: x['Time'] <= 140000,
                        oot_filter=lambda x: x['Time'] > 140000,
                        output_excel_file='test.xlsx')


if __name__ == '__main__':
    unittest.main()

# -*- encoding: utf-8 -*-

import unittest

import numpy as np
import pandas as pd

from addpath import addpath

try:
  addpath()
  from syriskmodels.scorecard import WOEBinFactory
  from syriskmodels.scorecard import WOEBin
  from syriskmodels.scorecard import woebin
  from syriskmodels.contrib.build_scorecard import build_scorecard
  from syriskmodels.scorecard import RuleOptimBin
  from syriskmodels.scorecard import QuantileInitBin
  from syriskmodels.scorecard import sc_bins_to_df
except ModuleNotFoundError as e:
  raise e


class TestRuleOptimBin(unittest.TestCase):

  def setUp(self) -> None:
    self.df: pd.DataFrame = pd.read_hdf('dataset.h5', key='germancredit')
    self.df2: pd.DataFrame = pd.read_hdf('dataset.h5', key='creditcard')
    pd.set_option('display.max_columns', 100)

  def test_rulebin(self):
    dtm = self.df2[['V3', 'Class']].copy()
    dtm.rename(columns={'V3': 'value', 'Class': 'y'}, inplace=True)
    dtm['variable'] = 'V3'
    q_bin = QuantileInitBin(initial_bins=50)
    breaks = q_bin.woebin(dtm)
    r_bin = RuleOptimBin()
    print(r_bin.woebin(dtm, breaks))

    print(
        woebin(
            self.df2,
            x=['V3'],
            y='Class',
            methods=[QuantileInitBin(initial_bins=50),
                     RuleOptimBin()])['V3'])

  def test_rulebin2(self):
    dtm = self.df2[['V4', 'Class']].copy()
    dtm.rename(columns={'V4': 'value', 'Class': 'y'}, inplace=True)
    dtm['variable'] = 'V4'
    q_bin = QuantileInitBin(initial_bins=50)
    breaks = q_bin.woebin(dtm)
    r_bin = RuleOptimBin()
    print(r_bin.woebin(dtm, breaks))

    print(
        woebin(
            self.df2,
            x=['V4'],
            y='Class',
            methods=[QuantileInitBin(initial_bins=50),
                     RuleOptimBin()])['V4'])

  def test_rulebin3(self):
    variables = ['V' + str(i) for i in range(1, 29)]
    bins = woebin(
        self.df2,
        x=variables,
        y='Class',
        methods=[QuantileInitBin(50), RuleOptimBin()])
    woe, _ = sc_bins_to_df(bins)
    woe.to_csv('rule_bin.csv', index=False)


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
    build_scorecard(
        self.df2,
        features=features,
        target='Class',
        train_filter=lambda x: x['Time'] <= 140000,
        oot_filter=lambda x: x['Time'] > 140000,
        output_excel_file='test.xlsx')


if __name__ == '__main__':
  unittest.main()

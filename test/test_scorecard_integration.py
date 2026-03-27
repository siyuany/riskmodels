# -*- encoding: utf-8 -*-

import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from syriskmodels.scorecard import (
    WOEBinFactory,
    WOEBin,
    woebin,
    RuleOptimBin,
    QuantileInitBin,
    sc_bins_to_df,
)
from syriskmodels.contrib.build_scorecard import build_scorecard


def _test_data_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _germancredit_csv():
    return os.path.abspath(os.path.join(_test_data_dir(), 'germancredit.csv'))


def _creditcard_csv():
    return os.path.abspath(os.path.join(_test_data_dir(), 'creditcard.csv'))


def _load_csvs_in_subprocess():
    """在子进程中仅用 pandas 读取两个 CSV，再通过 pickle 传回，避免 pytest 下主进程读 CSV 产生 _NoValueType。"""
    path1 = _germancredit_csv()
    path2 = _creditcard_csv()
    code = """
import pandas as pd
import sys
p1, p2, out1, out2 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
pd.read_csv(p1).to_pickle(out1)
pd.read_csv(p2).to_pickle(out2)
"""
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f1, \
         tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f2:
        f1.close()
        f2.close()
        try:
            subprocess.run(
                [sys.executable, '-c', code, path1, path2, f1.name, f2.name],
                check=True,
                cwd=_test_data_dir(),
            )
            df1 = pd.read_pickle(f1.name)
            df2 = pd.read_pickle(f2.name)
            return df1, df2
        finally:
            os.unlink(f1.name)
            os.unlink(f2.name)


class TestRuleOptimBinIntegration(unittest.TestCase):

    def setUp(self) -> None:
        self.df, self.df2 = _load_csvs_in_subprocess()
        pd.set_option('display.max_columns', 100)



    def test_rulebin_single_variable(self):
        dtm = self.df2[['V3', 'Class']].copy()
        dtm.rename(columns={'V3': 'value', 'Class': 'y'}, inplace=True)
        dtm['variable'] = 'V3'
        q_bin = QuantileInitBin(initial_bins=50)
        breaks = q_bin.woebin(dtm)
        r_bin = RuleOptimBin()
        r_bin.woebin(dtm, breaks)

        woebin_res = woebin(
            self.df2,
            x=['V3'],
            y='Class',
            methods=[QuantileInitBin(initial_bins=50), RuleOptimBin()],
        )['V3']
        assert isinstance(woebin_res, pd.DataFrame)



    def test_rulebin_another_variable(self):
        dtm = self.df2[['V4', 'Class']].copy()
        dtm.rename(columns={'V4': 'value', 'Class': 'y'}, inplace=True)
        dtm['variable'] = 'V4'
        q_bin = QuantileInitBin(initial_bins=50)
        breaks = q_bin.woebin(dtm)
        r_bin = RuleOptimBin()
        r_bin.woebin(dtm, breaks)

        woebin_res = woebin(
            self.df2,
            x=['V4'],
            y='Class',
            methods=[QuantileInitBin(initial_bins=50), RuleOptimBin()],
        )['V4']
        assert isinstance(woebin_res, pd.DataFrame)

    def test_rulebin_multiple_variables(self):
        variables = ['V' + str(i) for i in range(1, 29)]
        bins = woebin(
            self.df2,
            x=variables,
            y='Class',
            methods=[QuantileInitBin(50), RuleOptimBin()],
        )
        woe, _ = sc_bins_to_df(bins)
        assert isinstance(woe, pd.DataFrame)
        assert not woe.empty


class TestWOEBinIntegration(unittest.TestCase):

    def setUp(self) -> None:
        self.df, self.df2 = _load_csvs_in_subprocess()

    def test_split_vec_to_df(self):
        x = ['b%,%d', 'a', 'c%,%e']
        df = WOEBin.split_vec_to_df(x)
        assert set(df['bin_chr'].unique()) == set(x)

    def test_woebin_cat_vars(self):
        tmp_df = pd.DataFrame({
            'variable': 'property',
            'value': self.df['property'],
            'y': np.where(self.df['creditability'] == 'good', 0, 1),
        })
        woe_bin_method = WOEBinFactory.build(['quantile', 'tree'])
        binning_result = woe_bin_method(tmp_df)
        assert isinstance(binning_result, pd.DataFrame)
        assert not binning_result.empty

    def test_chi2_woebin(self):
        binner = WOEBinFactory.build(['quantile', 'chi2'])
        tmp_df = pd.DataFrame({
            'variable': 'V3',
            'value': self.df2['V3'],
            'y': self.df2['Class'],
        })
        binning_result = binner(tmp_df)
        assert isinstance(binning_result, pd.DataFrame)
        assert not binning_result.empty

    def test_build_scorecard_pipeline(self):
        features = self.df2.columns.tolist()[1:-1]
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            build_scorecard(
                self.df2,
                features=features,
                target='Class',
                train_filter=lambda x: x['Time'] <= 140000,
                oot_filter=lambda x: x['Time'] > 140000,
                output_excel_file=tmp_path,
            )
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == '__main__':
    unittest.main()


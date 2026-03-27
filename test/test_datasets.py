# -*- encoding: utf-8 -*-
"""datasets 模块单元测试"""
import pytest
import pandas as pd

from syriskmodels.datasets import load_germancredit, load_creditcard, get_data_dir


class TestGetDataDir:
    """get_data_dir() 测试"""

    def test_returns_path(self):
        data_dir = get_data_dir()
        assert data_dir.exists(), f'数据目录不存在: {data_dir}'
        assert data_dir.is_dir()

    def test_contains_data_files(self):
        data_dir = get_data_dir()
        files = [f.name for f in data_dir.glob('*.csv.gz')]
        assert 'germancredit.csv.gz' in files
        assert 'creditcard.csv.gz' in files


class TestLoadGermancredit:
    """load_germancredit() 测试"""

    def test_returns_dataframe(self):
        df = load_germancredit()
        assert isinstance(df, pd.DataFrame)

    def test_shape(self):
        df = load_germancredit()
        assert df.shape[0] == 1000
        assert df.shape[1] == 21

    def test_target_column_exists(self):
        df = load_germancredit()
        assert 'creditability' in df.columns

    def test_target_values(self):
        df = load_germancredit()
        assert set(df['creditability'].unique()) == {0, 1}


class TestLoadCreditcard:
    """load_creditcard() 测试"""

    def test_returns_dataframe(self):
        df = load_creditcard()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        df = load_creditcard()
        assert 'Time' in df.columns
        assert 'Class' in df.columns
        assert 'V1' in df.columns

    def test_target_values(self):
        df = load_creditcard()
        assert set(df['Class'].unique()) == {0, 1}

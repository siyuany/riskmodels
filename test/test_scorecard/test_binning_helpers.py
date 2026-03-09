# -*- encoding: utf-8 -*-
"""
测试分箱辅助函数模块
"""
import pytest
import pandas as pd
import numpy as np
from syriskmodels.scorecard.utils.binning_helpers import (
    extract_numeric_breaks,
    format_numeric_bin_names,
    extract_breaks_from_binning,
    compute_woe,
    compute_iv,
    merge_adjacent_bins
)


class TestExtractNumericBreaks:
    """测试数值型切分点提取"""
    
    def test_normal_bins(self):
        """测试正常分箱"""
        binning = pd.DataFrame({
            'bin_chr': ['[-inf, 20)', '[20, 40)', '[40, inf)']
        })
        breaks = extract_numeric_breaks(binning)
        assert list(breaks) == [20.0, 40.0, np.inf]
    
    def test_single_bin(self):
        """测试单分箱"""
        binning = pd.DataFrame({
            'bin_chr': ['[-inf, inf)']
        })
        breaks = extract_numeric_breaks(binning)
        assert list(breaks) == [np.inf]
    
    def test_merged_bins(self):
        """测试合并后的分箱"""
        binning = pd.DataFrame({
            'bin_chr': ['[-inf, 20)', '[20,60)']  # 已合并
        })
        breaks = extract_numeric_breaks(binning)
        assert list(breaks) == [20.0, 60.0]
    
    def test_many_bins(self):
        """测试多个分箱"""
        binning = pd.DataFrame({
            'bin_chr': [f'[{i*10},{(i+1)*10})' for i in range(10)]
        })
        breaks = extract_numeric_breaks(binning)
        expected = [(i+1)*10.0 for i in range(10)]
        expected[-1] = np.inf  # 最后一个应该是 inf
        assert list(breaks[:-1]) == expected[:-1]
        assert breaks.iloc[-1] == np.inf
    
    def test_empty_dataframe(self):
        """测试空 DataFrame"""
        binning = pd.DataFrame(columns=['bin_chr'])
        with pytest.raises(Exception):
            extract_numeric_breaks(binning)
    
    def test_returns_series(self):
        """测试返回 Series"""
        binning = pd.DataFrame({
            'bin_chr': ['[-inf, 20)', '[20, 40)']
        })
        breaks = extract_numeric_breaks(binning)
        assert isinstance(breaks, pd.Series)


class TestFormatNumericBinNames:
    """测试分箱名格式化"""
    
    def test_merge_adjacent_intervals(self):
        """测试相邻区间合并"""
        binning = pd.DataFrame({
            'bin_chr': ['[20,40)%,%[40,60)', '[60,80)']
        })
        result = format_numeric_bin_names(binning)
        assert result['bin_chr'].iloc[0] == '[20,60)'
    
    def test_no_merge_needed(self):
        """测试无需合并"""
        binning = pd.DataFrame({
            'bin_chr': ['[20,40)', '[40,60)']
        })
        result = format_numeric_bin_names(binning)
        assert list(result['bin_chr']) == ['[20,40)', '[40,60)']
    
    def test_multiple_merges(self):
        """测试多次合并"""
        binning = pd.DataFrame({
            'bin_chr': ['[10,20)%,%[20,30)%,%[30,40)']
        })
        result = format_numeric_bin_names(binning)
        assert result['bin_chr'].iloc[0] == '[10,40)'
    
    def test_mixed_merged_and_not(self):
        """测试混合合并和未合并"""
        binning = pd.DataFrame({
            'bin_chr': ['[10,20)%,%[20,30)', '[30,40)', '[40,50)%,%[50,60)']
        })
        result = format_numeric_bin_names(binning)
        expected = ['[10,30)', '[30,40)', '[40,60)']
        assert list(result['bin_chr']) == expected
    
    def test_returns_dataframe(self):
        """测试返回 DataFrame"""
        binning = pd.DataFrame({
            'bin_chr': ['[20,40)']
        })
        result = format_numeric_bin_names(binning)
        assert isinstance(result, pd.DataFrame)
    
    def test_does_not_modify_original(self):
        """测试不修改原始数据"""
        binning = pd.DataFrame({
            'bin_chr': ['[20,40)%,%[40,60)']
        })
        original = binning['bin_chr'].iloc[0]
        result = format_numeric_bin_names(binning)
        assert binning['bin_chr'].iloc[0] == original
        assert result['bin_chr'].iloc[0] == '[20,60)'


class TestExtractBreaksFromBinning:
    """测试统一提取接口"""
    
    def test_numeric_variable(self):
        """测试数值型变量"""
        binning = pd.DataFrame({
            'bin_chr': ['[-inf, 20)', '[20, 40)', '[40, inf)']
        })
        breaks = extract_breaks_from_binning(binning, is_numeric=True)
        assert list(breaks) == [20.0, 40.0, np.inf]
    
    def test_categorical_variable(self):
        """测试类别型变量"""
        binning = pd.DataFrame({
            'bin_chr': ['A%,%B', 'C', 'D%,%E']
        })
        breaks = extract_breaks_from_binning(binning, is_numeric=False)
        assert list(breaks) == ['A%,%B', 'C', 'D%,%E']
    
    def test_numeric_with_merge(self):
        """测试数值型含合并"""
        binning = pd.DataFrame({
            'bin_chr': ['[20,40)%,%[40,60)', '[60,80)']
        })
        breaks = extract_breaks_from_binning(binning, is_numeric=True)
        assert list(breaks) == [60.0, 80.0]
    
    def test_categorical_single_categories(self):
        """测试类别型单个类别"""
        binning = pd.DataFrame({
            'bin_chr': ['A', 'B', 'C']
        })
        breaks = extract_breaks_from_binning(binning, is_numeric=False)
        assert list(breaks) == ['A', 'B', 'C']
    
    def test_numeric_preserves_order(self):
        """测试数值型保持顺序"""
        binning = pd.DataFrame({
            'bin_chr': ['[-inf, 10)', '[10, 20)', '[20, inf)']
        })
        breaks = extract_breaks_from_binning(binning, is_numeric=True)
        assert breaks.iloc[0] < breaks.iloc[1] < breaks.iloc[2]


class TestComputeWOE:
    """测试 WOE 计算"""
    
    def test_normal_case(self):
        """测试正常情况"""
        good = np.array([80, 60, 40])
        bad = np.array([20, 40, 60])
        woe = compute_woe(good, bad, epsilon=0.5)
        assert len(woe) == 3
        # 第一箱好客户多，WOE 应为正
        assert woe[0] > 0
        # 第三箱坏客户多，WOE 应为负
        assert woe[2] < 0
    
    def test_zero_good(self):
        """测试好客户为 0"""
        good = np.array([0, 60])
        bad = np.array([10, 40])
        woe = compute_woe(good, bad, epsilon=0.5)
        assert not np.isnan(woe[0])  # 不应为 NaN
        assert woe[0] < 0  # 应为负
    
    def test_zero_bad(self):
        """测试坏客户为 0"""
        good = np.array([80, 60])
        bad = np.array([0, 40])
        woe = compute_woe(good, bad, epsilon=0.5)
        assert not np.isnan(woe[0])  # 不应为 NaN
        assert woe[0] > 0  # 应为正
    
    def test_all_zero(self):
        """测试全为 0"""
        good = np.array([0, 0])
        bad = np.array([0, 0])
        woe = compute_woe(good, bad, epsilon=0.5)
        assert len(woe) == 2
        # 不应为 NaN
        assert not np.any(np.isnan(woe))
    
    def test_epsilon_effect(self):
        """测试 epsilon 影响"""
        good = np.array([0])
        bad = np.array([0])
        woe_small = compute_woe(good, bad, epsilon=0.1)
        woe_large = compute_woe(good, bad, epsilon=1.0)
        assert woe_small != woe_large
    
    def test_equal_good_bad(self):
        """测试好坏客户相等"""
        good = np.array([50, 50])
        bad = np.array([50, 50])
        woe = compute_woe(good, bad, epsilon=0.5)
        assert np.allclose(woe, 0)  # WOE 应接近 0
    
    def test_returns_numpy_array(self):
        """测试返回 numpy 数组"""
        good = np.array([80, 60])
        bad = np.array([20, 40])
        woe = compute_woe(good, bad, epsilon=0.5)
        assert isinstance(woe, np.ndarray)


class TestComputeIV:
    """测试 IV 计算"""
    
    def test_normal_case(self):
        """测试正常情况"""
        woe = np.array([0.5, 0, -0.5])
        good = np.array([80, 60, 40])
        bad = np.array([20, 40, 60])
        iv = compute_iv(woe, good, bad)
        assert iv > 0
        assert isinstance(iv, (float, np.floating))
    
    def test_zero_iv(self):
        """测试零 IV"""
        woe = np.array([0, 0])
        good = np.array([50, 50])
        bad = np.array([50, 50])
        iv = compute_iv(woe, good, bad)
        assert iv == 0
    
    def test_high_iv(self):
        """测试高 IV"""
        woe = np.array([2.0, -2.0])
        good = np.array([90, 10])
        bad = np.array([10, 90])
        iv = compute_iv(woe, good, bad)
        assert iv > 1.0
    
    def test_single_bin(self):
        """测试单分箱"""
        woe = np.array([0.5])
        good = np.array([80])
        bad = np.array([20])
        iv = compute_iv(woe, good, bad)
        assert iv > 0
    
    def test_with_zero_counts(self):
        """测试含零计数"""
        woe = np.array([1.0, -1.0])
        good = np.array([0, 100])
        bad = np.array([10, 90])
        iv = compute_iv(woe, good, bad)
        assert not np.isnan(iv)


class TestMergeAdjacentBins:
    """测试相邻分箱合并"""
    
    def test_merge_first_two(self):
        """测试合并前两箱"""
        binning = pd.DataFrame({
            'bin_chr': ['[20,40)', '[40,60)', '[60,80)'],
            'good': [30, 30, 40],
            'bad': [10, 10, 20],
            'count': [40, 40, 60],
            'count_distr': [0.3, 0.3, 0.4]
        })
        result = merge_adjacent_bins(binning, idx=1)
        assert len(result) == 2
        assert result['bin_chr'].iloc[0] == '[20,60)'
        assert result['good'].iloc[0] == 60
        assert result['bad'].iloc[0] == 20
    
    def test_merge_last_two(self):
        """测试合并后两箱"""
        binning = pd.DataFrame({
            'bin_chr': ['[20,40)', '[40,60)', '[60,80)'],
            'good': [30, 30, 40],
            'bad': [10, 10, 20],
        })
        result = merge_adjacent_bins(binning, idx=2)
        assert len(result) == 2
        assert result['bin_chr'].iloc[1] == '[40,80)'
        assert result['good'].iloc[1] == 70
    
    def test_invalid_idx_zero(self):
        """测试无效索引 (第一箱)"""
        binning = pd.DataFrame({
            'bin_chr': ['[20,40)', '[40,60)'],
            'good': [30, 30],
            'bad': [10, 10],
        })
        with pytest.raises((IndexError, ValueError)):
            merge_adjacent_bins(binning, idx=0)  # 第一箱无法向前合并
    
    def test_invalid_idx_out_of_range(self):
        """测试索引越界"""
        binning = pd.DataFrame({
            'bin_chr': ['[20,40)', '[40,60)'],
            'good': [30, 30],
            'bad': [10, 10],
        })
        with pytest.raises(IndexError):
            merge_adjacent_bins(binning, idx=10)
    
    def test_preserves_other_columns(self):
        """测试保留其他列"""
        binning = pd.DataFrame({
            'bin_chr': ['[20,40)', '[40,60)', '[60,80)'],
            'good': [30, 30, 40],
            'bad': [10, 10, 20],
            'woe': [0.5, 0.3, -0.2]
        })
        result = merge_adjacent_bins(binning, idx=1)
        assert 'woe' in result.columns
        assert len(result) == 2
    
    def test_categorical_merge(self):
        """测试类别型合并"""
        binning = pd.DataFrame({
            'bin_chr': ['A%,%B', 'C', 'D'],
            'good': [30, 30, 40],
            'bad': [10, 10, 20],
        })
        result = merge_adjacent_bins(binning, idx=1)
        assert len(result) == 2
        assert result['bin_chr'].iloc[0] == 'A%,%B%,%C'

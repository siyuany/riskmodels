# -*- encoding: utf-8 -*-
"""
测试常量定义模块
"""
import pytest

# 这些测试会在 constants.py 实现前失败 (Red 阶段)
from syriskmodels.scorecard.constants import (
    VariableStatus,
    BinningMethod,
    NumericBinPatterns,
    DefaultValues
)


class TestVariableStatus:
    """测试变量状态枚举"""
    
    def test_ok_value(self):
        """测试 OK 值"""
        assert VariableStatus.OK == 0
    
    def test_constant_value(self):
        """测试 CONSTANT 值"""
        assert VariableStatus.CONSTANT == 10
    
    def test_too_many_categories_value(self):
        """测试 TOO_MANY_CATEGORIES 值"""
        assert VariableStatus.TOO_MANY_CATEGORIES == 20
    
    def test_comparison(self):
        """测试枚举比较"""
        assert VariableStatus.OK < VariableStatus.CONSTANT
        assert VariableStatus.CONSTANT < VariableStatus.TOO_MANY_CATEGORIES
    
    def test_enum_names(self):
        """测试枚举名称"""
        assert VariableStatus(0).name == 'OK'
        assert VariableStatus(10).name == 'CONSTANT'
        assert VariableStatus(20).name == 'TOO_MANY_CATEGORIES'


class TestBinningMethod:
    """测试分箱方法常量"""
    
    def test_quantile(self):
        assert BinningMethod.QUANTILE == 'quantile'
    
    def test_histogram(self):
        assert BinningMethod.HISTOGRAM == 'hist'
    
    def test_chimerge(self):
        assert BinningMethod.CHIMERGE == 'chi2'
    
    def test_tree(self):
        assert BinningMethod.TREE == 'tree'
    
    def test_rule(self):
        assert BinningMethod.RULE == 'rule'
    
    def test_all_methods_are_strings(self):
        """测试所有方法都是字符串"""
        assert isinstance(BinningMethod.QUANTILE, str)
        assert isinstance(BinningMethod.HISTOGRAM, str)
        assert isinstance(BinningMethod.CHIMERGE, str)
        assert isinstance(BinningMethod.TREE, str)
        assert isinstance(BinningMethod.RULE, str)


class TestNumericBinPatterns:
    """测试数值型分箱正则模式"""
    
    def test_interval_pattern_match(self):
        """测试区间模式匹配"""
        pattern = NumericBinPatterns.INTERVAL_PATTERN
        assert pattern.match('[-inf, 20)')
        assert pattern.match('[20, 40)')
        assert pattern.match('[40, inf)')
    
    def test_interval_pattern_no_match(self):
        """测试不匹配的情况"""
        pattern = NumericBinPatterns.INTERVAL_PATTERN
        assert not pattern.match('20')
        assert not pattern.match('A%,%B')
        assert not pattern.match('missing')
    
    def test_interval_pattern_extract(self):
        """测试区间模式提取"""
        pattern = NumericBinPatterns.INTERVAL_PATTERN
        match = pattern.match('[20, 40)')
        assert match is not None
        assert match.group(1) == '20'
        assert match.group(2) == '40'
    
    def test_merge_pattern(self):
        """测试合并模式"""
        pattern = NumericBinPatterns.MERGE_PATTERN
        assert pattern.search('[20,40)%,%[40,60)')
        assert pattern.search('[10,20)%,%[20,30)%,%[30,40)')
    
    def test_merge_pattern_no_match(self):
        """测试合并不匹配的情况"""
        pattern = NumericBinPatterns.MERGE_PATTERN
        assert not pattern.search('[20,40)')
        assert not pattern.search('A%,%B')
    
    def test_missing_flag(self):
        """测试 missing 标记"""
        assert NumericBinPatterns.MISSING_FLAG == 'missing'
    
    def test_category_separator(self):
        """测试类别分隔符"""
        assert NumericBinPatterns.CATEGORY_SEPARATOR == '%,%'
    
    def test_category_separator_in_action(self):
        """测试分隔符实际使用"""
        bin_name = 'A' + NumericBinPatterns.CATEGORY_SEPARATOR + 'B'
        assert bin_name == 'A%,%B'
        parts = bin_name.split(NumericBinPatterns.CATEGORY_SEPARATOR)
        assert parts == ['A', 'B']


class TestDefaultValues:
    """测试默认值常量"""
    
    def test_initial_bins(self):
        assert DefaultValues.INITIAL_BINS == 20
    
    def test_bin_num_limit(self):
        assert DefaultValues.BIN_NUM_LIMIT == 5
    
    def test_min_iv_increase(self):
        assert DefaultValues.MIN_IV_INCREASE == 0.05
    
    def test_count_distr_limit(self):
        assert DefaultValues.COUNT_DISTR_LIMIT == 0.02
    
    def test_max_categories(self):
        assert DefaultValues.MAX_CATEGORIES == 50
    
    def test_epsilon(self):
        assert DefaultValues.EPSILON == 0.5
    
    def test_significant_figs(self):
        assert DefaultValues.SIGNIFICANT_FIGS == 4
    
    def test_default_cores_factor(self):
        assert DefaultValues.DEFAULT_CORES_FACTOR == 5
    
    def test_max_core_usage(self):
        assert DefaultValues.MAX_CORE_USAGE == 0.9
    
    def test_parallel_threshold_samples(self):
        assert DefaultValues.PARALLEL_THRESHOLD_SAMPLES == 1000
    
    def test_chi2_default_pvalue(self):
        assert DefaultValues.CHI2_DEFAULT_PVALUE == 0.05
    
    def test_scorecard_base_points(self):
        assert DefaultValues.BASE_POINTS == 600
    
    def test_scorecard_base_odds(self):
        assert DefaultValues.BASE_ODDS == 50
    
    def test_scorecard_pdo(self):
        assert DefaultValues.PDO == 20
    
    def test_numeric_values_are_correct_type(self):
        """测试数值类型正确"""
        assert isinstance(DefaultValues.INITIAL_BINS, int)
        assert isinstance(DefaultValues.BIN_NUM_LIMIT, int)
        assert isinstance(DefaultValues.MIN_IV_INCREASE, float)
        assert isinstance(DefaultValues.EPSILON, float)
        assert isinstance(DefaultValues.MAX_CORE_USAGE, float)

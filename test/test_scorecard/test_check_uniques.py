# -*- encoding: utf-8 -*-
"""
测试变量唯一值检查函数

针对 syriskmodels.scorecard.utils.validation.check_uniques 函数的测试
"""
import pytest
import pandas as pd
import numpy as np

from syriskmodels.scorecard.utils.validation import check_uniques
from syriskmodels.scorecard.constants import VariableStatus


class TestCheckUniques:
    """测试变量唯一值检查"""
    
    def test_normal_numeric_var(self):
        """测试正常数值型变量"""
        s = pd.Series([1, 2, 3, 4, 5])
        status = check_uniques(s)
        assert status == VariableStatus.OK
    
    def test_constant_var(self):
        """测试常量变量"""
        s = pd.Series([999, 999, 999, 999])
        status = check_uniques(s)
        assert status == VariableStatus.CONSTANT
    
    def test_constant_with_nan(self):
        """测试含空值的常量变量"""
        s = pd.Series([999, 999, np.nan, 999])
        status = check_uniques(s)
        assert status == VariableStatus.CONSTANT
    
    def test_too_many_categories(self):
        """测试类别过多"""
        s = pd.Series([f'cat_{i}' for i in range(100)])
        status = check_uniques(s, max_cate_num=50)
        assert status == VariableStatus.TOO_MANY_CATEGORIES
    
    def test_many_categories_but_numeric(self):
        """测试数值型但类别多（应该通过）"""
        s = pd.Series(range(100))
        status = check_uniques(s, max_cate_num=50)
        assert status == VariableStatus.OK  # 数值型不受限
    
    def test_all_nan(self):
        """测试全空值"""
        s = pd.Series([np.nan, np.nan, np.nan])
        status = check_uniques(s)
        assert status == VariableStatus.CONSTANT
    
    def test_single_value(self):
        """测试单一样本"""
        s = pd.Series([1])
        status = check_uniques(s)
        assert status == VariableStatus.CONSTANT
    
    def test_empty_series(self):
        """测试空序列"""
        s = pd.Series([], dtype=float)
        status = check_uniques(s)
        assert status == VariableStatus.CONSTANT
    
    def test_custom_max_cate_num(self):
        """测试自定义最大类别数"""
        s = pd.Series(['A', 'B', 'C', 'D', 'E'])
        status = check_uniques(s, max_cate_num=3)
        assert status == VariableStatus.TOO_MANY_CATEGORIES
        status = check_uniques(s, max_cate_num=10)
        assert status == VariableStatus.OK
    
    def test_two_unique_values(self):
        """测试两个唯一值（正常）"""
        s = pd.Series([0, 1, 0, 1, 0, 1])
        status = check_uniques(s)
        assert status == VariableStatus.OK
    
    def test_mixed_numeric_with_special_values(self):
        """测试含特殊值的数值型变量"""
        s = pd.Series([1, 2, 3, -999, -999, np.nan])
        status = check_uniques(s)
        assert status == VariableStatus.OK  # 有多个唯一值
    
    def test_categorical_with_special_values(self):
        """测试含特殊值的类别型变量"""
        s = pd.Series(['A', 'B', 'C', '-999', '-999'])
        status = check_uniques(s)
        assert status == VariableStatus.OK
    
    def test_boundary_max_cate_num(self):
        """测试边界类别数"""
        # 刚好等于 max_cate_num
        s = pd.Series([f'cat_{i}' for i in range(50)])
        status = check_uniques(s, max_cate_num=50)
        assert status == VariableStatus.OK  # 刚好等于限制
        # 超过 1 个
        s = pd.Series([f'cat_{i}' for i in range(51)])
        status = check_uniques(s, max_cate_num=50)
        assert status == VariableStatus.TOO_MANY_CATEGORIES

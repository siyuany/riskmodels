# -*- encoding: utf-8 -*-
"""
测试变量唯一值检查函数

注意：这些测试针对现有 scorecard.py 中的 check_uniques 函数
在重构后，该函数将使用新的常量和异常类
"""
import pytest
import pandas as pd
import numpy as np

# 当前从原模块导入，重构后将使用新模块
from syriskmodels.scorecard import check_uniques


class TestCheckUniques:
    """测试变量唯一值检查"""
    
    def test_normal_numeric_var(self):
        """测试正常数值型变量"""
        s = pd.Series([1, 2, 3, 4, 5])
        status = check_uniques(s)
        assert status == 0  # OK
    
    def test_constant_var(self):
        """测试常量变量"""
        s = pd.Series([999, 999, 999, 999])
        status = check_uniques(s)
        assert status == 10  # CONSTANT
    
    def test_constant_with_nan(self):
        """测试含空值的常量变量"""
        s = pd.Series([999, 999, np.nan, 999])
        status = check_uniques(s)
        assert status == 10  # CONSTANT
    
    def test_too_many_categories(self):
        """测试类别过多"""
        s = pd.Series([f'cat_{i}' for i in range(100)])
        status = check_uniques(s, max_cate_num=50)
        assert status == 20  # TOO_MANY_VALUES
    
    def test_many_categories_but_numeric(self):
        """测试数值型但类别多（应该通过）"""
        s = pd.Series(range(100))
        status = check_uniques(s, max_cate_num=50)
        assert status == 0  # OK - 数值型不受限
    
    def test_all_nan(self):
        """测试全空值"""
        s = pd.Series([np.nan, np.nan, np.nan])
        status = check_uniques(s)
        assert status == 10  # CONSTANT (0 个唯一值)
    
    def test_single_value(self):
        """测试单一样本"""
        s = pd.Series([1])
        status = check_uniques(s)
        assert status == 10  # CONSTANT
    
    def test_empty_series(self):
        """测试空序列"""
        s = pd.Series([], dtype=float)
        status = check_uniques(s)
        assert status == 10  # CONSTANT
    
    def test_custom_max_cate_num(self):
        """测试自定义最大类别数"""
        s = pd.Series(['A', 'B', 'C', 'D', 'E'])
        status = check_uniques(s, max_cate_num=3)
        assert status == 20  # TOO_MANY_VALUES
        status = check_uniques(s, max_cate_num=10)
        assert status == 0  # OK
    
    def test_two_unique_values(self):
        """测试两个唯一值（正常）"""
        s = pd.Series([0, 1, 0, 1, 0, 1])
        status = check_uniques(s)
        assert status == 0  # OK
    
    def test_mixed_numeric_with_special_values(self):
        """测试含特殊值的数值型变量"""
        s = pd.Series([1, 2, 3, -999, -999, np.nan])
        status = check_uniques(s)
        assert status == 0  # OK - 有多个唯一值
    
    def test_categorical_with_special_values(self):
        """测试含特殊值的类别型变量"""
        s = pd.Series(['A', 'B', 'C', '-999', '-999'])
        status = check_uniques(s)
        assert status == 0  # OK
    
    def test_boundary_max_cate_num(self):
        """测试边界类别数"""
        # 刚好等于 max_cate_num
        s = pd.Series([f'cat_{i}' for i in range(50)])
        status = check_uniques(s, max_cate_num=50)
        assert status == 0  # OK - 刚好等于限制
        # 超过 1 个
        s = pd.Series([f'cat_{i}' for i in range(51)])
        status = check_uniques(s, max_cate_num=50)
        assert status == 20  # TOO_MANY_VALUES


class TestCheckUniquesWithFixtures:
    """使用 fixture 测试 check_uniques"""
    
    def test_clean_data_variables(self, clean_data):
        """测试干净数据中的变量"""
        for col in ['age', 'income']:
            status = check_uniques(clean_data[col])
            assert status == 0  # OK
        
        status = check_uniques(clean_data['city'])
        assert status == 0  # OK - 只有 3 个类别
    
    def test_constant_var_fixture(self, data_with_constant_var):
        """测试常量变量 fixture"""
        status = check_uniques(data_with_constant_var['constant'])
        assert status == 10  # CONSTANT
    
    def test_too_many_categories_fixture(self, data_with_too_many_categories):
        """测试类别过多 fixture"""
        status = check_uniques(data_with_too_many_categories['many_cats'])
        assert status == 20  # TOO_MANY_VALUES
    
    def test_special_values_fixture(self, data_with_special_values):
        """测试含特殊值 fixture"""
        status = check_uniques(data_with_special_values['age'])
        assert status == 0  # OK - 有多个唯一值（包括 -999）
    
    def test_all_nan_fixture(self, data_all_nan):
        """测试全空值 fixture"""
        status = check_uniques(data_all_nan['all_nan'])
        assert status == 10  # CONSTANT
    
    def test_single_sample_fixture(self, data_single_sample):
        """测试单样本 fixture"""
        status = check_uniques(data_single_sample['age'])
        assert status == 10  # CONSTANT
    
    def test_empty_dataframe_fixture(self, data_empty):
        """测试空数据框 fixture"""
        if len(data_empty) == 0:
            # 空 DataFrame 的 Series 也是空的
            s = pd.Series([], dtype=object)
            status = check_uniques(s)
            assert status == 10  # CONSTANT

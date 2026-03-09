# -*- encoding: utf-8 -*-
"""
测试常量定义模块

注意：根据评审反馈，仅保留 VariableStatus 枚举
BinningMethod、NumericBinPatterns、DefaultValues 已删除（过度设计）
"""
from syriskmodels.scorecard.constants import VariableStatus


class TestVariableStatus:
    """测试变量状态枚举"""
    
    def test_ok_value(self):
        """测试 OK 值为 0"""
        assert VariableStatus.OK == 0
    
    def test_constant_value(self):
        """测试 CONSTANT 值为 10"""
        assert VariableStatus.CONSTANT == 10
    
    def test_too_many_categories_value(self):
        """测试 TOO_MANY_CATEGORIES 值为 20"""
        assert VariableStatus.TOO_MANY_CATEGORIES == 20
    
    def test_enum_names(self):
        """测试枚举名称"""
        assert VariableStatus(0).name == 'OK'
        assert VariableStatus(10).name == 'CONSTANT'
        assert VariableStatus(20).name == 'TOO_MANY_CATEGORIES'
    
    def test_enum_comparison(self):
        """测试枚举可比较"""
        assert VariableStatus.OK < VariableStatus.CONSTANT
        assert VariableStatus.CONSTANT < VariableStatus.TOO_MANY_CATEGORIES
    
    def test_enum_equality(self):
        """测试枚举相等性"""
        assert VariableStatus.OK == VariableStatus.OK
        assert VariableStatus.CONSTANT != VariableStatus.OK

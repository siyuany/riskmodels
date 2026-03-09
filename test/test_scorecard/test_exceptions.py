# -*- encoding: utf-8 -*-
"""
测试异常定义模块
"""
import pytest
from syriskmodels.scorecard.exceptions import (
    WOEBinError,
    ConstantVariableError,
    TooManyCategoriesError,
    InvalidBreaksError,
    DataValidationError,
    BinningAlgorithmError,
    WOEComputationError
)


class TestWOEBinError:
    """测试基类异常"""
    
    def test_base_exception_inheritance(self):
        """测试继承关系"""
        assert issubclass(ConstantVariableError, WOEBinError)
        assert issubclass(TooManyCategoriesError, WOEBinError)
        assert issubclass(InvalidBreaksError, WOEBinError)
        assert issubclass(DataValidationError, WOEBinError)
        assert issubclass(BinningAlgorithmError, WOEBinError)
        assert issubclass(WOEComputationError, WOEBinError)
    
    def test_base_exception_message(self):
        """测试基类异常消息"""
        with pytest.raises(WOEBinError) as exc_info:
            raise WOEBinError("测试错误")
        assert "测试错误" in str(exc_info.value)
    
    def test_base_exception_can_be_caught(self):
        """测试基类异常可以捕获子类"""
        with pytest.raises(WOEBinError):
            raise ConstantVariableError('test_var')


class TestConstantVariableError:
    """测试常量变量异常"""
    
    def test_exception_message(self):
        """测试异常消息"""
        with pytest.raises(ConstantVariableError) as exc_info:
            raise ConstantVariableError('age')
        assert "变量 'age' 为常量变量" in str(exc_info.value)
    
    def test_exception_variable_name(self):
        """测试变量名包含"""
        var_name = 'income'
        with pytest.raises(ConstantVariableError) as exc_info:
            raise ConstantVariableError(var_name)
        assert var_name in str(exc_info.value)
    
    def test_exception_chinese_var_name(self):
        """测试中文变量名"""
        with pytest.raises(ConstantVariableError) as exc_info:
            raise ConstantVariableError('年龄')
        assert '年龄' in str(exc_info.value)
    
    def test_exception_is_woebin_error(self):
        """测试是 WOEBinError 子类"""
        with pytest.raises(WOEBinError):
            raise ConstantVariableError('test')


class TestTooManyCategoriesError:
    """测试类别过多异常"""
    
    def test_exception_message(self):
        """测试异常消息"""
        with pytest.raises(TooManyCategoriesError) as exc_info:
            raise TooManyCategoriesError('city', 100, 50)
        assert "100" in str(exc_info.value)
        assert "50" in str(exc_info.value)
    
    def test_exception_parameters(self):
        """测试参数信息"""
        var_name = 'category'
        n_cats = 200
        max_allowed = 50
        with pytest.raises(TooManyCategoriesError) as exc_info:
            raise TooManyCategoriesError(var_name, n_cats, max_allowed)
        assert var_name in str(exc_info.value)
        assert str(n_cats) in str(exc_info.value)
        assert str(max_allowed) in str(exc_info.value)
    
    def test_exception_is_woebin_error(self):
        """测试是 WOEBinError 子类"""
        with pytest.raises(WOEBinError):
            raise TooManyCategoriesError('test', 100, 50)


class TestInvalidBreaksError:
    """测试无效切分点异常"""
    
    def test_exception_message(self):
        """测试异常消息"""
        with pytest.raises(InvalidBreaksError) as exc_info:
            raise InvalidBreaksError("切分点必须为数值型")
        assert "切分点" in str(exc_info.value)
    
    def test_exception_custom_message(self):
        """测试自定义消息"""
        msg = "切分点数量与分箱数不匹配"
        with pytest.raises(InvalidBreaksError) as exc_info:
            raise InvalidBreaksError(msg)
        assert msg in str(exc_info.value)
    
    def test_exception_is_woebin_error(self):
        """测试是 WOEBinError 子类"""
        with pytest.raises(WOEBinError):
            raise InvalidBreaksError('test')


class TestDataValidationError:
    """测试数据验证异常"""
    
    def test_exception_message(self):
        """测试异常消息"""
        with pytest.raises(DataValidationError) as exc_info:
            raise DataValidationError("目标变量包含空值")
        assert "目标变量" in str(exc_info.value)
    
    def test_exception_y_not_in_columns(self):
        """测试 y 标不在列中"""
        with pytest.raises(DataValidationError) as exc_info:
            raise DataValidationError("目标变量 'y' 不在数据集中")
        assert 'y' in str(exc_info.value)
    
    def test_exception_is_woebin_error(self):
        """测试是 WOEBinError 子类"""
        with pytest.raises(WOEBinError):
            raise DataValidationError('test')


class TestBinningAlgorithmError:
    """测试分箱算法异常"""
    
    def test_exception_message(self):
        """测试异常消息"""
        with pytest.raises(BinningAlgorithmError) as exc_info:
            raise BinningAlgorithmError("ChiMerge 算法执行失败")
        assert "算法" in str(exc_info.value)
    
    def test_exception_with_details(self):
        """测试带详细信息的异常"""
        msg = "Tree 分箱失败：无法找到满足条件的切分点"
        with pytest.raises(BinningAlgorithmError) as exc_info:
            raise BinningAlgorithmError(msg)
        assert "Tree" in str(exc_info.value)
        assert "切分点" in str(exc_info.value)
    
    def test_exception_is_woebin_error(self):
        """测试是 WOEBinError 子类"""
        with pytest.raises(WOEBinError):
            raise BinningAlgorithmError('test')


class TestWOEComputationError:
    """测试 WOE 计算异常"""
    
    def test_exception_message(self):
        """测试异常消息"""
        with pytest.raises(WOEComputationError) as exc_info:
            raise WOEComputationError("无法计算 WOE：好样本和坏样本均为 0")
        assert "无法计算 WOE" in str(exc_info.value)
        assert "好样本和坏样本均为 0" in str(exc_info.value)
    
    def test_exception_is_woebin_error(self):
        """测试是 WOEBinError 子类"""
        with pytest.raises(WOEBinError):
            raise WOEComputationError('test')


class TestExceptionHierarchy:
    """测试异常层次结构"""
    
    def test_all_exceptions_are_woebin_error(self):
        """测试所有异常都是 WOEBinError 子类"""
        exceptions = [
            ConstantVariableError,
            TooManyCategoriesError,
            InvalidBreaksError,
            DataValidationError,
            BinningAlgorithmError
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, WOEBinError)
    
    def test_can_catch_specific_exception(self):
        """测试可以捕获特定异常"""
        with pytest.raises(ConstantVariableError):
            raise ConstantVariableError('test')
        with pytest.raises(TooManyCategoriesError):
            raise TooManyCategoriesError('test', 100, 50)
    
    def test_can_catch_base_exception_for_all(self):
        """测试可以用基类捕获所有异常"""
        exceptions_to_test = [
            (ConstantVariableError, ('test',)),
            (TooManyCategoriesError, ('test', 100, 50)),
            (InvalidBreaksError, ('test',)),
            (DataValidationError, ('test',)),
            (BinningAlgorithmError, ('test',))
        ]
        for exc_class, args in exceptions_to_test:
            with pytest.raises(WOEBinError):
                raise exc_class(*args)

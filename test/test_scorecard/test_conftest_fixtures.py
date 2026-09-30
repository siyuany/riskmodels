# -*- encoding: utf-8 -*-
"""B-13 回归：conftest fixture 必须可被真实消费。

历史缺陷（W1 报告 B-13）：``data_with_constant_var`` 等 5 个 fixture 直接
调用 ``clean_data()``（fixture 函数的普通调用返回 ``_FixtureFunction``
对象），一旦被使用必然 ``AttributeError``；由于当时**没有任何测试消费
这些 fixture**，缺陷一直潜伏。

W2 修复：
1. conftest 中这 5 个 fixture 全部改为依赖注入形式
   （``def data_with_...(clean_data): ...``）；
2. 本文件真实消费全部数据类 fixture，把潜伏缺陷变成显式回归；
3. ``test_known_bugs.py::test_b13_conftest_fixtures_use_dependency_injection``
   静态钉住"依赖注入 + 至少一个消费者"两个不变量。
"""
import numpy as np
import pandas as pd
import pytest

from syriskmodels.scorecard import WOEBinFactory, woebin
from syriskmodels.scorecard.constants import VariableStatus
from syriskmodels.scorecard.utils.validation import check_uniques


def test_clean_data_shape_and_dtypes(clean_data):
    """clean_data：1000 行、列齐全、y 为 0/1 二值。"""
    assert isinstance(clean_data, pd.DataFrame)
    assert clean_data.shape[0] == 1000
    assert {'id', 'age', 'income', 'city', 'y'} <= set(clean_data.columns)
    assert set(clean_data['y'].unique()) <= {0, 1}
    assert clean_data['y'].nunique() == 2


def test_data_with_constant_var(data_with_constant_var):
    """常量列被识别为 CONSTANT，woebin 返回 'CONST' 状态字符串。"""
    df = data_with_constant_var
    assert (df['constant'] == 999).all()
    assert check_uniques(df['constant']) == VariableStatus.CONSTANT

    bins = woebin(df, y='y', x=['constant'], methods=['quantile'],
                  no_cores=1)
    assert bins['constant'] == 'CONST'


def test_data_with_too_many_categories(data_with_too_many_categories):
    """1000 个类别超过 max_cate_num=50 → TOO_MANY_CATEGORIES / 'TOO_MANY_VALUES'。"""
    df = data_with_too_many_categories
    assert df['many_cats'].nunique() == 1000
    assert check_uniques(df['many_cats'], 50) == \
        VariableStatus.TOO_MANY_CATEGORIES

    bins = woebin(df, y='y', x=['many_cats'], methods=['quantile'],
                  no_cores=1)
    assert bins['many_cats'] == 'TOO_MANY_VALUES'


def test_data_with_special_values(data_with_special_values):
    """-999 哨兵值单独成箱；income 的 NaN 走隐式 'missing' 特殊值路径。"""
    df = data_with_special_values
    assert int((df['age'] == -999).sum()) == 50
    assert int(df['income'].isna().sum()) == 30

    bins = woebin(df, y='y', x=['age', 'income'], methods=['quantile'],
                  initial_bins=10,
                  special_values={'age': ['-999']}, no_cores=1)

    age_bins = bins['age']
    sv = age_bins[age_bins['is_special_values']]
    assert sv['bin'].tolist() == ['-999']
    assert int(sv['count'].iloc[0]) == 50

    income_bins = bins['income']
    assert 'missing' in set(income_bins.loc[
        income_bins['is_special_values'], 'bin'])


def test_data_with_mixed_types(data_with_mixed_types):
    """字符列混入数值（object 列）：钉住当前生产行为。

    现状：``QuantileInitBin`` 对类别列做 ``np.unique`` 排序，str/int 混合
    的 object 列会抛 ``TypeError: '<' not supported between instances of
    'str' and 'int'``。这是**已知限制**（混合类型变量未做归一化，记录于
    W2 报告 W3 候选项），此处钉住行为防止静默变化；fixture 本身可被消费
    （B-13 的核心诉求）即已达成。
    """
    df = data_with_mixed_types
    assert df['city'].dtype == object
    assert not pd.api.types.is_numeric_dtype(df['city'])
    assert int((df['city'] == -999).sum()) == 10

    with pytest.raises(TypeError):
        woebin(df, y='y', x=['city'], methods=['quantile', 'tree'],
               bin_num_limit=4, no_cores=1)


def test_data_all_nan(data_all_nan):
    """全 NaN 列：唯一值数为 0 → CONSTANT，woebin 返回 'CONST'。"""
    df = data_all_nan
    assert df['all_nan'].isna().all()
    assert check_uniques(df['all_nan']) == VariableStatus.CONSTANT

    bins = woebin(df, y='y', x=['all_nan'], methods=['quantile'],
                  no_cores=1)
    assert bins['all_nan'] == 'CONST'


def test_data_single_sample(data_single_sample):
    """单样本：唯一值数 1 → CONSTANT。"""
    assert check_uniques(data_single_sample['age']) == VariableStatus.CONSTANT


def test_data_empty(data_empty):
    """空数据集：唯一值数 0 → CONSTANT（check_uniques 不崩溃）。"""
    assert len(data_empty) == 0
    assert check_uniques(data_empty['age']) == VariableStatus.CONSTANT


def test_dtm_numeric_direct_binner_call(dtm_numeric):
    """dtm 格式数据直接驱动组合分箱器（数值型路径）。"""
    binner = WOEBinFactory.build(['quantile', 'tree'], initial_bins=10,
                                 bin_num_limit=3)
    result = binner(dtm_numeric)
    assert isinstance(result, pd.DataFrame)
    assert int(result['count'].sum()) == len(dtm_numeric)
    assert result['total_iv'].nunique() == 1


def test_dtm_categorical_direct_binner_call(dtm_categorical):
    """dtm 格式数据直接驱动组合分箱器（类别型路径）。"""
    binner = WOEBinFactory.build(['quantile', 'tree'], bin_num_limit=3)
    result = binner(dtm_categorical)
    assert isinstance(result, pd.DataFrame)
    assert int(result['count'].sum()) == len(dtm_categorical)


def test_dtm_with_special_values_direct_binner_call(dtm_with_special_values):
    """dtm 含 -999/-1/NaN：特殊值路径由显式 special_values 驱动。

    注入 NaN 后 value 列升格为 float64，且特殊值集合含 'missing'（NaN）→
    数值特殊值标签为 float 形式（'-999.0'/'-1.0'，见 B-4/B-5 语义）。
    """
    binner = WOEBinFactory.build(['quantile', 'tree'], initial_bins=10,
                                 bin_num_limit=3)
    result = binner(dtm_with_special_values,
                    special_values=['-999', '-1', 'missing'])
    assert isinstance(result, pd.DataFrame)
    sv = result[result['is_special_values']]
    assert set(sv['bin']) == {'missing', '-999.0', '-1.0'}
    assert int(result['count'].sum()) == len(dtm_with_special_values)


def test_binning_result_fixtures_shape(binning_result_numeric,
                                      binning_result_categorical):
    """两个 binning_result fixture 的结构契约（供下游测试复用）。"""
    assert list(binning_result_numeric.columns) == [
        'variable', 'bin_chr', 'good', 'bad']
    assert (binning_result_numeric['good'] > 0).all()
    assert list(binning_result_categorical.columns) == [
        'variable', 'bin_chr', 'good', 'bad']
    assert binning_result_categorical['bin_chr'].tolist() == [
        'A%,%B', 'C', 'D']

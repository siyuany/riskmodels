# -*- encoding: utf-8 -*-
"""
pytest 测试配置和 fixture
"""
import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def clean_data():
    """干净数据（无异常）"""
    np.random.seed(42)
    return pd.DataFrame({
        'id': range(1000),
        'age': np.random.randint(20, 60, 1000),
        'income': np.random.normal(50000, 15000, 1000),
        'city': np.random.choice(['A', 'B', 'C'], 1000),
        'y': np.random.choice([0, 1], 1000)
    })


@pytest.fixture
def data_with_constant_var():
    """包含常量变量的数据"""
    df = clean_data()
    df['constant'] = 999  # 常量列
    return df


@pytest.fixture
def data_with_too_many_categories():
    """类别过多的数据"""
    df = clean_data()
    df['many_cats'] = [f'cat_{i}' for i in range(1000)]  # 1000 个类别
    return df


@pytest.fixture
def data_with_special_values():
    """包含特殊值的数据"""
    df = clean_data()
    df.loc[df.sample(50, random_state=42).index, 'age'] = -999  # 异常值
    df.loc[df.sample(30, random_state=42).index, 'income'] = np.nan  # 空值
    return df


@pytest.fixture
def data_with_mixed_types():
    """混合类型异常数据"""
    df = clean_data()
    # 字符列中混入数值
    df.loc[df.sample(10, random_state=42).index, 'city'] = -999
    return df


@pytest.fixture
def data_all_nan():
    """全空值列"""
    df = clean_data()
    df['all_nan'] = np.nan
    return df


@pytest.fixture
def data_single_sample():
    """单样本数据"""
    return pd.DataFrame({
        'age': [30],
        'income': [50000],
        'city': ['A'],
        'y': [1]
    })


@pytest.fixture
def data_empty():
    """空数据集"""
    return pd.DataFrame(columns=['age', 'income', 'city', 'y'])


@pytest.fixture
def dtm_numeric():
    """数值型变量 dtm 格式数据"""
    np.random.seed(42)
    return pd.DataFrame({
        'variable': 'age',
        'y': np.random.choice([0, 1], 100),
        'value': np.random.randint(20, 60, 100)
    })


@pytest.fixture
def dtm_categorical():
    """类别型变量 dtm 格式数据"""
    np.random.seed(42)
    return pd.DataFrame({
        'variable': 'city',
        'y': np.random.choice([0, 1], 100),
        'value': np.random.choice(['A', 'B', 'C', 'D'], 100)
    })


@pytest.fixture
def dtm_with_special_values():
    """包含特殊值的 dtm 数据"""
    np.random.seed(42)
    df = pd.DataFrame({
        'variable': 'age',
        'y': np.random.choice([0, 1], 100),
        'value': np.random.randint(20, 60, 100)
    })
    # 添加特殊值
    df.loc[0, 'value'] = -999
    df.loc[1, 'value'] = np.nan
    df.loc[2, 'value'] = -1
    return df


@pytest.fixture
def binning_result_numeric():
    """数值型分箱结果示例"""
    return pd.DataFrame({
        'variable': ['age', 'age', 'age'],
        'bin_chr': ['[-inf, 20)', '[20, 40)', '[40, inf)'],
        'good': [30, 25, 20],
        'bad': [10, 15, 20]
    })


@pytest.fixture
def binning_result_categorical():
    """类别型分箱结果示例"""
    return pd.DataFrame({
        'variable': ['city', 'city', 'city'],
        'bin_chr': ['A%,%B', 'C', 'D'],
        'good': [25, 25, 25],
        'bad': [15, 15, 15]
    })

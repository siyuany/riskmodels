# -*- encoding: utf-8 -*-

import numpy as np
import pandas as pd


def sample_stats(df, target):
    """
    统计样本中好坏样本数量及坏样本占比，使用示例如下：

    >>> sample_df = pd.DataFrame({
    ...     'id': np.arange(10),
    ...     'prod': ['a', 'a', 'a', 'b', 'b', 'b', 'b', 'c', 'c', 'c'],
    ...     'y': [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]})
    >>> sample_df
       id prod  y
    0   0    a  0
    1   1    a  0
    2   2    a  1
    3   3    b  1
    4   4    b  0
    5   5    b  1
    6   6    b  0
    7   7    c  1
    8   8    c  0
    9   9    c  0
    >>> sample_stats(sample_df, target='y')
    TotalCnt     10
    GoodCnt       6
    BadCnt        4
    BadRate     0.4
    dtype: object
    >>> sample_df.groupby('prod').apply(sample_stats, target='y')
          TotalCnt  GoodCnt  BadCnt   BadRate
    prod
    a            3        2       1  0.333333
    b            4        2       2  0.500000
    c            3        2       1  0.333333

    Args:
        df: pd.DataFrame, 样本集
        target: 目标变量名

    Returns:
        pd.Series，各个统计量组成的数列

    """
    y = df[target]
    ttl_cnt = len(y)
    good_cnt = np.sum(y == 0)
    bad_cnt = np.sum(y == 1)
    bad_rate = bad_cnt / (good_cnt + bad_cnt)
    return pd.Series([ttl_cnt, good_cnt, bad_cnt, bad_rate],
                     index=['TotalCnt', 'GoodCnt', 'BadCnt', 'BadRate'],
                     dtype='object')


def monotonic(series: pd.Series) -> str:
    """
    判断一组序列(pd.Series)的单调类型，包括以下五种类型：

    * 'increasing' - 单调递增
    * 'decreasing' - 单调递减
    * 'up_u_shape' - 先单调递减，后单调递增
    * 'down_u_shape' - 先单调递增，后单调递增

    Args:
        series: pd.Series, 数值型数组序列

    Returns:
        str, 单调类型

    """
    if series.is_monotonic_increasing:
        return 'increasing'
    elif series.is_monotonic_decreasing:
        return 'decreasing'
    else:
        for i in range(2, len(series)):
            if series[:i].is_monotonic_increasing and \
                    series[i:].is_monotonic_decreasing:
                return 'down_u_shape'
            elif series[:i].is_monotonic_decreasing and \
                    series[i:].is_monotonic_increasing:
                return 'up_u_shape'
    return 'non_monotonic'

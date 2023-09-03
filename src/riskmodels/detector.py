#!/usr/bin/python
"""
数据探索，包括变量类型、空值率、分布等统计类数据
本模块来自 toad (https://github.com/amphibian-dev/toad)
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import NoneType
from typing import Union

import pandas as pd


def getTopValues(series, top=5, reverse=False):
    """Get top/bottom n values
    Args:
        series (Series): data series
        top (number): number of top/bottom n values
        reverse (bool): it will return bottom n values if True is given
    Returns:
        Series: Series of top/bottom n values and percentage. ['value:percent', None]
    """
    itype = 'top'
    counts = series.value_counts()
    counts = list(zip(counts.index, counts, counts.divide(series.size)))

    if reverse:
        counts.reverse()
        itype = 'bottom'

    template = "{0[0]}:{0[2]:.2%}"
    indexs = [itype + str(i + 1) for i in range(top)]
    values = [
        template.format(counts[i]) if i < len(counts) else None
        for i in range(top)
    ]

    return pd.Series(values, index=indexs)


def getDescribe(series, percentiles=[.25, .5, .75]):
    """Get describe of series
    Args:
        series (Series): data series
        percentiles: the percentiles to include in the output
    Returns:
        Series: the describe of data include mean, std, min, max and percentiles
    """
    d = series.describe(percentiles)
    return d.drop('count')


def countBlank(series, blanks=None):
    """Count number and percentage of blank values in series
    Args:
        series (Series): data series
        blanks (list): list of blank values
    Returns:
        number: number of blanks
        str: the percentage of blank values
    """
    if blanks is None:
        blanks = []

    if len(blanks) > 0:
        isnull = series.replace(blanks, None).isnull()
    else:
        isnull = series.isnull()
    n = isnull.sum()
    ratio = isnull.mean()

    return n, "{0:.2%}".format(ratio)


def isNumeric(series):
    """Check if the series' type is numeric
    Args:
        series (Series): data series
    Returns:
        bool
    """
    return series.dtype.kind in 'ifc'


def _detect(name, series):
    numeric_index = [
        'mean', 'std', 'min', '1%', '10%', '50%', '75%', '90%', '99%', 'max'
    ]
    discrete_index = [
        'top1', 'top2', 'top3', 'top4', 'top5', 'bottom5', 'bottom4', 'bottom3',
        'bottom2', 'bottom1'
    ]

    details_index = [
        numeric_index[i] + '_or_' + discrete_index[i]
        for i in range(len(numeric_index))
    ]
    details = []

    if isNumeric(series):
        desc = getDescribe(series, percentiles=[.01, .1, .5, .75, .9, .99])
        details = desc.tolist()
    else:
        top5 = getTopValues(series)
        bottom5 = getTopValues(series, reverse=True)
        details = top5.tolist() + bottom5[::-1].tolist()

    # print(details_index)
    nblank, pblank = countBlank(series)

    row = pd.Series(
        index=['type', 'size', 'missing', 'unique'] + details_index,
        data=[series.dtype, series.size, pblank,
              series.nunique()] + details)

    row.name = name
    return row


def detect(dataframe, n_cores: Union[NoneType, int] = None):
    """ Detect data
    Args:
        dataframe (DataFrame): data that will be detected
        n_cores (int): parallel computing if n_cores is None or n_cores > 1.
    Returns:
        DataFrame: report of detecting
    """

    rows = []
    n_cores = int(n_cores) if n_cores >= 1 else None
    n_cols = dataframe.shape[1]
    if n_cols < 10 or n_cores == 1:
        for name, series in dataframe.items():
            row = _detect(name, series)
            rows.append(row)
    else:
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            futures = [
                executor.submit(_detect, name, series)
                for name, series in dataframe.items()
            ]
            for fut in as_completed(futures):
                rows.append(fut.result())

    return pd.DataFrame(rows)

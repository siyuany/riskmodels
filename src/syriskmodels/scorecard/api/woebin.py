# -*- encoding: utf-8 -*-
"""
WOE 分箱 API 模块

提供分箱主函数 woebin
"""
import time
import itertools
import multiprocessing as mp
from typing import Dict, List, Union, Optional, Any

import numpy as np
import pandas as pd

import syriskmodels.logging as logging
from syriskmodels.utils import str_to_list
from syriskmodels.scorecard.core.factory import WOEBinFactory
from syriskmodels.scorecard.utils.validation import (
    check_y,
    x_variable,
    check_breaks_list,
    check_special_values
)


def woebin(
    dt: pd.DataFrame,
    y: Union[str, List[str]],
    x: Optional[Union[str, List[str]]] = None,
    var_skip: Optional[Union[str, List[str]]] = None,
    breaks_list: Optional[Dict[str, List]] = None,
    special_values: Optional[Union[List, Dict[str, List]]] = None,
    positive: Union[int, float] = 1,
    no_cores: Optional[int] = None,
    methods: Optional[List[Union[str, type]]] = None,
    max_cate_num: int = 50,
    replace_blank: Union[float, int] = np.nan,
    **kwargs
) -> Dict[str, Union[pd.DataFrame, str]]:
    """WOE 分箱主函数
    
    对数据集中的变量进行 WOE 分箱，返回每个变量的分箱统计结果。
    
    参数:
        dt: 包含目标变量和解释变量的数据框
        y: 目标变量名
        x: 解释变量名列表，默认为除 y 外的所有列
        var_skip: 需要跳过的变量列表
        breaks_list: 用户自定义切分点字典
        special_values: 特殊值列表或字典
        positive: 正样本标识值，默认 1
        no_cores: 多进程数量，None 时自动检测
        methods: 分箱方法列表，默认 ['quantile', 'tree']
        max_cate_num: 类别变量最大允许类别数，默认 50
        replace_blank: 空字符串替换值，默认 np.nan
        **kwargs: 传递给分箱器的其他参数
    
    返回:
        分箱结果字典，key 为变量名，value 为分箱统计 DataFrame
        特殊情况可能返回字符串：'CONST' (常量变量) 或 'TOO_MANY_VALUES' (类别过多)
    
    示例:
        >>> bins = woebin(df, y='target', x=['age', 'income'])
        >>> bins['age']
           variable  bin_chr  count  count_distr  good  bad  ...
    """
    if methods is None:
        methods = ['quantile', 'tree']
    
    # start time
    start_time = time.time()
    
    # arguments
    dt = dt.copy(deep=True)
    y = str_to_list(y)
    x = str_to_list(x)
    if x is not None:
        dt = dt[y + x]
    
    # check y
    dt = check_y(dt, y[0], positive=positive)
    
    # x variable names
    xs = x_variable(dt, y, x, var_skip)
    xs_len = len(xs)
    
    # breaks_list
    breaks_list = check_breaks_list(breaks_list)
    
    # special_values
    special_values = check_special_values(special_values, xs)
    
    # binning for each x variable
    # loop on xs
    if (no_cores is None) or (no_cores < 1):
        all_cores = mp.cpu_count() - 1
        no_cores = int(
            np.ceil(xs_len / 5 if xs_len / 5 < all_cores else all_cores * 0.9)
        )
    
    # y list to str
    y = y[0]
    
    woe_bin = WOEBinFactory.build(methods, **kwargs)
    
    tasks = [
        (
            # dtm definition
            pd.DataFrame({
                'y': dt[y],
                'variable': x_i,
                'value': dt[x_i]
            }),
            # breaks_list
            breaks_list.get(x_i),
            # special_values
            special_values.get(x_i),
            max_cate_num,
            replace_blank,
        ) for x_i in xs
    ]
    
    logging.info(f'开始分箱，特征数 {len(tasks)}，样本数 {len(dt)}')
    
    if no_cores == 1:
        bins = dict(zip(xs, itertools.starmap(woe_bin, tasks)))
    else:
        pool = mp.Pool(processes=no_cores)
        bins = dict(zip(xs, pool.starmap(woe_bin, tasks)))
        pool.close()
    
    # running time
    running_time = time.time() - start_time
    logging.info('分箱完成：Binning on {} rows and {} columns in {}'.format(
        dt.shape[0], len(xs), time.strftime("%H:%M:%S",
                                            time.gmtime(running_time))))
    
    return bins

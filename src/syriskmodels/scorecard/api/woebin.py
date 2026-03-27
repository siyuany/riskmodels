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

    对数据集中的变量进行 WOE (Weight of Evidence) 分箱，返回每个变量的
    分箱统计结果，包含 WOE 值、IV 值等指标。

    参数:
        dt: 包含目标变量和解释变量的数据框
        y: 目标变量名（0/1 二分类，1 为正样本）
        x: 解释变量名列表，默认为除 y 外的所有列
        var_skip: 需要跳过的变量列表
        breaks_list: 用户自定义切分点字典，格式为 ``{变量名: [切分点列表]}``
        special_values: 特殊值列表或字典。列表形式应用于所有变量，
            字典形式为 ``{变量名: [特殊值列表]}``
        positive: 正样本标识值，默认 1
        no_cores: 多进程数量，None 时自动检测 CPU 核数
        methods: 分箱方法列表，默认 ``['quantile', 'tree']``。
            首元素必须为无监督细分箱方法 (``'quantile'`` 或 ``'hist'``)，
            后续为粗分箱方法 (``'tree'`` 或 ``'chi2'``)。
            常见组合:

            - ``['quantile', 'tree']``: 等频细分箱 + 树粗分箱（默认）
            - ``['quantile', 'chi2']``: 等频细分箱 + ChiMerge 粗分箱
            - ``['quantile', 'tree', 'chi2']``: 等频 → 树 → ChiMerge 三级分箱
            - ``['quantile']``: 仅等频分箱（纯无监督）

        max_cate_num: 类别变量最大允许类别数，超过则跳过，默认 50
        replace_blank: 空字符串替换值，默认 ``np.nan``
        **kwargs: 传递给分箱器的其他参数，常用参数包括:

            - ``initial_bins`` (int): 细分箱的数量，默认 20
            - ``bin_num_limit`` (int): 最终分箱的最大数量（不含特殊值），默认 5
            - ``count_distr_limit`` (float): 分箱样本占总样本最小比例，默认 0.05
            - ``stop_limit`` (float): 分箱停止条件阈值，默认 0.05
            - ``ensure_monotonic`` (bool): 是否保证单调性（仅树分箱），默认 False

    返回:
        Dict[str, Union[pd.DataFrame, str]]，key 为变量名，value 为:

        - pd.DataFrame: 分箱结果，包含 variable, bin, bin_chr, count,
          count_distr, good, bad, badprob, woe, bin_iv, total_iv,
          breaks, is_special_values 等列
        - ``'CONST'``: 常量变量（被跳过）
        - ``'TOO_MANY_VALUES'``: 类别过多（被跳过）

    示例:
        >>> from syriskmodels.scorecard import woebin, sc_bins_to_df
        >>> bins = woebin(df, y='target', x=['age', 'income'],
        ...              methods=['quantile', 'tree'],
        ...              bin_num_limit=5, count_distr_limit=0.05)
        >>> woe_df, iv_df = sc_bins_to_df(bins)
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
        # 确保至少有一个进程
        no_cores = max(no_cores, 1)
    
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

# -*- encoding: utf-8 -*-
"""
WOE 转换 API 模块

提供 woebin_ply, woebin_breaks 等转换函数
"""
import time
import itertools
import multiprocessing as mp
from typing import Dict, List, Union, Tuple

import numpy as np
import pandas as pd

import syriskmodels.logging as logging
from syriskmodels.scorecard.core.base import WOEBin


def woebin_ply(
    dt: pd.DataFrame,
    bins: Dict[str, Union[pd.DataFrame, str]],
    no_cores: int = None,
    replace_blank: bool = False,
    value: str = 'woe'
) -> pd.DataFrame:
    """应用 WOE 分箱结果转换数据

    将 ``woebin()`` 返回的分箱结果应用到数据集，将原始值转换为 WOE 值、
    分箱索引或分箱区间。转换后的列名带有后缀 ``_woe``、``_index`` 或 ``_bin``。

    参数:
        dt: 包含变量原始值的数据框，列名需与分箱结果中的变量名一致
        bins: ``woebin()`` 返回的分箱结果字典
        no_cores: 多进程数量，None 时自动检测
        replace_blank: 是否将空字符串 ``''`` 替换为 ``np.nan``
        value: 返回值类型，可选 ``['woe', 'index', 'bin']``

            - ``'woe'``: 将原始值替换为 WOE 值，列名为 ``变量名_woe``
            - ``'index'``: 将原始值替换为分箱索引 (0, 1, 2,...)，列名为 ``变量名_index``
            - ``'bin'``: 返回分箱区间文本，数值型为 ``[a,b)``，类别型为 ``a%,%b``，
              列名为 ``变量名_bin``

    返回:
        pd.DataFrame，包含:

        - 入参 dt 中**不在** bins 中的原始列（保持不变）
        - bins 中匹配到的变量，按 ``value`` 参数转换后的新列

    示例:
        >>> bins = woebin(train_df, y='target')
        >>> train_woe = woebin_ply(train_df, bins, value='woe')
        >>> train_woe.columns  # 原始列被替换为 age_woe, income_woe 等
        >>> train_bin = woebin_ply(train_df, bins, value='bin')
        >>> train_bin['age_bin'].head()  # 输出: [-inf,25), [25,35), ...
    """
    # start time
    start_time = time.time()
    
    # x variables
    x_vars_bin = bins.keys()
    x_vars_dt = dt.columns.tolist()
    x_vars = list(set(x_vars_bin).intersection(x_vars_dt))
    n_x = len(x_vars)
    
    # initial data set
    dat = dt.loc[:, list(set(x_vars_dt) - set(x_vars))].copy()
    
    if no_cores is None or no_cores < 1:
        all_cores = mp.cpu_count() - 1
        no_cores = int(np.ceil(n_x / 5 if n_x / 5 < all_cores else all_cores * 0.9))
    no_cores = max(no_cores, 1)
    
    tasks = [
        (
            pd.DataFrame({
                'y': 0,  # 不重要
                'variable': var,
                'value': dt[var]
            }),
            bins[var],
            value
        ) for var in x_vars
    ]
    
    if no_cores == 1:
        dat_suffix = list(itertools.starmap(WOEBin.apply, tasks))
    else:
        pool = mp.Pool(processes=no_cores)
        dat_suffix = pool.starmap(WOEBin.apply, tasks)
        pool.close()
    
    dat = pd.concat([dat] + dat_suffix, axis=1)
    
    # running time
    running_time = time.time() - start_time
    logging.info('Woe transformation on {} rows and {} columns in {}'.format(
        dt.shape[0], n_x, time.strftime("%H:%M:%S", time.gmtime(running_time))))
    
    return dat


def woebin_breaks(
    bins: Dict[str, Union[pd.DataFrame, str]]
) -> Tuple[Dict[str, List], Dict[str, List]]:
    """从 woebin 返回结果中提取切分点及特殊值（向后兼容 legacy 接口）

    参数:
        bins: woebin 函数的返回结果

    返回:
        (breaks, special_values) 元组：
        - breaks: 各变量数值 / 类别切分点列表（不含特殊值）
        - special_values: 各变量对应的特殊值列表（若无则不包含该键）
    """

    def _get_breaks(binning: pd.DataFrame) -> Dict[str, List]:
        # 提取特殊值
        if 'is_special_values' in binning.columns and binning['is_special_values'].any():
            special_values = binning[binning['is_special_values']]['breaks']
            special_values = special_values.tolist()

            # 与 legacy 行为保持一致：剔除 'missing'
            if 'missing' in special_values:
                special_values.remove('missing')
            if len(special_values) == 0:
                special_values = None
        else:
            special_values = None

        # 提取普通切分点
        if 'is_special_values' in binning.columns:
            brks = binning[~binning['is_special_values']]['breaks']
        else:
            brks = binning['breaks']
        brks = brks.tolist()

        return {'breaks': brks, 'special_values': special_values}

    brk_spcs = {
        key: _get_breaks(value)
        for key, value in bins.items()
        if isinstance(value, pd.DataFrame)
    }

    breaks = {k: v['breaks'] for k, v in brk_spcs.items()}
    special_values = {
        k: v['special_values']
        for k, v in brk_spcs.items() if v['special_values']
    }
    return breaks, special_values

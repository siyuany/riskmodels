# -*- encoding: utf-8 -*-
"""
WOE 转换 API 模块

提供 woebin_ply, woebin_breaks 等转换函数
"""
import time
import itertools
from typing import Dict, List, Union, Tuple, Optional

import pandas as pd

import syriskmodels.logging as logging
from syriskmodels.utils import resolve_no_cores, parallel_starmap
from syriskmodels.scorecard.core.base import WOEBin


def woebin_ply(
    dt: pd.DataFrame,
    bins: Dict[str, Union[pd.DataFrame, str]],
    no_cores: int = 1,
    replace_blank: bool = False,
    value: str = 'woe',
    parallel_timeout: Optional[float] = None
) -> pd.DataFrame:
    """应用 WOE 分箱结果转换数据

    将 ``woebin()`` 返回的分箱结果应用到数据集，将原始值转换为 WOE 值、
    分箱索引或分箱区间。转换后的列名带有后缀 ``_woe``、``_index`` 或 ``_bin``。

    参数:
        dt: 包含变量原始值的数据框，列名需与分箱结果中的变量名一致
        bins: ``woebin()`` 返回的分箱结果字典
        no_cores: 多进程数量，默认 1（串行）。W2/B-11：``None`` 或 ``<1``
            一律视为 1；只有**显式**传入 ``>1`` 才启用并行。交互式环境 +
            spawn 语义时显式并行会自动回退串行并发出 ``UserWarning``
        replace_blank: 是否将空字符串 ``''`` 替换为 ``np.nan``
        value: 返回值类型，可选 ``['woe', 'index', 'bin']``
        parallel_timeout: 显式并行（``no_cores>1``）时的整体超时秒数，
            默认 None（不限时）。超时终止进程池并抛 ``TimeoutError``；
            worker 内异常会传播回主进程重新抛出（B-11）

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
    
    # B-11：默认串行；None/<1 视为 1，仅显式 >1 才启用并行
    no_cores = resolve_no_cores(no_cores)
    
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
        dat_suffix = parallel_starmap(WOEBin.apply, tasks, no_cores,
                                      timeout=parallel_timeout)
    
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

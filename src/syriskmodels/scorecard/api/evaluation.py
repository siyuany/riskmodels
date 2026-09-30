# -*- encoding: utf-8 -*-
"""
评估 API 模块

提供 woebin_psi, woebin_plot 等评估函数
"""
from typing import Dict, Union, List, Optional
import numpy as np
import pandas as pd

import syriskmodels.logging as logging
from syriskmodels.evaluate import psi


def woebin_psi(
    df_base: pd.DataFrame,
    df_cmp: pd.DataFrame,
    bins: Dict[str, Union[pd.DataFrame, str]]
) -> pd.DataFrame:
    """计算变量 PSI（保持与 legacy 接口和行为一致）

    W2 修复（B-9）：旧实现用 ``pd.pivot_table`` 汇总两侧计数，单侧缺失的
    分箱会得到 NaN 分布，再被 ``psi()`` 的 NaN→0 + 重归一化**静默放大**。
    现在对两侧分箱计数取**并集** reindex，缺失侧显式 ``fillna(0)`` 后再
    归一化：

    * 正常场景（两侧分箱齐全）输出与旧实现完全一致；
    * 单侧缺失场景给出有限、可解释的 PSI（0 占比经 ``psi()`` 的 epsilon
      平滑），并触发 ``UserWarning`` 明确告警。

    参数:
        df_base: 基准数据集，一般为训练集
        df_cmp: 比较数据集，一般为测试集或 OOT
        bins: 变量分箱结果，由 `woebin` 返回

    返回:
        包含 variable, bin, base_distr, cmp_distr, psi 五列的 DataFrame
    """
    import warnings

    from syriskmodels.scorecard.api.transform import woebin_ply

    # 使用 bin 形式应用分箱
    X_base = woebin_ply(df_base, bins, value='bin', replace_blank=False)
    X_cmp = woebin_ply(df_cmp, bins, value='bin', replace_blank=False)

    vars_base = [v for v in X_base.columns if v.endswith('_bin')]
    vars_cmp = [v for v in X_cmp.columns if v.endswith('_bin')]
    # 排序保证输出行顺序确定（旧实现依赖 set 交集的哈希顺序）
    variables = sorted(set(vars_base).intersection(vars_cmp))

    psi_dfs: List[pd.DataFrame] = []

    for variable in variables:
        base_counts = X_base[variable].value_counts()
        cmp_counts = X_cmp[variable].value_counts()
        # 两侧分箱并集（Index.union 保持排序，与旧 pivot_table 行序一致）
        all_bins = base_counts.index.union(cmp_counts.index)

        only_base = all_bins.difference(cmp_counts.index)
        only_cmp = all_bins.difference(base_counts.index)
        if len(only_base) or len(only_cmp):
            warnings.warn(
                f'woebin_psi: 变量 {variable[:-4]} 存在单侧缺失分箱'
                f'（仅 base 侧: {list(only_base)}；仅 cmp 侧: {list(only_cmp)}），'
                f'缺失侧已按 0 占比显式补齐后计算 PSI。',
                UserWarning,
                stacklevel=2,
            )

        base_counts = base_counts.reindex(all_bins, fill_value=0)
        cmp_counts = cmp_counts.reindex(all_bins, fill_value=0)

        tmp = pd.DataFrame({
            'base': base_counts.to_numpy(),
            'cmp': cmp_counts.to_numpy(),
            'variable': variable[:-4],
            'bin': np.asarray(all_bins),
        })

        tmp = tmp.assign(
            base_distr=lambda x: x['base'] / x['base'].sum(),
            cmp_distr=lambda x: x['cmp'] / x['cmp'].sum(),
        ).assign(
            psi=lambda x: psi(x['base_distr'], x['cmp_distr'])
        )[['variable', 'bin', 'base_distr', 'cmp_distr', 'psi']]

        psi_dfs.append(tmp)

    return pd.concat(psi_dfs, ignore_index=True)


def _plot_single_bin(binx: pd.DataFrame, title: Optional[str], show_iv: bool):
    """绘制单个变量的分箱图，基本沿用 legacy 版样式。"""
    import matplotlib.pyplot as plt

    y_right_max = np.ceil(binx['badprob'].max() * 10)
    if y_right_max % 2 == 1:
        y_right_max = y_right_max + 1
    if y_right_max - binx['badprob'].max() * 10 <= 0.3:
        y_right_max = y_right_max + 2
    y_right_max = y_right_max / 10
    if (y_right_max > 1 or y_right_max <= 0 or pd.isna(y_right_max) or
            y_right_max is None):
        y_right_max = 1

    y_left_max = np.ceil(binx['count_distr'].max() * 10) / 10
    if (y_left_max > 1 or y_left_max <= 0 or pd.isna(y_left_max) or
            y_left_max is None):
        y_left_max = 1

    title_string = binx.loc[0, 'variable']
    if show_iv:
        title_string = f"{title_string}  (iv:{binx.loc[0, 'total_iv']:.4f})"
    if title is not None:
        title_string = f"{title}-{title_string}"

    ind = np.arange(len(binx.index))
    width = 0.35

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    p1 = ax1.bar(ind, binx['good_distr'], width, color=(24 / 254, 192 / 254, 196 / 254))
    p2 = ax1.bar(
        ind,
        binx['bad_distr'],
        width,
        bottom=binx['good_distr'],
        color=(246 / 254, 115 / 254, 109 / 254),
    )
    for i in ind:
        ax1.text(
            i,
            binx.loc[i, 'count_distr'] * 1.02,
            f"{round(binx.loc[i, 'count_distr'] * 100, 1)}%, "
            f"{binx.loc[i, 'count']}",
            ha='center',
        )

    ax2.plot(ind, binx['badprob'], marker='o', color='blue')
    for i in ind:
        ax2.text(
            i,
            binx.loc[i, 'badprob'] * 1.02,
            f"{round(binx.loc[i, 'badprob'] * 100, 1)}%",
            color='blue',
            ha='center',
        )

    ax1.set_ylabel('Bin count distribution')
    ax2.set_ylabel('Bad probability', color='blue')
    ax1.set_yticks(np.arange(0, y_left_max + 0.2, 0.2))
    ax2.set_yticks(np.arange(0, y_right_max + 0.2, 0.2))
    ax2.tick_params(axis='y', colors='blue')
    # W2 修复（B-2）：woebin 输出的分箱名列是 'bin'（legacy scorecardpy 时代
    # 为 'bin_chr'，本仓库的重构输出中不存在该列，旧代码在此处必然 KeyError）
    plt.xticks(ind, binx['bin'])
    plt.title(title_string, loc='left')
    plt.legend((p2[0], p1[0]), ('bad', 'good'), loc='upper right')

    return fig


def woebin_plot(
    bins: Dict[str, Union[pd.DataFrame, str]],
    x: Optional[Union[str, List[str]]] = None,
    title: Optional[str] = None,
    show_iv: bool = True
):
    """绘制分箱可视化图（兼容 legacy 行为）

    参数:
        bins: woebin 分箱结果（dict 或已拼接 DataFrame）
        x: 要绘制的变量名或列表，None 时绘制所有变量
        title: 图表标题前缀
        show_iv: 是否在标题中显示 IV 值

    返回:
        {variable: Figure} 的字典
    """
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError:
        logging.error("matplotlib 未安装，无法绘制分箱图")
        return None

    # 允许传入 dict 或 DataFrame，保持与 legacy 一致
    if isinstance(bins, dict):
        bins_df = pd.concat(bins, ignore_index=True)
    else:
        bins_df = bins.copy()

    # 计算 good_distr / bad_distr
    # W2 修复（B-2）：pandas 3 的 groupby.apply 不再把分组键保留为列，
    # 旧实现 apply(_gb_distr) 后 bins_df['variable'] 抛 KeyError。
    # 改用 transform 的等价向量化实现，图形语义不变。
    count_sum = bins_df.groupby('variable', observed=False)['count'].transform(
        'sum')
    bins_df = bins_df.assign(
        good_distr=bins_df['good'] / count_sum,
        bad_distr=bins_df['bad'] / count_sum,
    )

    if x is None:
        # 排序与旧 groupby.apply（按分组键排序输出）行为一致，且保证确定性
        xs = sorted(bins_df['variable'].unique())
    elif isinstance(x, str):
        xs = [x]
    else:
        xs = x

    plot_list = {}
    for var in xs:
        binx = bins_df[bins_df['variable'] == var].reset_index(drop=True)
        if binx.empty:
            continue
        plot_list[var] = _plot_single_bin(binx, title, show_iv)

    return plot_list

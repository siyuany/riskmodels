# -*- encoding: utf-8 -*-
"""
评估 API 模块

提供 woebin_psi, woebin_plot 等评估函数
"""
from typing import Dict, Union, List, Optional
import numpy as np
import pandas as pd

import syriskmodels.logging as logging


def woebin_psi(
    df_base: pd.DataFrame,
    df_cmp: pd.DataFrame,
    bins: Dict[str, Union[pd.DataFrame, str]]
) -> pd.DataFrame:
    """计算 PSI (Population Stability Index)
    
    比较两个数据集在各变量分箱上的分布差异。
    
    参数:
        df_base: 基准数据集
        df_cmp: 对比数据集
        bins: woebin 分箱结果
    
    返回:
        PSI 统计 DataFrame，包含每个变量的 PSI 值
    
    示例:
        >>> bins = woebin(train_df, y='target')
        >>> psi_df = woebin_psi(train_df, test_df, bins)
    """
    from syriskmodels.scorecard.api.transform import woebin_ply
    
    psi_list = []
    
    for var, binning in bins.items():
        if isinstance(binning, str):
            # 常量变量或类别过多，跳过
            continue
        
        # 计算基准分布
        base_binning = binning.copy()
        base_total = base_binning['count'].sum()
        base_binning['base_pct'] = base_binning['count'] / base_total
        
        # 计算对比分布
        cmp_values = df_cmp[var]
        cmp_binning = pd.cut(cmp_values, binning['breaks'], right=False, include_lowest=True)
        cmp_counts = cmp_binning.value_counts().reindex(binning['bin_chr'], fill_value=0)
        cmp_total = cmp_counts.sum()
        cmp_pct = cmp_counts / cmp_total
        
        # 计算 PSI
        base_binning['cmp_pct'] = cmp_pct.values
        base_binning['psi'] = (base_binning['base_pct'] - base_binning['cmp_pct']) * np.log(
            base_binning['base_pct'] / base_binning['cmp_pct']
        )
        
        psi = base_binning['psi'].sum()
        psi_list.append({
            'variable': var,
            'psi': psi
        })
    
    psi_df = pd.DataFrame(psi_list)
    psi_df.sort_values(by='psi', ascending=False, inplace=True)
    
    return psi_df


def woebin_plot(
    bins: Dict[str, Union[pd.DataFrame, str]],
    x: Optional[Union[str, List[str]]] = None,
    title: Optional[str] = None,
    show_iv: bool = True
):
    """绘制分箱可视化图
    
    参数:
        bins: woebin 分箱结果
        x: 要绘制的变量名或列表，None 时绘制所有变量
        title: 图表标题前缀
        show_iv: 是否在标题中显示 IV 值
    
    返回:
        matplotlib Figure 对象
    
    示例:
        >>> bins = woebin(df, y='target')
        >>> woebin_plot(bins, x=['age', 'income'])
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.error("matplotlib 未安装，无法绘制分箱图")
        return None
    
    if x is None:
        x = list(bins.keys())
    elif isinstance(x, str):
        x = [x]
    
    n_vars = len(x)
    if n_vars == 0:
        return None
    
    # 创建子图
    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 5 * n_vars))
    if n_vars == 1:
        axes = [axes]
    
    for i, var in enumerate(x):
        binning = bins[var]
        if isinstance(binning, str):
            continue
        
        ax = axes[i]
        
        # 绘制柱状图
        ind = np.arange(len(binning))
        width = 0.35
        
        ax.bar(ind, binning['good_distr'], width, label='Good', color='C0')
        ax.bar(ind, binning['bad_distr'], width, bottom=binning['good_distr'], label='Bad', color='C1')
        
        # 绘制坏样本率折线
        ax2 = ax.twinx()
        ax2.plot(ind, binning['badprob'], 'r-o', label='Bad Prob')
        
        # 设置标签
        ax.set_xlabel('Bin')
        ax.set_ylabel('Distribution')
        ax2.set_ylabel('Bad Probability')
        
        # 标题
        title_str = var
        if show_iv and 'total_iv' in binning.columns:
            title_str += f" (IV: {binning['total_iv'].iloc[0]:.4f})"
        if title:
            title_str = f"{title} - {title_str}"
        ax.set_title(title_str)
        
        # 图例
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # x 轴标签
        ax.set_xticks(ind)
        ax.set_xticklabels(binning['bin_chr'], rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

# -*- encoding: utf-8 -*-
"""
评分卡 API 模块

提供 make_scorecard, sc_bins_to_df 等评分卡相关函数
"""
from typing import Dict, List, Union, Tuple
import numpy as np
import pandas as pd

from syriskmodels.utils import monotonic


def sc_bins_to_df(
    sc_bins: Dict[str, Union[pd.DataFrame, str]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """将 woebin 返回的结果转换为 WOE 数据框和 IV 数据框
    
    参数:
        sc_bins: 由 woebin 返回的分箱结果字典
    
    返回:
        (woe_df, iv_df) 元组
        - woe_df: 包含所有变量分箱统计的 DataFrame
        - iv_df: 包含每个变量 IV 统计的 DataFrame
    
    示例:
        >>> bins = woebin(df, y='target')
        >>> woe_df, iv_df = sc_bins_to_df(bins)
    """
    woe_df = None
    
    for key, value in sc_bins.items():
        if isinstance(value, pd.DataFrame):
            if woe_df is None:
                woe_df = value
            else:
                woe_df = pd.concat([woe_df, value], axis=0, ignore_index=True)
    
    def iv_stats(x):
        iv = x.total_iv.max()
        badrate = x['bad'].sum() / x['count'].sum()
        lift = x.badprob / badrate
        iv_interval = None
        
        if iv < 0.02:
            iv_interval = '(0, 0.02)'
        elif iv < 0.05:
            iv_interval = '[0.02, 0.05)'
        elif iv < 0.08:
            iv_interval = '[0.05, 0.08)'
        elif iv < 0.1:
            iv_interval = '[0.08, 0.1)'
        elif iv < 0.2:
            iv_interval = '[0.1, 0.2)'
        else:
            iv_interval = '[0.2, +)'
        
        badrate = x[~x.is_special_values].badprob
        monotonic_type = monotonic(badrate)
        
        return pd.Series(
            [iv, iv_interval, monotonic_type, lift.max(), lift.min()],
            index=['IV', 'IV区间', '单调性', '最大Lift', '最小Lift'],
            dtype='object'
        )
    
    if woe_df is None:
        return None, None
    else:
        iv_df = woe_df.groupby(by='variable').apply(iv_stats)
        iv_df.sort_values(by='IV', ascending=False, inplace=True)
        return woe_df, iv_df


def make_scorecard(
    sc_bins: Dict[str, Union[pd.DataFrame, str]],
    coef: Dict[str, float],
    *,
    base_points: int = 600,
    base_odds: int = 50,
    pdo: int = 20
) -> pd.DataFrame:
    """生成评分卡
    
    参数:
        sc_bins: woebin 返回的分箱结果
        coef: 逻辑回归系数字典
        base_points: 基准分数，默认 600
        base_odds: 基准 odds，默认 50
        pdo: 翻倍 odds 的分数增量，默认 20
    
    返回:
        评分卡 DataFrame，包含 variable, bin, woe, score 列
    
    示例:
        >>> bins = woebin(df, y='target')
        >>> model = LogisticRegression().fit(X, y)
        >>> coef = {'const': -2.5, 'age_woe': 0.5, 'income_woe': 0.3}
        >>> scorecard = make_scorecard(bins, coef)
    """
    a = pdo / np.log(2)
    b = base_points - a * np.log(base_odds)
    
    base_score = -a * coef['const'] + b
    score_df = [
        pd.DataFrame({
            'variable': ['base score'],
            'bin': [''],
            'woe': [''],
            'score': [base_score]
        })
    ]
    
    for var in coef.keys():
        if var != 'const':
            # 变量名去掉 '_woe' 后缀
            var_name = var[:-4] if var.endswith('_woe') else var
            woe_df = sc_bins[var_name][['variable', 'bin', 'woe']].copy()
            woe_df['score'] = -a * coef[var] * woe_df['woe']
            score_df.append(woe_df)
    
    score_df = pd.concat(score_df, ignore_index=True)
    score_df['score'] = np.round(score_df['score'], 2)
    
    return score_df

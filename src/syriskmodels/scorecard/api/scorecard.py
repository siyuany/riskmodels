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
        iv_df = iv_df.sort_values(by='IV', ascending=False)
        return woe_df, iv_df


def make_scorecard(
    sc_bins: Dict[str, Union[pd.DataFrame, str]],
    coef: Dict[str, float],
    *,
    base_points: int = 600,
    base_odds: int = 50,
    pdo: int = 20
) -> pd.DataFrame:
    """将逻辑回归系数转换为评分卡

    基于分箱结果和逻辑回归系数，按照标准评分卡公式将 WOE 值转换为分数。
    公式: ``score_i = -(pdo/ln2) * coef_i * woe_i``，
    基础分: ``base_score = -(pdo/ln2) * intercept + (base_points - (pdo/ln2) * ln(base_odds))``

    参数:
        sc_bins: ``woebin()`` 返回的分箱结果字典
        coef: 逻辑回归系数字典，格式为
            ``{'const': 截距值, '变量名_woe': 系数值, ...}``。
            通常由 ``statsmodels.GLM.fit().params.to_dict()`` 获得。
            key 中的变量名需与 ``sc_bins`` 中的变量名对应（去掉 ``_woe`` 后缀匹配）。
        base_points: 基准分数，默认 600。当 odds 等于 ``base_odds`` 时的分数。
        base_odds: 基准 odds (好/坏比例)，默认 50
        pdo: odds 翻倍时的分数增量 (Points to Double the Odds)，默认 20

    返回:
        pd.DataFrame，包含以下列:

        - ``variable``: 变量名（首行为 ``'base score'``）
        - ``bin``: 分箱区间
        - ``woe``: WOE 值
        - ``score``: 该分箱对应的分数

    示例:
        >>> import statsmodels.api as sm
        >>> X = sm.add_constant(train_woe[selected_vars])
        >>> model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        >>> scorecard = make_scorecard(bins, model.params.to_dict(),
        ...                            base_points=600, base_odds=50, pdo=20)
        >>> scorecard.head()
          variable       bin   woe  score
        0  base score              452.31
        1         age  [-inf,25)  0.85  -12.34
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

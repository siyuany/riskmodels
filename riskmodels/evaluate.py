# -*- encoding: utf-8 -*-
from typing import Tuple, Union, Dict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from riskmodels.utils import sample_stats


def ks_score(y_true, y_pred) -> float:
    """计算模型 KS 统计量"""
    return ks_2samp(y_pred[y_true == 1], y_pred[y_true == 0])[0]


def model_perf(y_true, y_pred) -> Dict[str, float]:
    labeled_selector = (np.isin(y_true, [0, 1])) & (~y_pred.isna())
    y_true_new = y_true[labeled_selector]
    y_pred_new = y_pred[labeled_selector]

    auc = roc_auc_score(y_true_new, y_pred_new)
    if auc < 0.5:
        auc = 1 - auc
    ks = ks_score(y_true_new, y_pred_new)

    return {'auc': auc, 'ks': ks}


def model_eval(df, target, pred) -> pd.Series:
    perf = model_perf(df[target], df[pred])
    auc = perf['auc']
    ks = perf['ks']
    bad_rate = np.sum(df[target]) / np.sum(
        np.where(df[target].isin([0, 1]), 1, 0))
    return pd.Series([bad_rate, auc, ks], index=['bad_rate', 'auc', 'ks'])


def gains_table(
    y_true,
    y_score,
    split=None,
    breaks=None,
    ascending=True,
    return_breaks=False
) -> Union[Tuple[pd.DataFrame, np.ndarray], pd.DataFrame]:
    """
    A gains table includes a distribution of total, good, and bad cases by
    individual scores or score ranges.

    Args:
        y_true: 样本真实标签
        y_score: 样本对应模型分数
        split: 等分段数上限，默认 10。使用该参数时，不可同时指定 breaks 参数
        breaks: 分段点，默认为空。使用该参数时，不可同时指定 split 参数
        ascending: 为 True 时 gains table 按照分数由低到高统计；否则分数由高到低。
        return_breaks: 是否返回 breaks 分段点

    Returns:
        当 return_breaks=False 时，返回 gains_table (pd.DataFrame)；否则，返回
        元组 (gains_table, breaks)

    """
    if split is not None and breaks is not None:
        assert (
            'Either split or breaks should be None. Got split=%s, breaks=%s' %
            (split, breaks))
    if split is None and breaks is None:
        split = 10

    pred_df = pd.DataFrame({'score': y_score, 'y': y_true})
    if breaks is None:
        _, breaks = pd.qcut(pred_df.score,
                            np.linspace(0, 1, split + 1),
                            retbins=True)
    else:
        breaks = np.asarray(breaks)

    pred_df['score_seg'] = pd.cut(pred_df.score, bins=breaks)
    gains_df = pred_df.groupby('score_seg').apply(
        sample_stats, target='y').sort_index(ascending=ascending)
    gains_df = gains_df.assign(
        Odds=lambda x: x.GoodCnt / x.BadCnt,
        CumBadRate=lambda x: x.BadCnt.cumsum() / x.TotalCnt.cumsum(),
        BadPercent=lambda x: x.BadCnt / x.BadCnt.sum(),
        CumBadPercent=lambda x: x.BadCnt.cumsum() / x.BadCnt.sum(),
        GoodPercent=lambda x: x.GoodCnt.cumsum() / x.GoodCnt.sum(),
        CumGoodPercent=lambda x: x.GoodCnt.cumsum() / x.GoodCnt.sum(),
        TotalPercent=lambda x: x.TotalCnt / x.TotalCnt.sum(),
        Lift=lambda x: (x.BadCnt / x.TotalCnt) /
        (x.BadCnt.sum() / x.TotalCnt.sum())).assign(
            KS=lambda x: np.abs(x.CumBadPercent - x.CumGoodPercent))

    gains_df = gains_df[[
        'TotalCnt', 'GoodCnt', 'BadCnt', 'Odds', 'BadRate', 'Lift',
        'CumBadRate', 'BadPercent', 'CumBadPercent', 'GoodPercent',
        'CumGoodPercent', 'KS', 'TotalPercent'
    ]]

    if return_breaks:
        return gains_df, breaks
    else:
        return gains_df


def psi(base_distr, cmp_distr, epsilon=1e-3):
    """计算PSI (Population Stability Index)"""
    base_distr = np.asarray(base_distr)
    base_distr = base_distr / base_distr.sum()
    if np.any(base_distr == 0):
        base_distr = base_distr + epsilon
        base_distr = base_distr / base_distr.sum()

    cmp_distr = np.asarray(cmp_distr)
    cmp_distr = cmp_distr / cmp_distr.sum()
    if np.any(cmp_distr == 0):
        cmp_distr = cmp_distr + epsilon
        cmp_distr = cmp_distr / cmp_distr.sum()

    # calculate psi
    psi_value = (base_distr - cmp_distr) * np.log(base_distr / cmp_distr)
    return psi_value.sum()


def swap_analysis(df, base_model, compare_model, target_col):
    df = df[df[target_col].isin([0, 1]) & (~df[base_model].isna()) &
            (~df[compare_model].isna())].copy()
    df[base_model + '_seg'] = pd.qcut(df[base_model],
                                      np.linspace(0, 1, 11),
                                      duplicates='drop')
    df[compare_model + '_seg'] = pd.qcut(df[compare_model],
                                         np.linspace(0, 1, 11),
                                         duplicates='drop')
    total_cnt = pd.pivot_table(df,
                               index=base_model + '_seg',
                               columns=compare_model + '_seg',
                               values=target_col,
                               aggfunc='count')
    bad_cnt = pd.pivot_table(df,
                             index=base_model + '_seg',
                             columns=compare_model + '_seg',
                             values=target_col,
                             aggfunc='sum')
    bad_rate = bad_cnt / total_cnt

    return bad_rate


def swap_analysis_simple(df,
                         base_model,
                         compare_model,
                         target_col,
                         reject_ratio=0.2):
    df = df[df[target_col].isin([0, 1]) & (~df[base_model].isna()) &
            (~df[compare_model].isna())].copy()
    df[base_model + '_seg2'] = pd.qcut(df[base_model], [0, reject_ratio, 1.],
                                       duplicates='drop')
    df[compare_model + '_seg2'] = pd.qcut(df[compare_model],
                                          [0, reject_ratio, 1.],
                                          duplicates='drop')
    total_cnt2 = pd.pivot_table(df,
                                index=base_model + '_seg2',
                                columns=compare_model + '_seg2',
                                values=target_col,
                                aggfunc='count')
    bad_cnt2 = pd.pivot_table(df,
                              index=base_model + '_seg2',
                              columns=compare_model + '_seg2',
                              values=target_col,
                              aggfunc='sum')
    bad_rate2 = bad_cnt2 / total_cnt2
    swap_result = pd.concat([total_cnt2 / np.sum(total_cnt2.values), bad_rate2],
                            axis=1)
    swap_result.columns = [['total%', 'total%', 'bad_rate', 'bad_rate'],
                           swap_result.columns.tolist()]

    return swap_result

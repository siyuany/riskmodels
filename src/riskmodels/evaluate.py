# -*- encoding: utf-8 -*-
from typing import Tuple, Union, Dict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from riskmodels.utils import sample_stats


def ks_score(y_true, y_pred) -> float:
    """
    计算模型 KS 统计量

    Args
        - y_true: 真实Y标，0-1变量
        - y_pred: 预测值

    Returns
        KS值
    """
    return ks_2samp(y_pred[y_true == 1], y_pred[y_true == 0])[0]


def model_perf(y_true, y_pred) -> Dict[str, float]:
    """
    评估模型表现，该函数封装了`sklearn.metrics.roc_auc_score` 和 `ks_score` 两个函数
    Args:
        y_true:
        y_pred:

    Returns:
        `{'auc': auc score, 'ks': ks score}`

    """
    labeled_selector = (np.isin(y_true, [0, 1])) & (~y_pred.isna())
    y_true_new = y_true[labeled_selector]
    y_pred_new = y_pred[labeled_selector]

    if np.sum(y_true_new) > 0 and np.prod(y_true_new) == 0:
        auc = roc_auc_score(y_true_new, y_pred_new)
        if auc < 0.5:
            auc = 1 - auc
        ks = ks_score(y_true_new, y_pred_new)
    else:
        auc = ks = np.nan

    return {'auc': auc, 'ks': ks}


def model_eval(df, target, pred) -> pd.Series:
    perf = model_perf(df[target], df[pred])
    auc = perf['auc']
    ks = perf['ks']
    tmp_df = df[df[target].isin([0, 1])]
    bad_rate = np.mean(tmp_df[target])
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
        _, breaks = pd.qcut(
            pred_df.score,
            np.linspace(0, 1, split + 1),
            retbins=True,
            duplicates='drop')
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
        GoodPercent=lambda x: x.GoodCnt / x.GoodCnt.sum(),
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

    def distr_preprocess(distr_arr):
        distr_arr = np.asarray(distr_arr)
        distr_arr = np.where(np.isnan(distr_arr), 0, distr_arr)
        distr_arr = distr_arr / np.sum(distr_arr)

        # 填充0值
        if np.any(distr_arr == 0):
            distr_arr = distr_arr + epsilon
            distr_arr = distr_arr / base_distr.sum()

        return distr_arr

    base_distr = distr_preprocess(base_distr)
    cmp_distr = distr_preprocess(cmp_distr)

    # calculate psi
    psi_value = (base_distr - cmp_distr) * np.log(base_distr / cmp_distr)
    return psi_value.sum()


def swap_analysis(df,
                  benchmark_model,
                  challenger_model,
                  target_col,
                  benchmark_breaks=None,
                  challenger_breaks=None,
                  segments=10,
                  right=False,
                  retval='badrate'):
    """
    双模型swap分析
    Args:
        df: 包含模型分及Y标的数据框
        benchmark_model: 基准模型
        challenger_model: 挑战者模型
        target_col: Y标列名
        benchmark_breaks: 基准模型切分点，默认为空
        challenger_breaks: 挑战者模型切分点，默认为空
        segments: 等分段数，默认为10段
        right: right=True时为左开右闭区间，否则为右闭左开区间
        retval: 返回统计量，
            'badrate'返回坏率，
            'badcnt'返回坏样本数，
            'totalcnt'返回总样本数，
            'all'返回'totalcnt', 'badcnt', 'badrate'构成的元组

    Returns:
        见 retval 参数说明

    """
    assert retval in ['badrate', 'badcnt', 'totalcnt', 'all'], ValueError(
        'argument retval should be one of \'badrate\', \'badcnt\', '
        f'\'totalcnt\', and \'all\', get {retval}')
    df = df[df[target_col].isin([0, 1]) & (~df[benchmark_model].isna()) &
            (~df[challenger_model].isna())].copy()

    if benchmark_breaks is None:
        df[benchmark_model + '_seg'] = pd.qcut(
            df[benchmark_model], np.linspace(0, 1, segments + 1), duplicates='drop')
    else:
        df[benchmark_model + '_seg'] = pd.cut(
            df[benchmark_model], benchmark_breaks, right=right)

    if challenger_breaks is None:
        df[challenger_model + '_seg'] = pd.qcut(
            df[challenger_model],
            np.linspace(0, 1, segments + 1),
            duplicates='drop')
    else:
        df[challenger_model + '_seg'] = pd.cut(
            df[challenger_model], challenger_breaks, right=right)
    total_cnt = pd.pivot_table(
        df,
        index=benchmark_model + '_seg',
        columns=challenger_model + '_seg',
        values=target_col,
        aggfunc='count')
    if retval == 'totalcnt':
        return total_cnt

    bad_cnt = pd.pivot_table(
        df,
        index=benchmark_model + '_seg',
        columns=challenger_model + '_seg',
        values=target_col,
        aggfunc='sum')
    if retval == 'badcnt':
        return bad_cnt

    bad_rate = bad_cnt / total_cnt
    if retval == 'badrate':
        return bad_rate

    return total_cnt, bad_cnt, bad_rate


def swap_analysis_simple(df,
                         benchmark_model,
                         challenger_model,
                         target_col,
                         reject_ratio=0.2):
    """
    简易双模型swap分析，该函数将两个模型等量的尾部客户标记为拒绝样本，尾部客户占比由`reject_ratio`参数指定，并
    计算换入换出的样本坏率。**注意：该函数要求模型分越高，风险越低**
    Args:
        df: 包含模型分及Y标的数据框
        benchmark_model: 基准模型
        challenger_model: 挑战者模型
        target_col: Y标列名
        reject_ratio: 待拒绝尾部客户占比

    Returns:
        `pd.DataFrame`
    """
    df = df[df[target_col].isin([0, 1]) & (~df[benchmark_model].isna()) &
            (~df[challenger_model].isna())].copy()
    df[benchmark_model + '_seg2'] = pd.qcut(
        df[benchmark_model], [0, reject_ratio, 1.], duplicates='drop')
    df[challenger_model + '_seg2'] = pd.qcut(
        df[challenger_model], [0, reject_ratio, 1.], duplicates='drop')
    total_cnt2 = pd.pivot_table(
        df,
        index=benchmark_model + '_seg2',
        columns=challenger_model + '_seg2',
        values=target_col,
        aggfunc='count')
    bad_cnt2 = pd.pivot_table(
        df,
        index=benchmark_model + '_seg2',
        columns=challenger_model + '_seg2',
        values=target_col,
        aggfunc='sum')
    bad_rate2 = bad_cnt2 / total_cnt2
    swap_result = pd.concat([total_cnt2 / np.sum(total_cnt2.values), bad_rate2],
                            axis=1)
    swap_result.columns = [['total%', 'total%', 'bad_rate', 'bad_rate'],
                           swap_result.columns.tolist()]
    swap_result.index.name = 'base'
    swap_result.columns.name = 'cmp'

    return swap_result

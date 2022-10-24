# -*- encoding: utf-8 -*-

import pandas as pd

from riskmodels.scorecard import woebin
from riskmodels.scorecard import woebin_breaks


def risk_trends_consistency(df, sc_bins, target):
    """
    判断变量在不同数据集上风险趋势与分箱结果中风险趋势是否一致

    Args:
        df: pd.DataFrame
        sc_bins: riskmodels.scorecard.woebin 返回的结果
        target: 目标变量的列名

    Returns:
        {变量名: 分箱badprob序列的 Spearman 秩相关系数}，秩相关系数有效区间 [-1,1]；
        其中 -1 代表趋势完全相反，1 代表趋势完全一致。

    """
    bin_var = list(sc_bins.keys())
    df_var = df.columns.tolist()
    variables = list(set(bin_var).intersection(df_var))

    brk, spc_val = woebin_breaks(sc_bins)
    df_bins = woebin(df,
                     x=variables,
                     y=target,
                     breaks_list=brk,
                     special_values=spc_val)

    consistency = {}
    for v in variables:
        old_bin = sc_bins[v]
        old_bin = old_bin[~old_bin['is_special_values']]
        new_bin = df_bins[v]
        bin_merge = pd.merge(old_bin[['variable', 'bin', 'badprob']],
                             new_bin[['variable', 'bin', 'badprob']],
                             on=['variable', 'bin'],
                             suffixes=['_o', '_n'],
                             how='left')
        c = bin_merge[['badprob_n',
                       'badprob_o']].corr(method='spearman').iloc[0, 0]
        consistency[v] = c

    return consistency

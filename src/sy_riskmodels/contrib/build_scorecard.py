# -*- encoding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.io.excel import ExcelWriter
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sy_riskmodels.contrib.var_select import risk_trends_consistency
from sy_riskmodels.evaluate import (model_eval, gains_table, psi,
                                    swap_analysis_simple)
from sy_riskmodels.models import group_split_cv, stepwise_lr
from sy_riskmodels.scorecard import (make_scorecard, sc_bins_to_df, woebin,
                                     woebin_plot, woebin_ply, woebin_psi)
from sy_riskmodels.utils import str_to_list, sample_stats


def build_scorecard(sample_df,
                    features,
                    target,
                    train_filter,
                    oot_filter,
                    output_excel_file,
                    variable_iv_limit=0.02,
                    sample_stat_group_fields=None,
                    alternative_target=None,
                    regression_p_limit=0.05,
                    random_test_set=0.2,
                    special_values=None,
                    binning_methods=None,
                    binning_kwargs=None,
                    cv=None,
                    base_points=600,
                    base_odds=50,
                    pdo=20,
                    compare_model_fields=None,
                    random_state=0):
    """

    Args:
        sample_df:
        features:
        target:
        train_filter:
        oot_filter:
        output_excel_file:
        variable_iv_limit:
        sample_stat_group_fields:
        alternative_target:
        regression_p_limit:
        random_test_set:
        special_values:
        binning_methods:
        binning_kwargs:
        cv:
        base_points:
        base_odds:
        pdo:
        compare_model_fields:
        random_state:

    Returns:

    """
    if binning_methods is None:
        binning_methods = ['quantile', 'tree']

    if cv is None:
        cv = 3

    # check features
    features = list(set(features).intersection(sample_df.columns.tolist()))

    # check target
    assert target in sample_df.columns.tolist(), f'数据集中不存在目标字段{target}'

    # open excel file
    excel_file = ExcelWriter(output_excel_file)

    # 1. 样本统计
    if sample_stat_group_fields is not None:
        sample_stat_group_fields = str_to_list(sample_stat_group_fields)
        sample_stat_group_fields = list(
            set(sample_stat_group_fields).intersection(
                sample_df.columns.tolist()))

    row_cnt = 0
    if sample_stat_group_fields:
        for field in sample_stat_group_fields:
            tmp_df = sample_df.groupby(field).apply(sample_stats, target=target)
            tmp_df.to_excel(excel_file, sheet_name=f'样本统计', startrow=row_cnt)
            row_cnt += len(tmp_df) + 2

    # 2. 样本拆分
    sample_df['_train_test_flag_'] = np.where(
        train_filter(sample_df), '01_train',
        np.where(oot_filter(sample_df), '03_oot', ''))

    rng = np.random.default_rng(random_state)

    if random_test_set:

        sample_df['_rand_'] = rng.random(len(sample_df))
        sample_df['_train_test_flag_'] = np.where(
            (sample_df['_train_test_flag_'] == '01_train') &
            (sample_df['_rand_'] < random_test_set), '02_test',
            sample_df['_train_test_flag_'])

    tmp_df = sample_df.groupby('_train_test_flag_').apply(sample_stats,
                                                          target=target)
    tmp_df.to_excel(excel_file, sheet_name='样本统计', startrow=row_cnt)

    train_df = sample_df[(sample_df['_train_test_flag_'] == '01_train') &
                         (sample_df[target].isin([0, 1]))].copy().reset_index(
                             drop=True)
    oot_df = sample_df[(sample_df['_train_test_flag_'] == '03_oot') &
                       (sample_df[target].isin([0, 1]))].copy().reset_index(
                           drop=True)

    # 3. 变量分箱
    if binning_kwargs is None:
        binning_kwargs = {}
    bins = woebin(train_df,
                  x=features,
                  y=target,
                  special_values=special_values,
                  methods=binning_methods,
                  **binning_kwargs)
    woe_df, iv_df = sc_bins_to_df(bins)
    woe_df.to_excel(excel_file, sheet_name='WOE分析', index_label='index')
    iv_df.to_excel(excel_file, sheet_name='IV分析', index_label='变量')

    # 变量筛选
    selected_variables = iv_df[
        (iv_df['IV'] > variable_iv_limit) &
        iv_df['单调性'].isin(['increasing', 'decreasing'])].index.tolist()

    var_risk_consist = risk_trends_consistency(
        oot_df, sc_bins={v: bins[v] for v in selected_variables}, target=target)
    selected_variables = [k for k, v in var_risk_consist.items() if v == 1.0]

    # 4.逐步回归
    train_X = woebin_ply(train_df[selected_variables], bins, value='woe')
    train_y = train_df[target]

    _, selected_variables = stepwise_lr(
        train_X,
        train_y.values,
        cv=(lambda: group_split_cv(train_df[cv]))
        if isinstance(cv, str) else cv,
        x=[f + '_woe' for f in selected_variables],
        max_num_features=30)

    print(f'selected variables: {",".join(selected_variables)}')

    # 5. fine tuning
    while True:
        X = train_X[selected_variables].copy()
        X = sm.add_constant(X)
        lr_model = sm.GLM(endog=train_y, exog=X, family=sm.families.Binomial())
        lr_model_result = lr_model.fit()

        if np.any(lr_model_result.params > 0) or np.any(
                lr_model_result.pvalues > regression_p_limit):
            t_values = lr_model_result.tvalues
            rm_var = t_values.index[t_values == t_values.max()].item()
            selected_variables.remove(rm_var)
        else:
            break

    print(selected_variables)
    print(lr_model_result.summary())
    model_summary = pd.DataFrame(
        {'': lr_model_result.summary().as_text().split('\n')})
    model_summary.to_excel(excel_file,
                           sheet_name='模型表达',
                           index=False,
                           header=False)

    # 计算VIF
    vif = {}
    for idx, feature in enumerate(X.columns[1:].tolist()):
        vif[feature] = variance_inflation_factor(train_X, idx)
    vif = pd.Series(vif).rename('VIF').to_frame()
    vif = vif.loc[selected_variables,]
    print(vif)
    vif.to_excel(excel_file,
                 sheet_name='模型表达',
                 index=False,
                 startcol=11,
                 startrow=14)

    scorecard = make_scorecard(bins,
                               lr_model_result.params.to_dict(),
                               base_points=base_points,
                               base_odds=base_odds,
                               pdo=pdo)
    print(scorecard)
    scorecard.to_excel(excel_file,
                       sheet_name='模型表达',
                       index=False,
                       startrow=len(model_summary) + 3)

    # 模型评估
    all_sample_X = woebin_ply(
        sample_df[[var[:-4] for var in selected_variables]], bins)
    # 这一步非常重要！！
    all_sample_X = all_sample_X[selected_variables]
    all_sample_X = sm.add_constant(all_sample_X)

    sample_df['prob'] = lr_model_result.predict(all_sample_X)
    perf = sample_df.groupby('_train_test_flag_').apply(model_eval,
                                                        target=target,
                                                        pred='prob')
    print(perf)
    perf.to_excel(excel_file, sheet_name='模型评估')

    row_cnt = len(perf) + 2
    if sample_stat_group_fields is not None:
        for field in sample_stat_group_fields:
            perf = sample_df.groupby(field).apply(model_eval,
                                                  target=target,
                                                  pred='prob')
            print(perf)
            perf.to_excel(excel_file, sheet_name='模型评估', startrow=row_cnt)
            row_cnt += len(perf) + 2

    A = pdo / np.log(2)
    B = base_points - A * np.log(base_odds)

    sample_df['score'] = np.round(A * np.log(
        (1 - sample_df['prob']) / sample_df['prob']) + B)
    train_df = sample_df[(sample_df['_train_test_flag_'] == '01_train') &
                         sample_df[target].isin([0, 1])]
    _, breaks = gains_table(train_df[target],
                            train_df['score'],
                            return_breaks=True)
    breaks[0] = -np.inf
    breaks[-1] = np.inf
    train_gains_table = gains_table(train_df[target],
                                    train_df['score'],
                                    breaks=breaks)
    pd.DataFrame(['Train Gains Table']).to_excel(excel_file,
                                                 sheet_name='模型评估',
                                                 index=False,
                                                 header=False,
                                                 startrow=row_cnt)
    train_gains_table.to_excel(excel_file,
                               sheet_name='模型评估',
                               startrow=row_cnt + 1)
    row_cnt += len(train_gains_table) + 3

    if random_test_set:
        test_df = sample_df[(sample_df['_train_test_flag_'] == '02_test') &
                            sample_df[target].isin([0, 1])]
        test_gains_table = gains_table(test_df[target],
                                       test_df['score'],
                                       breaks=breaks)
        pd.DataFrame(['Test Gains Table']).to_excel(excel_file,
                                                    sheet_name='模型评估',
                                                    index=False,
                                                    header=False,
                                                    startrow=row_cnt)
        test_gains_table.to_excel(excel_file,
                                  sheet_name='模型评估',
                                  startrow=row_cnt + 1)
        row_cnt += len(train_gains_table) + 3

    oot_df = sample_df[(sample_df['_train_test_flag_'] == '03_oot') &
                       sample_df[target].isin([0, 1])]
    oot_gains_table = gains_table(oot_df[target],
                                  oot_df['score'],
                                  breaks=breaks)
    pd.DataFrame(['OOT Gains Table']).to_excel(excel_file,
                                               sheet_name='模型评估',
                                               index=False,
                                               header=False,
                                               startrow=row_cnt)
    oot_gains_table.to_excel(excel_file,
                             sheet_name='模型评估',
                             startrow=row_cnt + 1)
    row_cnt += len(oot_gains_table) + 3

    # 其他标签
    if alternative_target is not None:
        alternative_target = str_to_list(alternative_target)

        for y in alternative_target:
            perf = sample_df.groupby('_train_test_flag_').apply(model_eval,
                                                                target=y,
                                                                pred='prob')
            pd.DataFrame([y]).to_excel(excel_file,
                                       sheet_name='模型评估',
                                       index=False,
                                       header=False,
                                       startrow=row_cnt)
            perf.to_excel(excel_file, sheet_name='模型评估', startrow=row_cnt + 1)
            row_cnt += len(perf) + 3

    # PSI
    var_psi = woebin_psi(
        train_df,
        oot_df,
        bins={
            k: v for k, v in bins.items() if k + '_woe' in selected_variables
        })
    model_psi = pd.DataFrame({
        'variable': 'model_score',
        'bin': train_gains_table.index,
        'base_distr': train_gains_table['TotalPercent'],
        'cmp_distr': oot_gains_table['TotalPercent']
    }).assign(psi=lambda x: psi(x['base_distr'], x['cmp_distr']))

    psi_df = pd.concat([var_psi, model_psi], ignore_index=True)
    print(psi_df)
    psi_df.to_excel(excel_file, sheet_name='PSI分析', index=False)

    # Swap分析（如有）
    compare_model_fields = str_to_list(compare_model_fields)
    row_cnt = 0
    if compare_model_fields:
        for model in compare_model_fields:
            print(f'新模型 vs {model.upper()}\n')
            swap_tbl = swap_analysis_simple(sample_df,
                                            benchmark_model='score',
                                            challenger_model=model,
                                            target_col=target,
                                            reject_ratio=0.2)
            swap_tbl.to_excel(excel_file, sheet_name='Swap分析', startrow=row_cnt)
            row_cnt += len(swap_tbl) + 5
            print(swap_tbl, '\n\n')

    excel_file.close()

    woe_plots = woebin_plot({k[:-4]: bins[k[:-4]] for k in selected_variables})
    if not os.path.exists('pic'):
        os.mkdir('pic')

    if not os.path.isdir('pic'):
        raise NotADirectoryError('当前目录下pic路径不是目录，无法保存WOE图')
    else:
        for v, plt in woe_plots.items():
            plt.savefig(f'pic/{v}.png')

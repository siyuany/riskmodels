# -*- encoding: utf-8 -*-
"""
syriskmodels 评分卡开发 — creditcard 端到端示例

使用方法: python creditcard_scorecard.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pandas import ExcelWriter

from syriskmodels.datasets import load_creditcard
from syriskmodels.utils import sample_stats
from syriskmodels.scorecard import (woebin, sc_bins_to_df, woebin_ply,
                                     woebin_plot, woebin_psi, make_scorecard)
from syriskmodels.models import stepwise_lr
from syriskmodels.evaluate import model_eval, gains_table, psi
from syriskmodels.contrib.var_select import risk_trends_consistency


def main():
    target = 'Class'
    
    # ============================================================
    # Phase 1: 数据准备
    # ============================================================
    print('=== Phase 1: 数据准备 ===')
    df_full = load_creditcard()
    
    df_fraud = df_full[df_full[target] == 1].copy()
    df_normal = df_full[df_full[target] == 0].sample(frac=0.2, random_state=42).copy()
    
    train_fraud_idx = df_fraud.sample(frac=0.7, random_state=42).index
    train_normal_idx = df_normal.sample(frac=0.7, random_state=42).index
    train_idx = train_fraud_idx.union(train_normal_idx)
    
    df_full['dataset'] = np.where(df_full.index.isin(train_idx), 'train', 'oot')
    
    df = pd.concat([
        df_full[df_full[target] == 1],
        df_full[(df_full[target] == 0) & (df_full.index.isin(df_normal.index))]
    ], ignore_index=True)
    
    print(f'采样后数据: {len(df)} 行 (保留全部 {(df[target]==1).sum()} 个欺诈样本)')
    
    print('样本统计:')
    print(sample_stats(df, target=target))
    
    features = [c for c in df.columns if c not in [target, 'Time', 'dataset']]
    print(f'\n特征数量: {len(features)} 个 (排除 Time 和 Class)')
    
    train_df = df[df['dataset'] == 'train'].reset_index(drop=True)
    oot_df = df[df['dataset'] == 'oot'].reset_index(drop=True)
    
    print(f'训练集: {len(train_df)} 行 (欺诈: {(train_df[target]==1).sum()}), OOT: {len(oot_df)} 行 (欺诈: {(oot_df[target]==1).sum()})')
    
    # ============================================================
    # Phase 2: WOE 分箱
    # ============================================================
    print('\n=== Phase 2: WOE 分箱 ===')
    bins = woebin(train_df, y=target, x=features,
                  methods=['quantile', 'tree'],
                  count_distr_limit=0.05, bin_num_limit=5)
    
    woe_df, iv_df = sc_bins_to_df(bins)
    print('\nIV 排名 (Top 10):')
    print(iv_df.head(10))
    
    # ============================================================
    # Phase 3: 特征筛选
    # ============================================================
    print('\n=== Phase 3: 特征筛选 ===')
    selected = iv_df[
        (iv_df['IV'] > 0.02) &
        iv_df['单调性'].isin(['increasing', 'decreasing'])
    ].index.tolist()
    print(f'IV + 单调性筛选后: {len(selected)} 个变量')
    
    consist = risk_trends_consistency(
        oot_df, sc_bins={v: bins[v] for v in selected}, target=target)
    selected = [k for k, v in consist.items() if v == 1.0]
    print(f'风险趋势一致性筛选后: {len(selected)} 个变量')
    
    # ============================================================
    # Phase 4: 逐步回归
    # ============================================================
    print('\n=== Phase 4: 逐步回归 ===')
    train_X = woebin_ply(train_df[selected], bins, value='woe')
    train_X[target] = train_df[target].values
    
    _, selected_woe = stepwise_lr(
        train_X, y=target,
        x=[f + '_woe' for f in selected],
        cv=3, max_num_features=30)
    print(f'逐步回归入选: {len(selected_woe)} 个变量')
    print(selected_woe)
    
    # ============================================================
    # Phase 5: 模型精调
    # ============================================================
    print('\n=== Phase 5: 模型精调 ===')
    iteration = 0
    while True:
        X = train_X[selected_woe].copy()
        X = sm.add_constant(X)
        model = sm.GLM(endog=train_df[target].values, exog=X,
                       family=sm.families.Binomial()).fit()
        
        coef_params = model.params.iloc[1:]
        coef_pvalues = model.pvalues.iloc[1:]
        if np.any(coef_params > 0) or np.any(coef_pvalues > 0.05):
            t_vals = model.tvalues.iloc[1:]
            rm_var = t_vals.index[t_vals == t_vals.max()].item()
            selected_woe.remove(rm_var)
            iteration += 1
            print(f'迭代 {iteration}: 移除 {rm_var}')
        else:
            break
    
    print('\n模型摘要:')
    print(model.summary())
    
    print('\nVIF 检查:')
    X_arr = X.to_numpy()
    for i, feat in enumerate(selected_woe):
        vif_val = variance_inflation_factor(X_arr, i + 1)
        print(f'{feat}: VIF={vif_val:.2f}')
    
    # ============================================================
    # Phase 6: 评分卡生成
    # ============================================================
    print('\n=== Phase 6: 评分卡生成 ===')
    base_points, base_odds, pdo = 600, 50, 20
    scorecard = make_scorecard(bins, model.params.to_dict(),
                               base_points=base_points, base_odds=base_odds, pdo=pdo)
    print(scorecard)
    
    # ============================================================
    # Phase 7: 模型评估
    # ============================================================
    print('\n=== Phase 7: 模型评估 ===')
    selected_raw = [v[:-4] for v in selected_woe]
    
    all_X = woebin_ply(df[selected_raw], bins)[selected_woe]
    all_X = sm.add_constant(all_X)
    df['prob'] = model.predict(all_X)
    
    A = pdo / np.log(2)
    B = base_points - A * np.log(base_odds)
    df['score'] = np.round(A * np.log((1 - df['prob']) / df['prob']) + B)
    
    perf = df.groupby('dataset').apply(model_eval, target=target, pred='prob', include_groups=False)
    print('\nAUC / KS:')
    print(perf)
    
    train_sub = df[df['dataset'] == 'train']
    gt_train, breaks = gains_table(train_sub[target], train_sub['score'],
                                    return_breaks=True)
    breaks[0] = -np.inf
    breaks[-1] = np.inf
    print('\nTrain Gains Table:')
    print(gt_train)
    
    oot_sub = df[df['dataset'] == 'oot']
    gt_oot = gains_table(oot_sub[target], oot_sub['score'], breaks=breaks)
    print('\nOOT Gains Table:')
    print(gt_oot)
    
    # ============================================================
    # Phase 8: PSI 分析
    # ============================================================
    print('\n=== Phase 8: PSI 分析 ===')
    var_psi = woebin_psi(
        train_sub, oot_sub,
        bins={k: v for k, v in bins.items() if k + '_woe' in selected_woe})
    
    model_psi = pd.DataFrame({
        'variable': 'model_score',
        'bin': gt_train.index,
        'base_distr': gt_train['TotalPercent'].values,
        'cmp_distr': gt_oot['TotalPercent'].values
    }).assign(psi=lambda x: psi(x['base_distr'], x['cmp_distr']))
    
    psi_df = pd.concat([var_psi, model_psi], ignore_index=True)
    print(psi_df)
    
    # ============================================================
    # Phase 9: 输出 Excel 报告
    # ============================================================
    print('\n=== Phase 9: 输出 Excel 报告 ===')
    output_file = 'creditcard_scorecard_report.xlsx'
    with ExcelWriter(output_file) as writer:
        sample_stat = df.groupby('dataset').apply(sample_stats, target=target, include_groups=False)
        sample_stat.to_excel(writer, sheet_name='样本统计')
        
        woe_df.to_excel(writer, sheet_name='WOE分析', index_label='index')
        iv_df.to_excel(writer, sheet_name='IV分析', index_label='变量')
        
        summary_df = pd.DataFrame({'': model.summary().as_text().split('\n')})
        summary_df.to_excel(writer, sheet_name='模型表达', index=False, header=False)
        scorecard.to_excel(writer, sheet_name='模型表达', index=False,
                           startrow=len(summary_df) + 3)
        
        perf.to_excel(writer, sheet_name='模型评估')
        row = len(perf) + 2
        gt_train.to_excel(writer, sheet_name='模型评估', startrow=row)
        row += len(gt_train) + 2
        gt_oot.to_excel(writer, sheet_name='模型评估', startrow=row)
        
        psi_df.to_excel(writer, sheet_name='PSI分析', index=False)
    
    print(f'\n报告已保存: {output_file}')


if __name__ == '__main__':
    main()
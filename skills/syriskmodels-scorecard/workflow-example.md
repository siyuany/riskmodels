# 端到端示例：germancredit 评分卡开发

本示例基于德国信用数据集，演示使用 syriskmodels 开发信用评分卡的完整流程。
代码可直接复制执行。

## 完整代码

```python
# -*- encoding: utf-8 -*-
"""
syriskmodels 评分卡开发 — germancredit 端到端示例
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from syriskmodels.datasets import load_germancredit
from syriskmodels.utils import sample_stats
from syriskmodels.scorecard import (woebin, sc_bins_to_df, woebin_ply,
                                     woebin_plot, woebin_psi, make_scorecard)
from syriskmodels.models import stepwise_lr
from syriskmodels.evaluate import model_eval, gains_table, psi
from syriskmodels.contrib.var_select import risk_trends_consistency

# ============================================================
# Phase 1: 数据准备
# ============================================================
df = load_germancredit()  # creditability 已自动映射: good→0, bad→1
target = 'creditability'

print('=== 样本统计 ===')
print(sample_stats(df, target=target))

# 确定特征列（排除目标变量）
features = [c for c in df.columns if c != target]

# 划分训练集 / OOT
train_df = df.sample(frac=0.7, random_state=42).reset_index(drop=True)
oot_df = df.drop(train_df.index).reset_index(drop=True)

print(f'\n训练集: {len(train_df)} 行, OOT: {len(oot_df)} 行')

# ============================================================
# Phase 2: WOE 分箱
# ============================================================
bins = woebin(train_df, y=target, x=features,
              methods=['quantile', 'tree'],
              count_distr_limit=0.05, bin_num_limit=5)

woe_df, iv_df = sc_bins_to_df(bins)
print('\n=== IV 排名 (Top 10) ===')
print(iv_df.head(10))

# ============================================================
# Phase 3: 特征筛选
# ============================================================
selected = iv_df[
    (iv_df['IV'] > 0.02) &
    iv_df['单调性'].isin(['increasing', 'decreasing'])
].index.tolist()
print(f'\nIV + 单调性筛选后: {len(selected)} 个变量')

consist = risk_trends_consistency(
    oot_df, sc_bins={v: bins[v] for v in selected}, target=target)
selected = [k for k, v in consist.items() if v == 1.0]
print(f'风险趋势一致性筛选后: {len(selected)} 个变量')

# ============================================================
# Phase 4: 逐步回归
# ============================================================
train_X = woebin_ply(train_df[selected], bins, value='woe')
train_X[target] = train_df[target]

_, selected_woe = stepwise_lr(
    train_X, y=target,
    x=[f + '_woe' for f in selected],
    cv=3, max_num_features=30)
print(f'\n逐步回归入选: {len(selected_woe)} 个变量')
print(selected_woe)

# ============================================================
# Phase 5: 模型精调
# ============================================================
while True:
    X = train_X[selected_woe].copy()
    X = sm.add_constant(X)
    model = sm.GLM(endog=train_df[target], exog=X,
                   family=sm.families.Binomial()).fit()

    coef_params = model.params.iloc[1:]
    coef_pvalues = model.pvalues.iloc[1:]
    if np.any(coef_params > 0) or np.any(coef_pvalues > 0.05):
        t_vals = model.tvalues.iloc[1:]
        rm_var = t_vals.index[t_vals == t_vals.max()].item()
        selected_woe.remove(rm_var)
        print(f'移除: {rm_var}')
    else:
        break

print('\n=== 模型摘要 ===')
print(model.summary())

# VIF
print('\n=== VIF ===')
X_arr = X.to_numpy()
for i, feat in enumerate(selected_woe):
    print(f'{feat}: VIF={variance_inflation_factor(X_arr, i + 1):.2f}')

# ============================================================
# Phase 6: 评分卡生成
# ============================================================
base_points, base_odds, pdo = 600, 50, 20
scorecard = make_scorecard(bins, model.params.to_dict(),
                           base_points=base_points, base_odds=base_odds, pdo=pdo)
print('\n=== 评分卡 ===')
print(scorecard)

# ============================================================
# Phase 7: 模型评估
# ============================================================
selected_raw = [v[:-4] for v in selected_woe]
all_X = woebin_ply(df[selected_raw], bins)[selected_woe]
all_X = sm.add_constant(all_X)
df['prob'] = model.predict(all_X)

A = pdo / np.log(2)
B = base_points - A * np.log(base_odds)
df['score'] = np.round(A * np.log((1 - df['prob']) / df['prob']) + B)

# 标记数据集
df['dataset'] = 'oot'
df.loc[train_df.index, 'dataset'] = 'train'

perf = df.groupby('dataset').apply(model_eval, target=target, pred='prob')
print('\n=== AUC / KS ===')
print(perf)

train_sub = df[df['dataset'] == 'train']
gt_train, breaks = gains_table(train_sub[target], train_sub['score'],
                                return_breaks=True)
breaks[0] = -np.inf
breaks[-1] = np.inf
print('\n=== Train Gains Table ===')
print(gt_train)

oot_sub = df[df['dataset'] == 'oot']
gt_oot = gains_table(oot_sub[target], oot_sub['score'], breaks=breaks)
print('\n=== OOT Gains Table ===')
print(gt_oot)

# ============================================================
# Phase 8: PSI 分析
# ============================================================
var_psi = woebin_psi(
    train_sub, oot_sub,
    bins={k: v for k, v in bins.items() if k + '_woe' in selected_woe})

model_psi = pd.DataFrame({
    'variable': 'model_score',
    'bin': gt_train.index,
    'base_distr': gt_train['TotalPercent'],
    'cmp_distr': gt_oot['TotalPercent']
}).assign(psi=lambda x: psi(x['base_distr'], x['cmp_distr']))

psi_df = pd.concat([var_psi, model_psi], ignore_index=True)
print('\n=== PSI ===')
print(psi_df)

# ============================================================
# Phase 9: 输出 Excel 报告
# ============================================================
from pandas import ExcelWriter

with ExcelWriter('germancredit_scorecard_report.xlsx') as writer:
    # 样本统计
    sample_stat = df.groupby('dataset').apply(sample_stats, target=target)
    sample_stat.to_excel(writer, sheet_name='样本统计')

    # WOE / IV
    woe_df.to_excel(writer, sheet_name='WOE分析', index_label='index')
    iv_df.to_excel(writer, sheet_name='IV分析', index_label='变量')

    # 模型表达
    summary_df = pd.DataFrame({'': model.summary().as_text().split('\n')})
    summary_df.to_excel(writer, sheet_name='模型表达', index=False, header=False)
    scorecard.to_excel(writer, sheet_name='模型表达', index=False,
                       startrow=len(summary_df) + 3)

    # 模型评估
    perf.to_excel(writer, sheet_name='模型评估')
    row = len(perf) + 2
    gt_train.to_excel(writer, sheet_name='模型评估', startrow=row)
    row += len(gt_train) + 2
    gt_oot.to_excel(writer, sheet_name='模型评估', startrow=row)

    # PSI
    psi_df.to_excel(writer, sheet_name='PSI分析', index=False)

print('\n报告已保存: germancredit_scorecard_report.xlsx')
```
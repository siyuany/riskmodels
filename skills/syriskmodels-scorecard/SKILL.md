---
name: syriskmodels-scorecard
description: Use when building credit risk scorecards with syriskmodels Python library, performing WOE binning, feature selection, logistic regression modeling, or scorecard validation
---

# syriskmodels 信用评分卡开发

## Overview

本 Skill 指导你使用 `syriskmodels` Python 库完成信用评分卡的全流程开发，从数据准备到模型文档输出。

`syriskmodels` 是一个信用风险建模工具库，提供 WOE 分箱、逐步回归、评分卡转换、模型评估等功能。

## When to Use

- 用户要求开发/构建信用评分卡 (credit scorecard)
- 用户要求进行 WOE 分箱或 IV 值分析
- 用户要求使用逻辑回归进行信用风险建模
- 用户要求对评分卡模型进行 KS、AUC、Gains Table、PSI 评估
- 用户提到使用 `syriskmodels`、`riskmodels` 或本仓库

## Quick Reference

| 任务 | 关键 API | 一行示例 |
|------|---------|---------|
| 加载演示数据 | `syriskmodels.datasets.load_germancredit()` | `df = load_germancredit()` |
| 样本统计 | `syriskmodels.utils.sample_stats()` | `df.groupby('month').apply(sample_stats, target='y')` |
| WOE 分箱 | `syriskmodels.scorecard.woebin()` | `bins = woebin(df, y='target', x=features)` |
| 整合分箱结果 | `syriskmodels.scorecard.sc_bins_to_df()` | `woe_df, iv_df = sc_bins_to_df(bins)` |
| WOE 转换 | `syriskmodels.scorecard.woebin_ply()` | `df_woe = woebin_ply(df, bins, value='woe')` |
| 逐步回归 | `syriskmodels.models.stepwise_lr()` | `_, selected = stepwise_lr(df, y='target', x=woe_vars)` |
| 评分卡生成 | `syriskmodels.scorecard.make_scorecard()` | `sc = make_scorecard(bins, model.params.to_dict())` |
| 模型评估 | `syriskmodels.evaluate.model_eval()` | `perf = df.groupby('flag').apply(model_eval, target='y', pred='prob')` |
| Gains Table | `syriskmodels.evaluate.gains_table()` | `gt, brk = gains_table(y, score, return_breaks=True)` |
| 变量 PSI | `syriskmodels.scorecard.woebin_psi()` | `psi_df = woebin_psi(train_df, oot_df, bins)` |
| 分布 PSI | `syriskmodels.evaluate.psi()` | `psi_val = psi(base_distr, cmp_distr)` |
| 风险趋势一致性 | `syriskmodels.contrib.var_select.risk_trends_consistency()` | `consist = risk_trends_consistency(oot_df, sc_bins=bins, target='y')` |
| Bivar 图 | `syriskmodels.scorecard.woebin_plot()` | `plots = woebin_plot(bins)` |

## 全流程指引

### Phase 1: 数据准备

**目的：** 加载数据、确认目标变量、识别特征列、划分训练集/OOT 集

**关键 API：**
- `syriskmodels.datasets.load_germancredit()` / `load_creditcard()` — 演示数据
- `syriskmodels.utils.sample_stats()` — 样本统计

**代码模板：**

```python
import pandas as pd
import numpy as np
from syriskmodels.datasets import load_germancredit
from syriskmodels.utils import sample_stats

# 1. 加载数据（演示数据或用户数据）
df = load_germancredit()  # creditability 列已自动映射为 0/1（good→0, bad→1）
target = 'creditability'

# 2. 样本统计
print(sample_stats(df, target=target))

# 3. 确定特征列（排除非特征列）
all_columns = df.columns.tolist()
exclude_cols = [target]  # 加入 ID列、时间列、目标衍生列等
features = [c for c in all_columns if c not in exclude_cols]

# 4. 划分训练集/OOT
train_df = df.sample(frac=0.7, random_state=42)
oot_df = df.drop(train_df.index)
```

**注意事项：**
- 目标变量必须是 0/1 二分类，1 代表坏样本
- **必须识别并排除非特征列**，包括但不限于：
  - ID 列（客户编号、申请编号等）
  - 时间列（申请日期、放款日期等）
  - 目标变量的衍生列（逾期天数、逾期金额等用于计算目标变量的列）
  - 其他业务标记列（样本标签、分组标记等）
- **必须列出识别到的可疑非特征列，与用户确认后再确定最终特征列表**
- 如用户提供了自己的数据，优先使用用户数据

### Phase 2: WOE 分箱

**目的：** 对特征变量进行 WOE 分箱，计算 WOE 和 IV 值

**关键 API：**
- `syriskmodels.scorecard.woebin()` — 核心分箱函数
- `syriskmodels.scorecard.sc_bins_to_df()` — 整合分箱结果
- `syriskmodels.scorecard.woebin_plot()` — 生成 bivar 图

**代码模板：**

```python
from syriskmodels.scorecard import woebin, sc_bins_to_df, woebin_plot

bins = woebin(train_df, y=target, x=features,
              methods=['quantile', 'tree'],
              count_distr_limit=0.05,
              bin_num_limit=5)

woe_df, iv_df = sc_bins_to_df(bins)
```

**注意事项：**
- `methods` 首元素必须是无监督细分箱（`'quantile'` 或 `'hist'`）
- `special_values` 处理特殊值（如 -999、-1 等缺失标记）
- `ensure_monotonic=True` 可强制单调性（仅树分箱支持）
- 常见 methods 组合见 `api-reference.md`

### Phase 3: 特征筛选（IV + 风险趋势一致性）

**目的：** 基于 IV 值和 OOT 风险趋势一致性筛选变量

**关键 API：**
- `iv_df` 的 IV 列和单调性列
- `syriskmodels.contrib.var_select.risk_trends_consistency()`

**代码模板：**

```python
from syriskmodels.contrib.var_select import risk_trends_consistency

# IV > 0.02 且单调
selected = iv_df[
    (iv_df['IV'] > 0.02) &
    iv_df['单调性'].isin(['increasing', 'decreasing'])
].index.tolist()

# OOT 风险趋势一致性
consist = risk_trends_consistency(
    oot_df, sc_bins={v: bins[v] for v in selected}, target=target)
selected = [k for k, v in consist.items() if v == 1.0]
```

**注意事项：**
- IV < 0.02 的变量预测能力极弱，通常剔除
- 单调性要求因业务而异，U 型变量可能有业务含义，需与用户确认
- `risk_trends_consistency()` 返回 Spearman 秩相关系数，1.0 为完全一致

### Phase 4: 逐步回归

**目的：** 使用双向逐步回归筛选入模变量

**关键 API：**
- `syriskmodels.scorecard.woebin_ply()` — WOE 转换
- `syriskmodels.models.stepwise_lr()` — 逐步回归

**代码模板：**

```python
from syriskmodels.scorecard import woebin_ply
from syriskmodels.models import stepwise_lr

train_X = woebin_ply(train_df[selected], bins, value='woe')
train_X[target] = train_df[target]  # 必须包含 target 列！

_, selected_woe = stepwise_lr(
    train_X, y=target,
    x=[f + '_woe' for f in selected],
    cv=3, max_num_features=30)
```

**注意事项：**
- `stepwise_lr` 的 `y` 参数是**列名字符串**（非 numpy array）
- 传入的 DataFrame **必须包含 target 列**
- `selected_woe` 返回的是 `xxx_woe` 格式的列名

### Phase 5: 模型精调（系数方向 + P 值 + VIF）

**目的：** 拟合 GLM，剔除系数为正或 P 值不显著的变量，检查 VIF

**关键 API：**
- `statsmodels.api.GLM`
- `statsmodels.stats.outliers_influence.variance_inflation_factor`

**代码模板：**

```python
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

while True:
    X = train_X[selected_woe].copy()
    X = sm.add_constant(X)
    model = sm.GLM(endog=train_df[target], exog=X,
                   family=sm.families.Binomial()).fit()

    # 排除截距项(const)检查系数方向和p值
    coef_params = model.params.iloc[1:]
    coef_pvalues = model.pvalues.iloc[1:]
    if np.any(coef_params > 0) or np.any(coef_pvalues > 0.05):
        t_vals = model.tvalues.iloc[1:]
        rm_var = t_vals.index[t_vals == t_vals.max()].item()
        selected_woe.remove(rm_var)
        print(f'移除变量: {rm_var}')
    else:
        break

print(model.summary())

# VIF 检查
X_arr = X.to_numpy()
for i, feat in enumerate(selected_woe):
    vif_val = variance_inflation_factor(X_arr, i + 1)  # i+1 跳过截距列
    print(f'{feat}: VIF={vif_val:.2f}')
```

**注意事项：**
- 系数方向检查和 P 值检查**必须排除截距项**（`iloc[1:]`）
- VIF 计算时用 `sm.add_constant()` 后的矩阵，索引从 1 开始（跳过截距列）
- VIF > 10 提示严重多重共线性，应考虑剔除
- 信用评分卡中**系数应为负数**（WOE 值越大代表越好，风险越低）

### Phase 6: 评分卡生成

**目的：** 将逻辑回归系数转换为评分卡

**关键 API：** `syriskmodels.scorecard.make_scorecard()`

**代码模板：**

```python
from syriskmodels.scorecard import make_scorecard

scorecard = make_scorecard(bins, model.params.to_dict(),
                           base_points=600, base_odds=50, pdo=20)
print(scorecard)
```

**注意事项：**
- `model.params.to_dict()` 返回 `{'const': ..., 'var_woe': ..., ...}` 格式
- `base_points` / `base_odds` / `pdo` 参数可根据业务需求调整

### Phase 7: 模型评估（AUC / KS / Gains Table）

**目的：** 评估模型区分能力

**关键 API：**
- `syriskmodels.evaluate.model_eval()` — AUC / KS
- `syriskmodels.evaluate.gains_table()` — Gains Table

**代码模板：**

```python
from syriskmodels.evaluate import model_eval, gains_table

# 计算预测概率和评分
selected_raw = [v[:-4] for v in selected_woe]  # 去掉 _woe 后缀
all_X = woebin_ply(df[selected_raw], bins)[selected_woe]
all_X = sm.add_constant(all_X)
df['prob'] = model.predict(all_X)

A = pdo / np.log(2)
B = base_points - A * np.log(base_odds)
df['score'] = np.round(A * np.log((1 - df['prob']) / df['prob']) + B)

# AUC / KS（按数据集分组）
perf = df.groupby('dataset_flag').apply(model_eval, target=target, pred='prob')
print(perf)

# Gains Table
gt_train, breaks = gains_table(train_df[target], train_df['score'],
                                return_breaks=True)
breaks[0] = -np.inf
breaks[-1] = np.inf

gt_oot = gains_table(oot_df[target], oot_df['score'], breaks=breaks)
```

**注意事项：**
- Train 集的 Gains Table 用等分点，OOT 使用 Train 的切分点（breaks）保持一致
- breaks 首尾设为 -inf/inf 确保所有样本被覆盖

### Phase 8: 稳定性分析（PSI）

**目的：** 评估变量和模型的稳定性

**关键 API：**
- `syriskmodels.scorecard.woebin_psi()` — 变量 PSI
- `syriskmodels.evaluate.psi()` — 分布 PSI

**代码模板：**

```python
from syriskmodels.scorecard import woebin_psi
from syriskmodels.evaluate import psi

# 变量 PSI
var_psi = woebin_psi(
    train_df, oot_df,
    bins={k: v for k, v in bins.items() if k + '_woe' in selected_woe})

# 模型 PSI（基于 Gains Table 的分数分布）
model_psi = pd.DataFrame({
    'variable': 'model_score',
    'bin': gt_train.index,
    'base_distr': gt_train['TotalPercent'],
    'cmp_distr': gt_oot['TotalPercent']
}).assign(psi=lambda x: psi(x['base_distr'], x['cmp_distr']))

psi_df = pd.concat([var_psi, model_psi], ignore_index=True)
print(psi_df)
```

**注意事项：**
- PSI < 0.1 表示稳定，0.1-0.25 需关注，> 0.25 表示显著偏移
- 变量 PSI 仅对入模变量计算

### Phase 9: 输出模型文档

**目的：** 将全部分析结果输出为 Excel 文件

**代码模板：**

```python
from pandas import ExcelWriter

with ExcelWriter('scorecard_report.xlsx') as writer:
    # Sheet 1: 样本统计
    sample_stats_df.to_excel(writer, sheet_name='样本统计')

    # Sheet 2: WOE 分析
    woe_df.to_excel(writer, sheet_name='WOE分析', index_label='index')

    # Sheet 3: IV 分析
    iv_df.to_excel(writer, sheet_name='IV分析', index_label='变量')

    # Sheet 4: 模型表达
    model_summary = pd.DataFrame({'': model.summary().as_text().split('\n')})
    model_summary.to_excel(writer, sheet_name='模型表达', index=False, header=False)
    vif_df.to_excel(writer, sheet_name='模型表达', startcol=11, startrow=14)
    scorecard.to_excel(writer, sheet_name='模型表达', index=False,
                       startrow=len(model_summary) + 3)

    # Sheet 5: 模型评估
    perf.to_excel(writer, sheet_name='模型评估')
    # ... Gains Table (Train/Test/OOT)

    # Sheet 6: PSI 分析
    psi_df.to_excel(writer, sheet_name='PSI分析', index=False)
```

**产出 Excel 结构：**

| Sheet | 内容 |
|-------|------|
| 样本统计 | 各维度样本分布 |
| WOE分析 | WOE 表 |
| IV分析 | IV 表 |
| 模型表达 | 回归摘要 + VIF + 评分卡 |
| 模型评估 | AUC/KS + Gains Table（Train/Test/OOT） |
| PSI分析 | 变量 PSI + 模型 PSI |

## Common Mistakes

| 错误 | 正确做法 |
|------|---------|
| `stepwise_lr(df, y=train_y.to_numpy(), ...)` | `y` 必须是列名字符串，且 df 中包含该列 |
| `variance_inflation_factor(train_X, idx)` | 用 `sm.add_constant()` 后的矩阵，索引 `idx + 1` 跳过截距列 |
| 检查系数方向包含截距项 | `model.params.iloc[1:]` 排除截距 |
| methods 列表首元素为监督方法 | 首元素必须是 `'quantile'` 或 `'hist'`（无监督细分箱） |
| 直接用 train 的 breaks 比较 OOT Gains Table | 需将 breaks 首尾设为 `-np.inf` / `np.inf` |
| 将 ID 列/时间列当作特征 | 必须识别并排除非特征列，与用户确认 |

## See Also

- `api-reference.md` — API 完整参考文档
- `workflow-example.md` — 基于 germancredit 数据集的端到端示例
# syriskmodels v0.3.1 — Scorecard Skill 设计文档

## 概述

为 syriskmodels 创建 superpowers skill，使 AI agent 能够引导用户完成信用评分卡的全流程开发。同时对 syriskmodels 库进行适配性增强。

## 决策记录

| 项 | 决定 |
|---|---|
| 方案 | 方案A：单一 Skill + 库增强 |
| 目标用户 | 兼顾新手和有经验的风控建模人员 |
| 覆盖范围 | 全流程（数据准备→分箱→筛选→建模→评分卡→评估） |
| 产出形式 | 生成可执行 Python 代码 + 模型文档（Excel 报告） |
| Skill 位置 | 仓库内 `skills/syriskmodels-scorecard/` |
| 数据集 | 自带演示数据集 + 支持用户自有数据 |

## 文件结构

```
lucky-knight/
├── skills/                                  # 新增
│   └── syriskmodels-scorecard/
│       ├── SKILL.md                         # 主 skill：流程指引 + Quick Reference
│       ├── api-reference.md                 # API 完整参考
│       └── workflow-example.md              # 端到端示例代码
├── src/syriskmodels/
│   ├── __init__.py                          # 版本号 → 0.3.1，新增 datasets 导出
│   ├── datasets.py                          # 新增：数据加载模块
│   └── contrib/build_scorecard.py           # BUG 修复（3处）
├── pyproject.toml                           # 版本号 → 0.3.1
└── ...
```

## 一、库改动

### 1.1 新增 `datasets.py`

提供类似 `sklearn.datasets` 的便捷数据加载接口：

```python
# src/syriskmodels/datasets.py

def load_creditcard() -> pd.DataFrame:
    """加载信用卡数据集"""

def load_germancredit() -> pd.DataFrame:
    """加载德国信用数据集"""

def get_data_dir() -> Path:
    """返回内置数据集目录路径"""
```

数据文件位置不变（`data/` 目录），通过包内相对路径定位。

### 1.2 BUG 修复（`contrib/build_scorecard.py`）

| # | 位置 | 问题 | 修复 |
|---|------|------|------|
| 1 | 第147-149行 | `stepwise_lr` 的 `y` 参数应为列名，传入了 numpy array | 将 target 列拼入 `train_X`，`y` 传列名字符串 |
| 2 | 第184行 | VIF 计算用全量 `train_X`，fine tuning 后列映射错位 | 改为 `variance_inflation_factor(X.to_numpy(), idx + 1)` |
| 3 | 第164-168行 | 系数方向/p值/t值检查包含截距项 | 用 `iloc[1:]` 排除截距项 |

### 1.3 Docstring 增强

对以下核心函数增强 docstring（增加参数说明、返回值示例、典型用法），不改变函数签名或行为：

- `woebin()` — 补充 methods 参数的组合示例
- `woebin_ply()` — 补充 value 参数的输出示例
- `make_scorecard()` — 补充 coef 参数的格式说明
- `stepwise_lr()` — 补充返回值说明
- `model_eval()` / `gains_table()` — 补充返回值 DataFrame 的列说明

### 1.4 版本号变更

- `pyproject.toml`: `version = "0.3.0"` → `"0.3.1"`
- `src/syriskmodels/__init__.py`: `__version__ = '0.3.0'` → `'0.3.1'`
- `__init__.py` 新增 `from .datasets import load_creditcard, load_germancredit` 导出

## 二、Skill 设计

### 2.1 SKILL.md Frontmatter

```yaml
---
name: syriskmodels-scorecard
description: Use when building credit risk scorecards with syriskmodels Python library, performing WOE binning, feature selection, logistic regression modeling, or scorecard validation
---
```

### 2.2 SKILL.md 内容结构

```
# syriskmodels 信用评分卡开发

## Overview
## When to Use
## Quick Reference（任务→关键API→一行示例 表格）
## 全流程指引
  ### Phase 1: 数据准备
  ### Phase 2: WOE 分箱
  ### Phase 3: 特征筛选
  ### Phase 4: 逐步回归
  ### Phase 5: 模型精调
  ### Phase 6: 评分卡生成
  ### Phase 7: 模型评估
  ### Phase 8: 稳定性分析（PSI）
  ### Phase 9: 输出模型文档
## Common Mistakes
## See Also
```

### 2.3 各 Phase 详细设计

每个 Phase 统一结构：**目的 → 关键 API → 代码模板 → 注意事项**

#### Phase 1: 数据准备

**目的：** 加载数据、确认目标变量、识别特征列、划分训练集/OOT集

**关键 API：**
- `syriskmodels.datasets.load_germancredit()` / `load_creditcard()` — 演示数据
- `syriskmodels.utils.sample_stats()` — 样本统计

**代码模板：**
```python
import pandas as pd
import numpy as np
from syriskmodels.datasets import load_germancredit
from syriskmodels.utils import sample_stats

df = load_germancredit()
target = 'creditability'

# 样本统计
sample_stats(df, target=target)

# 划分训练集/OOT
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
- agent 应列出识别到的可疑非特征列，**与用户确认**后再确定最终特征列表

#### Phase 2: WOE 分箱

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
- `methods` 首元素必须是无监督细分箱（`quantile` 或 `hist`）
- `special_values` 处理特殊值（如 -999、-1 等）
- `ensure_monotonic=True` 可强制单调性（仅树分箱支持）

#### Phase 3: 特征筛选（IV + 风险趋势一致性）

**目的：** 基于 IV 值和 OOT 风险趋势一致性筛选变量

**关键 API：**
- `iv_df` 的 IV 列和单调性列
- `syriskmodels.contrib.var_select.risk_trends_consistency()`

**代码模板：**
```python
from syriskmodels.contrib.var_select import risk_trends_consistency

selected = iv_df[
    (iv_df['IV'] > 0.02) &
    iv_df['单调性'].isin(['increasing', 'decreasing'])
].index.tolist()

consist = risk_trends_consistency(
    oot_df, sc_bins={v: bins[v] for v in selected}, target=target)
selected = [k for k, v in consist.items() if v == 1.0]
```

#### Phase 4: 逐步回归

**目的：** 使用双向逐步回归筛选入模变量

**关键 API：**
- `syriskmodels.scorecard.woebin_ply()` — WOE 转换
- `syriskmodels.models.stepwise_lr()` — 逐步回归

**代码模板：**
```python
from syriskmodels.scorecard import woebin_ply
from syriskmodels.models import stepwise_lr

train_X = woebin_ply(train_df[selected], bins, value='woe')
train_X[target] = train_df[target]

_, selected_woe = stepwise_lr(
    train_X, y=target,
    x=[f + '_woe' for f in selected],
    cv=3, max_num_features=30)
```

**注意事项：** `stepwise_lr` 的 `y` 参数为列名字符串，传入的 DataFrame 必须包含该列

#### Phase 5: 模型精调（系数方向 + P值 + VIF）

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

    coef_params = model.params.iloc[1:]
    coef_pvalues = model.pvalues.iloc[1:]
    if np.any(coef_params > 0) or np.any(coef_pvalues > 0.05):
        t_vals = model.tvalues.iloc[1:]
        rm_var = t_vals.index[t_vals == t_vals.max()].item()
        selected_woe.remove(rm_var)
    else:
        break

# VIF
X_arr = X.to_numpy()
for i, feat in enumerate(selected_woe):
    print(f'{feat}: VIF={variance_inflation_factor(X_arr, i + 1):.2f}')
```

**注意事项：**
- 系数方向检查和 P 值检查**必须排除截距项**（`iloc[1:]`）
- VIF > 10 提示严重多重共线性，应考虑剔除

#### Phase 6: 评分卡生成

**目的：** 将逻辑回归系数转换为评分卡

**关键 API：** `syriskmodels.scorecard.make_scorecard()`

**代码模板：**
```python
from syriskmodels.scorecard import make_scorecard

scorecard = make_scorecard(bins, model.params.to_dict(),
                           base_points=600, base_odds=50, pdo=20)
```

#### Phase 7: 模型评估（AUC / KS / Gains Table）

**目的：** 评估模型区分能力

**关键 API：**
- `syriskmodels.evaluate.model_eval()` — AUC / KS
- `syriskmodels.evaluate.gains_table()` — Gains Table

**代码模板：**
```python
from syriskmodels.evaluate import model_eval, gains_table

all_X = woebin_ply(df[原始特征列], bins)[selected_woe]
all_X = sm.add_constant(all_X)
df['prob'] = model.predict(all_X)

perf = df.groupby('dataset_flag').apply(model_eval, target=target, pred='prob')

gt, breaks = gains_table(train_y, train_scores, return_breaks=True)
```

#### Phase 8: 稳定性分析（PSI）

**目的：** 评估变量和模型的稳定性

**关键 API：**
- `syriskmodels.scorecard.woebin_psi()` — 变量 PSI
- `syriskmodels.evaluate.psi()` — 分布 PSI

#### Phase 9: 输出模型文档

**目的：** 将全部分析结果输出为 Excel 文件

**产出 Excel 结构：**
| Sheet | 内容 |
|-------|------|
| 样本统计 | 各维度样本分布 |
| WOE分析 | WOE 表 |
| IV分析 | IV 表 |
| 模型表达 | 回归摘要 + VIF + 评分卡 |
| 模型评估 | AUC/KS + Gains Table（Train/Test/OOT） |
| PSI分析 | 变量 PSI + 模型 PSI |

### 2.4 支持文件

#### `api-reference.md`

所有 syriskmodels 公共 API 的完整参考，按模块组织：
- `syriskmodels.datasets` — 数据加载
- `syriskmodels.scorecard` — 分箱与评分卡
- `syriskmodels.models` — 建模
- `syriskmodels.evaluate` — 评估
- `syriskmodels.utils` — 工具函数
- `syriskmodels.contrib` — 高级流程

每个函数包含：签名、参数说明、返回值、示例。

#### `workflow-example.md`

基于 `germancredit` 数据集的完整端到端示例代码，从数据加载到 Excel 输出，可直接复制执行。

## 三、不在范围内

- 不修改任何函数签名或行为逻辑（BUG 修复除外）
- 不新增分箱算法
- 不新增依赖
- 不改动测试用例（若 BUG 修复导致测试需更新，则跟进）

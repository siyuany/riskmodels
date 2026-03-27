# syriskmodels API Reference

本文档覆盖评分卡开发全流程中使用的 syriskmodels 公共 API。

## syriskmodels.datasets — 数据加载

### `load_germancredit() -> pd.DataFrame`

加载德国信用数据集（1000 行 x 21 列）。目标变量 `creditability`，1 = 坏样本。

```python
from syriskmodels.datasets import load_germancredit
df = load_germancredit()
```

### `load_creditcard() -> pd.DataFrame`

加载信用卡欺诈数据集。目标变量 `Class`，1 = 欺诈。包含 V1-V28 (PCA特征)、Time、Amount。

```python
from syriskmodels.datasets import load_creditcard
df = load_creditcard()
```

### `get_data_dir() -> Path`

返回内置数据集目录路径（仓库根目录的 `data/`）。

---

## syriskmodels.scorecard — 分箱与评分卡

### `woebin(dt, y, x=None, ..., methods=None, **kwargs) -> Dict[str, DataFrame | str]`

WOE 分箱主函数。

**参数:**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dt` | DataFrame | — | 包含目标变量和解释变量的数据框 |
| `y` | str / List[str] | — | 目标变量名 |
| `x` | str / List[str] / None | None | 解释变量名列表，None 时使用除 y 外所有列 |
| `var_skip` | str / List[str] / None | None | 跳过的变量列表 |
| `breaks_list` | Dict[str, List] / None | None | 自定义切分点 `{变量名: [切分点]}` |
| `special_values` | List / Dict / None | None | 特殊值（如 -999, -1） |
| `positive` | int / float | 1 | 正样本标识值 |
| `no_cores` | int / None | None | 多进程数量 |
| `methods` | List[str / type] / None | `['quantile', 'tree']` | 分箱方法列表 |
| `max_cate_num` | int | 50 | 类别变量最大类别数 |
| `**kwargs` | — | — | 传递给分箱器的参数 |

**常用 kwargs:**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `initial_bins` | 20 | 细分箱数量 |
| `bin_num_limit` | 5 | 粗分箱最大数量 |
| `count_distr_limit` | 0.05 | 分箱最小样本占比 |
| `stop_limit` | 0.05 | 分箱停止阈值 |
| `ensure_monotonic` | False | 强制单调性（仅树分箱） |

**methods 组合示例:**

| 组合 | 说明 |
|------|------|
| `['quantile', 'tree']` | 等频 + 树分箱（推荐） |
| `['quantile', 'chi2']` | 等频 + ChiMerge |
| `['quantile', 'tree', 'chi2']` | 等频 → 树 → ChiMerge 三级 |
| `['quantile']` | 仅等频（纯无监督） |
| `['hist', 'tree']` | 等距 + 树分箱 |

**返回:** `{变量名: DataFrame | 'CONST' | 'TOO_MANY_VALUES'}`

分箱结果 DataFrame 包含列：variable, bin, bin_chr, count, count_distr, good, bad, badprob, woe, bin_iv, total_iv, breaks, is_special_values

### `sc_bins_to_df(sc_bins) -> Tuple[DataFrame, DataFrame]`

将 `woebin()` 结果转为 woe_df 和 iv_df。

**返回:**
- `woe_df`: 所有变量的分箱统计汇总
- `iv_df`: 每个变量的 IV 统计，按 IV 降序排列，包含 IV、IV区间、单调性、最大Lift、最小Lift

```python
woe_df, iv_df = sc_bins_to_df(bins)
```

### `woebin_ply(dt, bins, value='woe') -> DataFrame`

应用分箱结果转换数据。

**参数:**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dt` | DataFrame | — | 原始数据 |
| `bins` | Dict | — | `woebin()` 返回结果 |
| `value` | str | `'woe'` | 转换类型: `'woe'` / `'index'` / `'bin'` |

**返回:** DataFrame，转换后的列名带后缀 `_woe`、`_index` 或 `_bin`

### `woebin_breaks(bins) -> Tuple[Dict, Dict]`

从分箱结果提取切分点和特殊值。

**返回:** `(breaks_dict, special_values_dict)`

### `woebin_psi(df_base, df_cmp, bins) -> DataFrame`

计算变量 PSI。

**参数:**
- `df_base`: 基准数据集（训练集）
- `df_cmp`: 比较数据集（OOT）
- `bins`: 入模变量的分箱结果

**返回:** DataFrame，包含 variable, bin, base_distr, cmp_distr, psi

### `woebin_plot(bins, x=None, show_iv=True) -> Dict[str, Figure]`

绘制分箱 bivar 图。

**返回:** `{变量名: matplotlib.Figure}`

### `make_scorecard(sc_bins, coef, *, base_points=600, base_odds=50, pdo=20) -> DataFrame`

将逻辑回归系数转换为评分卡。

**参数:**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `sc_bins` | Dict | — | `woebin()` 返回结果 |
| `coef` | Dict[str, float] | — | `{'const': 截距, '变量_woe': 系数, ...}` |
| `base_points` | int | 600 | 基准分 |
| `base_odds` | int | 50 | 基准 odds |
| `pdo` | int | 20 | 翻倍 odds 的分数增量 |

**返回:** DataFrame，包含 variable, bin, woe, score

---

## syriskmodels.models — 建模

### `stepwise_lr(df, y, x, cv=3, max_num_features=30, direction='bidirectional') -> Tuple[float, List[str]]`

双向逐步逻辑回归特征选择。

**参数:**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `df` | DataFrame | — | **必须包含 y 列** |
| `y` | str | — | 目标变量**列名**（非 array） |
| `x` | List[str] | — | 候选特征列名 |
| `cv` | int / generator | 3 | 交叉验证折数或自定义 splitter |
| `max_num_features` | int | 30 | 最大特征数 |
| `direction` | str | `'bidirectional'` | `'forward'` / `'backward'` / `'bidirectional'` |

**返回:**
- `best_metrics` (float): 最优评估指标
- `selected_features` (List[str]): 选中的特征列名

---

## syriskmodels.evaluate — 评估

### `model_eval(df, target, pred) -> pd.Series`

评估 Auc、KS 和坏样本率。常与 `groupby().apply()` 配合使用。

**返回:** `pd.Series`，index 为 `['bad_rate', 'auc', 'ks']`

### `gains_table(y_true, y_score, split=None, breaks=None, return_breaks=False)`

生成 Gains Table（模型排序能力评估表）。

**参数:**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `y_true` | array-like | — | 真实标签 |
| `y_score` | array-like | — | 模型分数 |
| `split` | int / None | 10 | 等分段数 |
| `breaks` | array / None | None | 自定义切分点 |
| `ascending` | bool | True | 按分数升序排列 |
| `return_breaks` | bool | False | 是否同时返回 breaks |

**返回 DataFrame 列:**
TotalCnt, GoodCnt, BadCnt, Odds, BadRate, Lift, CumBadRate, BadPercent, CumBadPercent, GoodPercent, CumGoodPercent, KS, TotalPercent

### `psi(base_distr, cmp_distr, epsilon=1e-3) -> float`

计算 PSI (Population Stability Index)。

### `ks_score(y_true, y_pred) -> float`

计算 KS 统计量。

---

## syriskmodels.utils — 工具函数

### `sample_stats(df, target) -> pd.Series`

统计样本分布。

**返回:** `pd.Series`，index 为 `['TotalCnt', 'GoodCnt', 'BadCnt', 'BadRate']`

```python
df.groupby('month').apply(sample_stats, target='y')
```

### `monotonic(series) -> str`

判断序列单调性。返回值: `'increasing'`, `'decreasing'`, `'up_u_shape'`, `'down_u_shape'`, `'non_monotonic'`

---

## syriskmodels.contrib — 高级流程

### `risk_trends_consistency(df, sc_bins, target) -> Dict[str, float]`

判断变量在不同数据集上的风险趋势一致性。

**参数:**
- `df`: 验证数据集（OOT）
- `sc_bins`: 训练集的分箱结果
- `target`: 目标变量列名

**返回:** `{变量名: Spearman秩相关系数}`。1.0 = 完全一致，-1.0 = 完全相反。

### `build_scorecard(sample_df, features, target, train_filter, oot_filter, output_excel_file, ...)`

一键评分卡构建流程。将数据准备、分箱、筛选、建模、评估、输出全部自动完成。

**注意：** 该函数为自动化流程，适合快速原型。如需精细控制每个步骤，请按 SKILL.md 中的分步流程操作。
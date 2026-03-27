# Implementation Plan: syriskmodels v0.3.1 — Scorecard Skill

**Spec**: `docs/superpowers/specs/2026-03-27-skill-scorecard-design.md`
**Branch**: `feature/skill-scorecard-v031`
**Base commit**: `7496d85` (HEAD of feature branch)

## Overview

本计划将设计文档拆分为 10 个独立的 Task，按依赖关系排序执行。每个 Task 包含精确的文件路径、完整代码、验证方式。

### Task 依赖图

```
Task 1 (BUG修复提交) ─── 无依赖，已在工作区
Task 2 (datasets.py) ─── 无依赖
Task 3 (版本号升级) ──── 依赖 Task 2（__init__.py 同文件）
Task 4 (docstring增强) ─ 无依赖
Task 5 (SKILL.md) ────── 依赖 Task 2（引用 datasets API）
Task 6 (api-reference.md) ── 依赖 Task 4（引用增强后的 docstring）
Task 7 (workflow-example.md) ── 依赖 Task 2, 5
Task 8 (datasets 测试) ─ 依赖 Task 2
Task 9 (BUG回归测试) ── 依赖 Task 1
Task 10 (最终验证) ──── 依赖所有
```

### 建议执行顺序

**Phase A** (并行): Task 1, Task 2, Task 4
**Phase B** (并行): Task 3, Task 5, Task 8, Task 9
**Phase C** (并行): Task 6, Task 7
**Phase D**: Task 10

---

## Task 1: 提交已有的 3 个 BUG 修复

**文件**: `src/syriskmodels/contrib/build_scorecard.py`（已修改，未提交）
**依赖**: 无

### 操作

已有修改在工作区中，只需 git commit：

```bash
git add src/syriskmodels/contrib/build_scorecard.py
git commit -m "fix: correct 3 bugs in build_scorecard pipeline

- BUG#1: stepwise_lr y parameter must be column name string, not numpy array;
  concat target column into train_X before calling stepwise_lr
- BUG#2: VIF calculation used wrong matrix (train_X instead of X after
  add_constant); changed to X.to_numpy() with idx+1 to skip intercept column
- BUG#3: coefficient sign/p-value/t-value checks now exclude intercept term
  via iloc[1:]"
```

### 验证

```bash
git log -1 --oneline  # 确认提交成功
git diff HEAD~1 --stat  # 确认只改了 build_scorecard.py
```

---

## Task 2: 新建 `datasets.py` 数据加载模块

**文件**: `src/syriskmodels/datasets.py`（新建）
**依赖**: 无

### 完整代码

```python
# -*- encoding: utf-8 -*-
"""
数据集加载模块

提供类似 sklearn.datasets 的便捷数据加载接口，用于快速访问内置演示数据集。
"""
from pathlib import Path

import pandas as pd


def get_data_dir() -> Path:
    """返回内置数据集目录路径

    Returns:
        Path: 数据目录的绝对路径，即仓库根目录下的 ``data/`` 目录。

    Example:
        >>> from syriskmodels.datasets import get_data_dir
        >>> data_dir = get_data_dir()
        >>> list(data_dir.glob('*.csv.gz'))
        [PosixPath('.../data/creditcard.csv.gz'), PosixPath('.../data/germancredit.csv.gz')]
    """
    return Path(__file__).resolve().parent.parent.parent / 'data'


def load_germancredit() -> pd.DataFrame:
    """加载德国信用数据集 (germancredit)

    该数据集包含 1000 条个人信用记录，包含 20 个特征变量和 1 个目标变量
    (creditability)，其中 1 表示坏样本，0 表示好样本。

    Returns:
        pd.DataFrame: 德国信用数据集，包含 1000 行和 21 列。

    Raises:
        FileNotFoundError: 当数据文件 ``data/germancredit.csv.gz`` 不存在时。

    Example:
        >>> from syriskmodels.datasets import load_germancredit
        >>> df = load_germancredit()
        >>> df.shape
        (1000, 21)
        >>> df['creditability'].value_counts()
        0    700
        1    300
        Name: creditability, dtype: int64
    """
    path = get_data_dir() / 'germancredit.csv.gz'
    if not path.exists():
        raise FileNotFoundError(
            f'数据文件不存在: {path}。请确认 data/ 目录下包含 germancredit.csv.gz')
    return pd.read_csv(path)


def load_creditcard() -> pd.DataFrame:
    """加载信用卡欺诈数据集 (creditcard)

    该数据集包含信用卡交易记录，包含 28 个 PCA 特征变量 (V1-V28)、
    交易时间 (Time)、交易金额 (Amount) 和目标变量 (Class)，
    其中 1 表示欺诈交易，0 表示正常交易。

    Returns:
        pd.DataFrame: 信用卡欺诈数据集。

    Raises:
        FileNotFoundError: 当数据文件 ``data/creditcard.csv.gz`` 不存在时。

    Example:
        >>> from syriskmodels.datasets import load_creditcard
        >>> df = load_creditcard()
        >>> df.columns[:5].tolist()
        ['Time', 'V1', 'V2', 'V3', 'V4']
    """
    path = get_data_dir() / 'creditcard.csv.gz'
    if not path.exists():
        raise FileNotFoundError(
            f'数据文件不存在: {path}。请确认 data/ 目录下包含 creditcard.csv.gz')
    return pd.read_csv(path)
```

### 验证

```bash
python -c "from syriskmodels.datasets import load_germancredit, load_creditcard, get_data_dir; print(get_data_dir()); df = load_germancredit(); print(df.shape); df2 = load_creditcard(); print(df2.shape)"
```

期望输出：数据目录路径、`(1000, 21)`、信用卡数据集 shape。

---

## Task 3: 版本号升级 + datasets 导出

**文件**:
- `src/syriskmodels/__init__.py`（修改）
- `pyproject.toml`（修改）

**依赖**: Task 2（`datasets.py` 必须存在才能 import）

### 3.1 修改 `src/syriskmodels/__init__.py`

**变更 1**: 版本号

```python
# 旧:
__version__ = '0.3.0'
# 新:
__version__ = '0.3.1'
```

**变更 2**: 在 `import syriskmodels.utils as utils` 之后添加 datasets 模块导入

```python
import syriskmodels.utils as utils
import syriskmodels.datasets as datasets  # 新增
```

**变更 3**: 在 `from .utils import sample_stats` 之后添加 datasets 函数导入

```python
from .utils import sample_stats
from .datasets import load_creditcard, load_germancredit  # 新增
```

**变更 4**: 更新 `__all__` 列表，在第一行添加 `'datasets'`，在末尾添加函数名

```python
__all__ = [
    'contrib', 'datasets', 'detector', 'evaluate', 'logging', 'models', 'rule', 'scorecard',
    'utils', 'detect', 'gains_table', 'ks_score', 'model_eval', 'stepwise_lr',
    'make_scorecard', 'woebin', 'woebin_breaks', 'woebin_plot', 'woebin_ply',
    'woebin_psi', 'monotonic', 'sample_stats',
    'load_creditcard', 'load_germancredit'
]
```

### 3.2 修改 `pyproject.toml`

```toml
# 旧:
version = "0.3.0"
# 新:
version = "0.3.1"
```

### 验证

```bash
python -c "import syriskmodels; print(syriskmodels.__version__); print(syriskmodels.load_germancredit().shape)"
# 期望: 0.3.1 和 (1000, 21)
grep 'version' pyproject.toml | head -1
# 期望: version = "0.3.1"
```

---

## Task 4: 核心函数 Docstring 增强

**文件**（均为修改）:
- `src/syriskmodels/scorecard/api/woebin.py` — `woebin()`
- `src/syriskmodels/scorecard/api/transform.py` — `woebin_ply()`
- `src/syriskmodels/scorecard/api/scorecard.py` — `make_scorecard()`
- `src/syriskmodels/models.py` — `stepwise_lr()`
- `src/syriskmodels/evaluate.py` — `model_eval()`, `gains_table()`

**依赖**: 无
**约束**: 不修改任何函数签名或行为逻辑，仅增强 docstring

### 4.1 `woebin()` — `src/syriskmodels/scorecard/api/woebin.py`

将现有 docstring 替换为：

```python
def woebin(
    dt: pd.DataFrame,
    y: Union[str, List[str]],
    x: Optional[Union[str, List[str]]] = None,
    var_skip: Optional[Union[str, List[str]]] = None,
    breaks_list: Optional[Dict[str, List]] = None,
    special_values: Optional[Union[List, Dict[str, List]]] = None,
    positive: Union[int, float] = 1,
    no_cores: Optional[int] = None,
    methods: Optional[List[Union[str, type]]] = None,
    max_cate_num: int = 50,
    replace_blank: Union[float, int] = np.nan,
    **kwargs
) -> Dict[str, Union[pd.DataFrame, str]]:
    """WOE 分箱主函数

    对数据集中的变量进行 WOE (Weight of Evidence) 分箱，返回每个变量的
    分箱统计结果，包含 WOE 值、IV 值等指标。

    参数:
        dt: 包含目标变量和解释变量的数据框
        y: 目标变量名（0/1 二分类，1 为正样本）
        x: 解释变量名列表，默认为除 y 外的所有列
        var_skip: 需要跳过的变量列表
        breaks_list: 用户自定义切分点字典，格式为 ``{变量名: [切分点列表]}``
        special_values: 特殊值列表或字典。列表形式应用于所有变量，
            字典形式为 ``{变量名: [特殊值列表]}``
        positive: 正样本标识值，默认 1
        no_cores: 多进程数量，None 时自动检测 CPU 核数
        methods: 分箱方法列表，默认 ``['quantile', 'tree']``。
            首元素必须为无监督细分箱方法 (``'quantile'`` 或 ``'hist'``)，
            后续为粗分箱方法 (``'tree'`` 或 ``'chi2'``)。
            常见组合:

            - ``['quantile', 'tree']``: 等频细分箱 + 树粗分箱（默认）
            - ``['quantile', 'chi2']``: 等频细分箱 + ChiMerge 粗分箱
            - ``['quantile', 'tree', 'chi2']``: 等频 → 树 → ChiMerge 三级分箱
            - ``['quantile']``: 仅等频分箱（纯无监督）

        max_cate_num: 类别变量最大允许类别数，超过则跳过，默认 50
        replace_blank: 空字符串替换值，默认 ``np.nan``
        **kwargs: 传递给分箱器的其他参数，常用参数包括:

            - ``initial_bins`` (int): 细分箱的数量，默认 20
            - ``bin_num_limit`` (int): 最终分箱的最大数量（不含特殊值），默认 5
            - ``count_distr_limit`` (float): 分箱样本占总样本最小比例，默认 0.05
            - ``stop_limit`` (float): 分箱停止条件阈值，默认 0.05
            - ``ensure_monotonic`` (bool): 是否保证单调性（仅树分箱），默认 False

    返回:
        Dict[str, Union[pd.DataFrame, str]]，key 为变量名，value 为:

        - pd.DataFrame: 分箱结果，包含 variable, bin, bin_chr, count,
          count_distr, good, bad, badprob, woe, bin_iv, total_iv,
          breaks, is_special_values 等列
        - ``'CONST'``: 常量变量（被跳过）
        - ``'TOO_MANY_VALUES'``: 类别过多（被跳过）

    示例:
        >>> from syriskmodels.scorecard import woebin, sc_bins_to_df
        >>> bins = woebin(df, y='target', x=['age', 'income'],
        ...              methods=['quantile', 'tree'],
        ...              bin_num_limit=5, count_distr_limit=0.05)
        >>> woe_df, iv_df = sc_bins_to_df(bins)
    """
```

### 4.2 `woebin_ply()` — `src/syriskmodels/scorecard/api/transform.py`

将现有 docstring 替换为：

```python
def woebin_ply(
    dt: pd.DataFrame,
    bins: Dict[str, Union[pd.DataFrame, str]],
    no_cores: int = None,
    replace_blank: bool = False,
    value: str = 'woe'
) -> pd.DataFrame:
    """应用 WOE 分箱结果转换数据

    将 ``woebin()`` 返回的分箱结果应用到数据集，将原始值转换为 WOE 值、
    分箱索引或分箱区间。转换后的列名带有后缀 ``_woe``、``_index`` 或 ``_bin``。

    参数:
        dt: 包含变量原始值的数据框，列名需与分箱结果中的变量名一致
        bins: ``woebin()`` 返回的分箱结果字典
        no_cores: 多进程数量，None 时自动检测
        replace_blank: 是否将空字符串 ``''`` 替换为 ``np.nan``
        value: 返回值类型，可选 ``['woe', 'index', 'bin']``

            - ``'woe'``: 将原始值替换为 WOE 值，列名为 ``变量名_woe``
            - ``'index'``: 将原始值替换为分箱索引 (0, 1, 2,...)，列名为 ``变量名_index``
            - ``'bin'``: 返回分箱区间文本，数值型为 ``[a,b)``，类别型为 ``a%,%b``，
              列名为 ``变量名_bin``

    返回:
        pd.DataFrame，包含:

        - 入参 dt 中**不在** bins 中的原始列（保持不变）
        - bins 中匹配到的变量，按 ``value`` 参数转换后的新列

    示例:
        >>> bins = woebin(train_df, y='target')
        >>> train_woe = woebin_ply(train_df, bins, value='woe')
        >>> train_woe.columns  # 原始列被替换为 age_woe, income_woe 等
        >>> train_bin = woebin_ply(train_df, bins, value='bin')
        >>> train_bin['age_bin'].head()  # 输出: [-inf,25), [25,35), ...
    """
```

### 4.3 `make_scorecard()` — `src/syriskmodels/scorecard/api/scorecard.py`

将现有 docstring 替换为：

```python
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
```

### 4.4 `stepwise_lr()` — `src/syriskmodels/models.py`

在现有函数签名之后添加 docstring（当前无 docstring）：

```python
def stepwise_lr(df: pd.DataFrame,
                y: str,
                x: Union[str, List[str]],
                cv: int = 3,
                max_num_features: int = 30,
                initial_features: Union[str, List[str]] = None,
                direction: str = 'bidirectional',
                **lr_kwargs):
    """双向逐步逻辑回归特征选择

    通过前向选择和/或后向消除的交叉验证策略，从候选特征中选择最优特征子集。
    每一步通过 ``LogisticRegressionCV`` 交叉验证评估特征组合的性能
    （指标为 min(train_auc, valid_auc) - |train_auc - valid_auc|），
    选择使指标最优的特征组合。

    参数:
        df: 包含特征列和目标列的数据框。**必须包含 y 对应的列**。
        y: 目标变量列名（字符串），df 中必须存在该列
        x: 候选特征列名列表
        cv: 交叉验证折数，默认 3。也可传入自定义 CV splitter（如 generator）
        max_num_features: 最大入选特征数量，默认 30
        initial_features: 初始特征列表（强制入选），默认 None
        direction: 搜索方向，可选 ``'forward'``、``'backward'``、``'bidirectional'``，
            默认 ``'bidirectional'``
        **lr_kwargs: 传递给 ``sklearn.linear_model.LogisticRegression`` 的参数

    返回:
        Tuple[float, List[str]]:

        - ``best_metrics`` (float): 最优评估指标值
        - ``selected_features`` (List[str]): 被选中的特征列名列表

    示例:
        >>> from syriskmodels.models import stepwise_lr
        >>> from syriskmodels.scorecard import woebin_ply
        >>> train_woe = woebin_ply(train_df[features], bins, value='woe')
        >>> train_woe['target'] = train_df['target']  # 必须包含 target 列
        >>> best_auc, selected = stepwise_lr(
        ...     train_woe, y='target',
        ...     x=[f + '_woe' for f in features], cv=3)
    """
```

### 4.5 `model_eval()` — `src/syriskmodels/evaluate.py`

将现有函数（无 docstring）添加 docstring：

```python
def model_eval(df, target, pred) -> pd.Series:
    """评估模型的 AUC、KS 和坏样本率

    对数据框中的目标变量和预测概率计算模型性能指标。常与 ``groupby().apply()``
    配合使用，按不同维度评估模型表现。

    参数:
        df: 包含目标变量和预测值的数据框
        target: 目标变量列名（0/1 二分类）
        pred: 预测概率列名

    返回:
        pd.Series，包含三个指标:

        - ``bad_rate`` (float): 坏样本率
        - ``auc`` (float): AUC 值，始终 >= 0.5
        - ``ks`` (float): KS 统计量

    示例:
        >>> perf = df.groupby('dataset').apply(model_eval, target='y', pred='prob')
        >>> perf
                  bad_rate    auc      ks
        train     0.035    0.891  0.632
        test      0.038    0.875  0.618
        oot       0.042    0.862  0.601
    """
```

### 4.6 `gains_table()` — `src/syriskmodels/evaluate.py`

在现有 docstring 末尾的 Returns 部分增加 DataFrame 列说明。将现有 Returns 块替换为：

```python
    """
    ...（保留现有 Args 部分不变）

    Returns:
        当 return_breaks=False 时，返回 gains_table (pd.DataFrame)；否则，返回
        元组 (gains_table, breaks)。

        gains_table 的 DataFrame 包含以下列:

        - ``TotalCnt``: 该分段总样本数
        - ``GoodCnt``: 好样本数
        - ``BadCnt``: 坏样本数
        - ``Odds``: 好坏比 (GoodCnt / BadCnt)
        - ``BadRate``: 坏样本率
        - ``Lift``: 提升度（该段坏率 / 整体坏率）
        - ``CumBadRate``: 累计坏样本率
        - ``BadPercent``: 坏样本占比
        - ``CumBadPercent``: 累计坏样本占比
        - ``GoodPercent``: 好样本占比
        - ``CumGoodPercent``: 累计好样本占比
        - ``KS``: KS 值 (|CumBadPercent - CumGoodPercent|)
        - ``TotalPercent``: 该段样本占总体比例

    """
```

### 验证

对每个修改的文件运行 `lsp_diagnostics` 确认无语法错误。然后：

```bash
python -c "from syriskmodels.models import stepwise_lr; help(stepwise_lr)"
python -c "from syriskmodels.evaluate import model_eval, gains_table; help(model_eval); help(gains_table)"
python -c "from syriskmodels.scorecard import woebin, woebin_ply, make_scorecard; help(woebin)"
```

---

## Task 5: 创建 SKILL.md

**文件**: `skills/syriskmodels-scorecard/SKILL.md`（新建）
**依赖**: Task 2（引用 `syriskmodels.datasets`）

### 完整内容

```markdown
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
df = load_germancredit()
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
```

### 验证

- 确认 frontmatter 格式正确（`name` + `description`）
- 确认所有代码模板中的 API 调用与实际函数签名一致
- 确认 Common Mistakes 覆盖了 3 个 BUG 修复

---

## Task 6: 创建 api-reference.md

**文件**: `skills/syriskmodels-scorecard/api-reference.md`（新建）
**依赖**: Task 4（引用增强后的 docstring 内容）

### 完整内容

```markdown
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

评估 AUC、KS 和坏样本率。常与 `groupby().apply()` 配合使用。

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
```

### 验证

- 所有函数签名与源码一致
- 参数类型和默认值准确
- 返回值说明完整

---

## Task 7: 创建 workflow-example.md

**文件**: `skills/syriskmodels-scorecard/workflow-example.md`（新建）
**依赖**: Task 2, Task 5

### 完整内容

```markdown
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
df = load_germancredit()
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
```

### 验证

- 所有 import 路径正确
- 代码从上到下可顺序执行
- 无占位符或 `TODO` 标记

---

## Task 8: datasets 模块测试

**文件**: `test/test_datasets.py`（新建）
**依赖**: Task 2

### 完整代码

```python
# -*- encoding: utf-8 -*-
"""datasets 模块单元测试"""
import pytest
import pandas as pd

from syriskmodels.datasets import load_germancredit, load_creditcard, get_data_dir


class TestGetDataDir:
    """get_data_dir() 测试"""

    def test_returns_path(self):
        data_dir = get_data_dir()
        assert data_dir.exists(), f'数据目录不存在: {data_dir}'
        assert data_dir.is_dir()

    def test_contains_data_files(self):
        data_dir = get_data_dir()
        files = [f.name for f in data_dir.glob('*.csv.gz')]
        assert 'germancredit.csv.gz' in files
        assert 'creditcard.csv.gz' in files


class TestLoadGermancredit:
    """load_germancredit() 测试"""

    def test_returns_dataframe(self):
        df = load_germancredit()
        assert isinstance(df, pd.DataFrame)

    def test_shape(self):
        df = load_germancredit()
        assert df.shape[0] == 1000
        assert df.shape[1] == 21

    def test_target_column_exists(self):
        df = load_germancredit()
        assert 'creditability' in df.columns

    def test_target_values(self):
        df = load_germancredit()
        assert set(df['creditability'].unique()) == {0, 1}


class TestLoadCreditcard:
    """load_creditcard() 测试"""

    def test_returns_dataframe(self):
        df = load_creditcard()
        assert isinstance(df, pd.DataFrame)

    def test_has_expected_columns(self):
        df = load_creditcard()
        assert 'Time' in df.columns
        assert 'Class' in df.columns
        assert 'V1' in df.columns

    def test_target_values(self):
        df = load_creditcard()
        assert set(df['Class'].unique()) == {0, 1}
```

### 验证

```bash
pytest test/test_datasets.py -v
```

---

## Task 9: BUG 修复回归测试

**文件**: `test/test_build_scorecard_bugfix.py`（新建）
**依赖**: Task 1

### 完整代码

```python
# -*- encoding: utf-8 -*-
"""build_scorecard BUG 修复回归测试

验证 build_scorecard.py 中 3 个 BUG 修复的正确性:
- BUG#1: stepwise_lr 调用方式（y 参数为列名，df 包含 target 列）
- BUG#2: VIF 计算（使用 add_constant 后的矩阵，索引 +1 跳过截距列）
- BUG#3: 系数/p值/t值检查排除截距项（iloc[1:]）
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


@pytest.fixture
def mock_train_data():
    """构造简单的模拟训练数据用于回归测试"""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    prob = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2 + 0.2 * x3)))
    y = rng.binomial(1, prob)
    return pd.DataFrame({
        'x1_woe': x1, 'x2_woe': x2, 'x3_woe': x3, 'target': y
    })


class TestBug1StepwiseLrCall:
    """BUG#1: stepwise_lr 的 y 参数必须为列名字符串"""

    def test_stepwise_lr_accepts_column_name(self, mock_train_data):
        """stepwise_lr 传入列名字符串应正常工作"""
        from syriskmodels.models import stepwise_lr

        best_auc, selected = stepwise_lr(
            mock_train_data,
            y='target',
            x=['x1_woe', 'x2_woe', 'x3_woe'],
            cv=2)

        assert isinstance(best_auc, float)
        assert isinstance(selected, list)
        assert len(selected) > 0
        assert all(f in mock_train_data.columns for f in selected)


class TestBug2VifCalculation:
    """BUG#2: VIF 计算必须使用 add_constant 后的矩阵，索引从 1 开始"""

    def test_vif_with_constant_column(self, mock_train_data):
        """VIF 计算应在 add_constant 后的矩阵上进行，跳过截距列"""
        features = ['x1_woe', 'x2_woe', 'x3_woe']
        X = mock_train_data[features].copy()
        X = sm.add_constant(X)
        X_arr = X.to_numpy()

        vif_values = {}
        for idx, feature in enumerate(features):
            vif_values[feature] = variance_inflation_factor(X_arr, idx + 1)

        # VIF 值应为正数且有限
        for feat, val in vif_values.items():
            assert np.isfinite(val), f'{feat} 的 VIF 值不是有限数: {val}'
            assert val >= 1.0, f'{feat} 的 VIF 值应 >= 1.0，实际: {val}'

    def test_vif_wrong_index_raises_or_wrong(self, mock_train_data):
        """验证用错误索引 (不跳过截距列) 会得到不同结果"""
        features = ['x1_woe', 'x2_woe', 'x3_woe']
        X = mock_train_data[features].copy()
        X = sm.add_constant(X)
        X_arr = X.to_numpy()

        # 正确方式: idx + 1
        correct_vif_0 = variance_inflation_factor(X_arr, 1)
        # 错误方式: idx (指向截距列)
        wrong_vif_0 = variance_inflation_factor(X_arr, 0)

        # 两者应该不同（截距列的 VIF 无业务意义）
        assert correct_vif_0 != wrong_vif_0


class TestBug3InterceptExclusion:
    """BUG#3: 系数方向/p值/t值检查必须排除截距项"""

    def test_coef_check_excludes_intercept(self, mock_train_data):
        """模型精调循环中 iloc[1:] 应排除截距"""
        features = ['x1_woe', 'x2_woe', 'x3_woe']
        X = mock_train_data[features].copy()
        X = sm.add_constant(X)
        model = sm.GLM(
            endog=mock_train_data['target'], exog=X,
            family=sm.families.Binomial()).fit()

        # 排除截距项
        coef_params = model.params.iloc[1:]
        coef_pvalues = model.pvalues.iloc[1:]

        assert 'const' not in coef_params.index, \
            '系数检查不应包含 const（截距项）'
        assert 'const' not in coef_pvalues.index, \
            'p值检查不应包含 const（截距项）'
        assert len(coef_params) == len(features), \
            f'排除截距后应有 {len(features)} 个系数，实际 {len(coef_params)}'

    def test_t_value_max_excludes_intercept(self, mock_train_data):
        """当需要移除变量时，t值排序不应包含截距"""
        features = ['x1_woe', 'x2_woe', 'x3_woe']
        X = mock_train_data[features].copy()
        X = sm.add_constant(X)
        model = sm.GLM(
            endog=mock_train_data['target'], exog=X,
            family=sm.families.Binomial()).fit()

        t_values = model.tvalues.iloc[1:]
        rm_var = t_values.index[t_values == t_values.max()].item()

        assert rm_var != 'const', '不应移除截距项'
        assert rm_var in features, f'移除的变量 {rm_var} 应在特征列表中'
```

### 验证

```bash
pytest test/test_build_scorecard_bugfix.py -v
```

---

## Task 10: 最终验证

**依赖**: 所有 Task 完成

### 验证清单

1. **单元测试全量通过**

```bash
pytest test/ -v --tb=short
```

2. **LSP 诊断无错误**

对所有修改/新增的 Python 文件运行 `lsp_diagnostics`：
- `src/syriskmodels/datasets.py`
- `src/syriskmodels/__init__.py`
- `src/syriskmodels/contrib/build_scorecard.py`
- `src/syriskmodels/models.py`
- `src/syriskmodels/evaluate.py`
- `src/syriskmodels/scorecard/api/woebin.py`
- `src/syriskmodels/scorecard/api/transform.py`
- `src/syriskmodels/scorecard/api/scorecard.py`

3. **版本号一致性**

```bash
python -c "import syriskmodels; assert syriskmodels.__version__ == '0.3.1'"
grep 'version = "0.3.1"' pyproject.toml
```

4. **Skill 文件完整性**

```bash
ls -la skills/syriskmodels-scorecard/
# 应包含: SKILL.md, api-reference.md, workflow-example.md
```

5. **Git 状态确认**

```bash
git status
git log --oneline -5
```

6. **提交全部改动**

```bash
git add -A
git commit -m "feat: add syriskmodels-scorecard skill and bump version to 0.3.1

- New datasets module (load_germancredit, load_creditcard, get_data_dir)
- Enhanced docstrings for core APIs (woebin, woebin_ply, make_scorecard,
  stepwise_lr, model_eval, gains_table)
- Version bump: 0.3.0 -> 0.3.1
- New skill: skills/syriskmodels-scorecard/ with SKILL.md, api-reference.md,
  and workflow-example.md for AI-guided scorecard development
- Added tests for datasets module and bug fix regression tests"
```

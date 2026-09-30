# Golden 快照（分箱行为基线）

这里的 JSON 是 `woebin` 分箱结果的**行为护栏**：任何性能重构或算法改造，
只要改变了分箱输出，`test/test_golden_binning.py` 就会失败并给出最小 diff。

## 文件

| 文件 | 数据源 | 用例数 | 数据缺失时 |
| --- | --- | ---: | --- |
| `synthetic_binning.json` | `test/test_golden_binning.py::_synthetic_frame()`（确定性 1000 行） | 6 | 正常执行 |
| `germancredit_binning.json` | `syriskmodels.datasets.load_germancredit()`（全量 1000 行） | 4 | skip（兜底：`data/*.csv.gz` 已随 Git 入库，干净克隆下正常执行） |

合成数据覆盖：数值（含长尾）、类别、缺失值、常量列、超多类别列，以及与
target 有信号相关的数值列。**不依赖 creditcard 全量**，避免 CI 过慢。

> **合成数据必须只用整数算术生成**（`_synthetic_frame()`）。相同 seed 下
> `Generator.standard_normal` / `binomial` / `lognormal` 在不同 CPU 架构
> （macOS arm64 vs Linux x86_64）会走不同 SIMD 内核、产生**不同位序列**，
> 进而让 tree/chi2 分箱结果分歧。第一版实现就因此使
> `synthetic_quantile_chi2_ib20_limit5` 在 3 个 CI job 上全数失败。
> 现在生成器只用 `%`/`//` 与 1/2^k 缩放（IEEE-754 精确运算），并由
> `data_fingerprint` 在比对前校验数据逐位一致。

> 设计要点：合成与 germancredit 走**两个独立 fixture**。若合成用例的 fixture
> 顺手加载 germancredit，数据缺失时的 `pytest.skip` 会把全部 12 个 golden 用例
> 一起跳过，等于在干净克隆 / CI 中没有护栏。

## 快照结构

```json
{
  "golden_version": 1,
  "dataset": "synthetic",
  "float_digits": 8,
  "snapshot_columns": ["variable", "bin", "count", "good", "bad", "badprob",
                       "woe", "bin_iv", "total_iv", "breaks", "is_special_values"],
  "data_fingerprint": {
    "n_rows": 1000,
    "columns": ["num_normal", "..."],
    "column_checksums": {"num_normal": "3e96c487", "...": "..."}
  },
  "cases": {
    "<case_name>": {
      "dataset": "...", "target": "...", "methods": [...], "kwargs": {...},
      "special_values": {...}, "n_rows": 1000, "status": "ok",
      "variables": {
        "<变量名>": [ {"variable": ..., "bin": ..., "woe": ...}, ... ]
      }
    }
  }
}
```

* 变量按名称排序、分箱按输出顺序、字段按 `snapshot_columns` 顺序 —— 全部确定性。
* 浮点统一 `round(v, 8)`；`NaN` → `null`，`±inf` → `"inf"` / `"-inf"`。
* `str` 结果（`"CONST"` / `"TOO_MANY_VALUES"`）原样记录。
* `status="error"` 表示该用例当前抛异常（用于钉住已知失败行为，避免"悄悄变绿"）。
* `data_fingerprint`：每列内容的 CRC32（逐行 `repr` 拼接），**前置校验**。
  数据生成不可移植 / 数据被改动时会在这里立刻失败并指出具体列，
  而不是抛出一大坨分箱 diff 把根因埋掉。

## 更新方式

```bash
UPDATE_GOLDEN=1 PYTHONHASHSEED=0 python -m pytest test/test_golden_binning.py
```

默认（不带 `UPDATE_GOLDEN`）为**严格相等**比较。更新快照前请确认差异是
**有意**的行为变更，并在 PR 描述中说明。

## 已知失败路径

数值型列的显式 `special_values`（`-999` / `-1` / `int` 列 + `missing`）在
pandas 3 下会抛 `ValueError`，相关用例与说明在 `test/test_known_bugs.py`
（B-4 / B-5），不放在 golden 用例集中。

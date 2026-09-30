# Golden 快照（分箱行为基线）

这里的 JSON 是 `woebin` 分箱结果的**行为护栏**：任何性能重构或算法改造，
只要改变了分箱输出，`test/test_golden_binning.py` 就会失败并给出最小 diff。

## 文件

| 文件 | 数据源 | 用例数 | 无数据时 |
| --- | --- | ---: | --- |
| `synthetic_binning.json` | `test/test_golden_binning.py::_synthetic_frame()`（固定种子 1000 行） | 6 | 正常执行 |
| `germancredit_binning.json` | `syriskmodels.datasets.load_germancredit()`（全量 1000 行） | 4 | skip（数据不入版本库） |

合成数据覆盖：数值（含长尾）、类别、缺失值、常量列、超多类别列，以及与
target 有信号相关的数值列。**不依赖 creditcard 全量**，避免 CI 过慢。

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

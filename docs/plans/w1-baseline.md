# W1 计划：建立可复现的测试与性能基线

> 状态：**已执行完成**（详见 [w1-baseline-report.md](w1-baseline-report.md)）
> 分支：`feature/w1-test-baseline`
> 注：`docs/` 已被 `.gitignore` 忽略，本计划与报告为**本地产物**；
> 需要入库时用 `git add -f docs/plans/w1-baseline-report.md`。

## 1. 背景 / 问题

`riskmodels`（`syriskmodels`）已完成 pandas 3 升级，但缺少"能挡住回归"的基础设施：

1. `pytest test/` 在干净克隆下 **7 failed**：集成测试硬编码
   `test/germancredit.csv` / `test/creditcard.csv`，而仓库只有被 gitignore 的
   `test/*.csv.gz` 软链接；
2. 无 CI：`.github/workflows` 不存在，pandas 3 兼容性缺陷（`woebin_plot`
   `KeyError`、数值型 `special_values` `ValueError`）长期无人发现；
3. 无 golden 快照：分箱行为一旦漂移，只能靠人眼比对；
4. 无性能基线：后续"性能重构 + 算法改造"没有护栏，无法证明"只是变快、没改行为"；
5. 测试本身存在确定性隐患（`np.random.rand` 未播种、`set` 遍历顺序依赖
   `PYTHONHASHSEED`、默认走 multiprocessing）。

## 2. 目标与非目标

### 目标（W1 交付）

- 干净克隆 + `data/*.csv.gz` 存在时，`pytest -q -m "not slow"` 全绿；
- `pytest -q -m "slow"` 在数据存在时通过、缺失时明确 skip 并说明原因；
- GitHub Actions CI（语法正确、可运行、带 pip 缓存、`PYTHONHASHSEED=0`）；
- golden 快照测试 + `UPDATE_GOLDEN=1` 重生成；
- 可重复运行的 benchmark 与 1 份 baseline JSON；
- Bug / 风险清单（含复现方式与目标阶段）。

### 非目标（W1 明确不做）

- **不修任何业务 Bug**（分箱 / 模型 / 评分卡逻辑一律不动）；
- 不做算法或接口重构（`woebin`、`woebin_ply`、`make_scorecard` 等签名与行为不变）；
- 不引入重型依赖（仅 `pytest` / `pytest-cov` / `openpyxl`）；
- 不解决 pandas 3 兼容缺陷本身，只记录 + `xfail` 回归用例。

## 3. 方案设计

| 维度 | 方案 |
| --- | --- |
| 数据访问 | 测试统一经 `syriskmodels.datasets.load_germancredit()/load_creditcard()`；缺失时 `skip`（带原因），另用 `SYRISKMODELS_DATA_DIR` 指向空目录来验证 `FileNotFoundError` 契约 |
| 确定性 | 全部固定种子（`np.random.default_rng`）；分箱调用显式 `no_cores=1`；要求 `PYTHONHASHSEED=0`（未设置时 golden 用例 skip 并提示）；变量遍历改为显式排序 |
| golden | JSON 快照，列 `variable/bin/count/good/bad/badprob/woe/bin_iv/total_iv/breaks/is_special_values`，浮点 `round(v, 8)`，结构完全排序 → 严格相等比较 |
| benchmark | `bench/common.py` 提供环境采集与 `time_case()`（1 warmup + N 次取中位数）；两个基准脚本 + `run_all.py`；输出 JSON 含机器/版本/参数/中位数/每变量耗时 |
| CI | 4 job：`unit`(3.11) / `unit-py312` / `integration`(slow) / `golden`；`addopts` 不加默认 `-m`，由 CI 显式指定 |
| 兼容性 | 公共 API（`woebin`、`woebin_ply`、`woebin_breaks`、`make_scorecard`、`sc_bins_to_df`、`woebin_psi`、`woebin_plot`、`stepwise_lr`）签名与行为零改动；`src/` 零改动 |

## 4. 兼容性影响

- 测试语义：断言口径**保持不变**，仅形式调整（`unittest.TestCase.setUp` →
  pytest autouse fixture、模块级数据缓存）；
- `pyproject.toml`：新增可选依赖与 pytest 配置，不影响运行时依赖；
- `.gitignore`：仅新增 `bench/results/*` 忽略（保留 `.gitkeep`），
  不动既有规则；
- 用户代码：无影响（`src/` 未改）。

## 5. 实施步骤

- [x] 建分支 `feature/w1-test-baseline`
- [x] 只读审计：README / pyproject / src 主要模块 / test 全部测试
- [x] 跑基线 `pytest test/` 记录失败与耗时
- [x] 任务 B：`pyproject.toml` dev extra + `[tool.pytest.ini_options]`
- [x] 任务 A：集成测试数据路径修复 + `no_cores=1` + slow 标记
- [x] 任务 A2：`test/conftest.py` 数据可用性处理 + `test_datasets` 契约用例
- [x] 任务 D：`test/test_golden_binning.py` + `test/golden/*.json`
- [x] 任务 D2：`test/test_known_bugs.py`（`xfail(strict=True)` 回归用例）
- [x] 任务 E：`bench/` 基准脚本 + `bench/results/baseline_20260930.json`
- [x] 任务 C：`.github/workflows/ci.yml`
- [x] 任务 F：`docs/plans/w1-baseline-report.md`
- [x] 最终验证：`-m "not slow"` / `-m "slow"` / `-m "golden"` 全跑

## 6. 测试计划

| 层级 | 覆盖 |
| --- | --- |
| 单元 | datasets 契约、detect 确定性、scorecard 常量/异常/validation/binning_helpers |
| 集成 | germancredit 全变量分箱；creditcard 全量分箱 + `build_scorecard` 流水线（xfail@B-2） |
| golden | 10 个用例（合成 6 + germancredit 4），12 个测试（含 2 份整体比对） |
| 回归 | B-1..B-12 的 xfail/护栏用例 |
| 性能 | 8 个固定用例的中位数基线 |

## 7. 风险与回滚

| 风险 | 处置 |
| --- | --- |
| golden 快照过大 / 误报 | 只存确定性字段、浮点 round；失败时打印逐 bin 逐字段最小 diff |
| `UPDATE_GOLDEN` 被滥用 | 默认严格比较；快照内嵌 `golden_version`；README 要求 PR 说明 |
| CI 无数据集导致 integration 空跑 | job 内打印数据可用性；支持 `RISKMODELS_DATA_URL` 自动拉取 |
| 测试改动掩盖既有缺陷 | 缺陷一律 `xfail(strict=True)`（修复后自动变红提醒清理），不做 `skip` |
| 回滚 | 全部改动集中在 `test/`、`bench/`、`docs/`、`pyproject.toml`、`.github/`；`git revert` 单个 merge 即可，`src/` 未改 |

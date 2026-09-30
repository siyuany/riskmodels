# W2 开发计划：缺陷修复 + 分箱性能内核重构（feature/w2-binning-kernel）

> 状态：**维护者已在任务书中审定**（本文件为归档副本，权威依据为
> `docs/plans/w1-baseline-report.md` 与维护者下发的 W2 指令）。
> 分支：`feature/w2-binning-kernel`（自 `develop@372265c` 开出）→ 目标合并回 `develop`。

## 1. 背景 / 问题

W1 已建立可复现的测试与性能基线：121 passed / 11 xfailed、CI 4 job 全绿、
golden 快照 12 用例、bench 基线（tree ib=100 50k×7 变量 6.26s，chi2 ib=100 3.43s）。
遗留两类问题：

1. **缺陷 B-1..B-13**：其中 B-2/B-3/B-4/B-5/B-9 为 P0（阻断端到端流程或结果错误），
   B-6/B-7/B-10/B-11 为 P1，B-8/B-12/B-13 为 P2（B-1、B-14 已在 W1 修复）。
2. **性能热点**：`TreeOptimBin.merge_binning` 每个候选切点全量 groupby 重建
   （O(n_bins²) pandas 开销）；`ChiMergeOptimBin.chi2_stat` 每轮全量重算；
   `initial_bins` 是主成本驱动（20→100 箱耗时 ×5.7）。

## 2. 目标与非目标

### 目标

1. 修完 B-1..B-13（B-1 验证不回退）；每项 xfail(strict=True) 改为正向断言。
2. 性能内核重构（**不改变分箱输出语义**）：
   - `BinCountTable`（`core/counts.py`）+ 聚合向量化；
   - `TreeOptimBin` / `ChiMergeOptimBin` NumPy 精确等价实现；
   - `RuleOptimBin` 向量化；
   - `ComposedWOEBin` 缓存首级计数表。
3. NumPy 等价性与 golden 全绿后，实现 Numba 编译后端
   （`@njit(cache=True, nogil=True, fastmath=False)`，engine='auto'|'numpy'|'numba'，
   import 失败自动 fallback，小数据不走 Numba，不与 mp 混用）。
4. W2 性能前后对比（bench/results/w2_before.json vs w2_after.json）、
   更新报告（`docs/plans/w2-kernel-refactor-report.md`）、CI 全绿、PR 就绪。

### 性能验收（creditcard 50k × 7 数值变量，no_cores=1）

| 用例 | W1 基线 | W2 目标 |
| --- | ---: | --- |
| tree ib=100 | 6.259s（本机复测 6.579s） | ≤1.3s（≥5x） |
| tree ib=20 | 1.107s（复测 1.260s） | 不高于基线 |
| chi2 ib=100 | 3.425s（复测 3.403s） | ≤0.7s（≥5x） |
| chi2 ib=20 | 0.411s（复测 0.391s） | 不高于基线 |
| quantile | — | 无回退 |
| Numba（若完成） | — | tree ib=100 单变量核心搜索 <1ms；chi2 ib=500 <5ms；端到端 ≤0.3s（如主导项允许） |

### 非目标（留给 W3+）

- 验证集感知分箱、稳定性约束分箱、WOE 平滑；
- stepwise 目标函数重构、beam search；
- build_scorecard pipeline/配置体系重构；
- 跨时间稳定性框架、校准与监控；
- `woebin_ply` 性能优化（W1 已证明非瓶颈，本轮不改）。

## 3. 方案设计

### 3.1 缺陷修复（Phase 1）

| Bug | 位置 | 修法 |
| --- | --- | --- |
| B-2 (P0) | `api/evaluation.py:181` | `groupby.apply` 丢失分组列 → 显式 merge/transform 补回 `variable`，图形语义不变；移除集成用例 xfail+witness |
| B-3 (P0) | `models.py:130-159` | 拆开"生成候选"与"择优更新"，direction 只控制候选集合 |
| B-4 (P0) | `core/base.py:104-118` | merge key dtype 统一（object↔float64），保持 NaN/特殊值/数值区间语义 |
| B-5 (P0) | 同上 | 仅无 NaN 时 astype 原 dtype，否则保持 float/object |
| B-6 (P1) | `core/factory.py:122` | 按构造函数签名过滤 kwargs 分发；保留实例/类/注册名三种传参 |
| B-7 (P1) | `bins/optimal.py:285-297` | `RuleOptimBin.__init__` 接收并透传 `**kwargs` |
| B-8 (P2) | `utils/validation.py:222-231` | `eval` → `ast.literal_eval`，保留 dict 校验 |
| B-9 (P0) | `api/evaluation.py` pivot 后处理 + `evaluate.psi` | 两侧分箱取并集 reindex，缺失侧显式 fillna(0) 后归一化；极端缺失给有限可解释 PSI 或告警 |
| B-10 (P1) | `contrib/build_scorecard.py:139-141` | 新增 `risk_consistency_dataset='valid'|'oot'`，默认 'valid'；选 'oot' 时 warning；OOT 只做最终评估 |
| B-11 (P1) | `api/woebin.py`、`api/transform.py` | 默认 `no_cores=1`；显式 >1 才并行，用安全进程上下文 + 超时/错误传播；独立 commit |
| B-12 (P2) | `utils/binning_helpers.py` | 审计后二选一：接入生产路径或标记 deprecated；测试改为验证与生产语义一致 |
| B-13 (P2) | `test/test_scorecard/conftest.py` | fixture 改依赖注入；新增实际消费这些 fixture 的测试 |

### 3.2 BinCountTable（Phase 2）

`src/syriskmodels/scorecard/core/counts.py`：

```python
@dataclass
class BinCountTable:
    variable: str
    bin_chr: list[str] | np.ndarray
    good: np.ndarray   # int64
    bad: np.ndarray    # int64
    is_numeric: bool
    # 只读属性：count / total / woe / iv
```

- `WOEBin.binning` 的 `_n0/_n1` lambda → 向量化 good/bad 聚合；
- 特殊值拆分改布尔掩码/映射（保持 B-4/B-5 修复后的 dtype 语义）；
- `initial_binning` 排序语义不变：数值按区间序、类别按 badprob 降序；
- 粗分箱算法只依赖该结构，不再在候选循环里碰原始 DataFrame。

### 3.3 Tree/ChiMerge/Rule NumPy 等价实现（Phase 3-5）

- **Tree**：前缀和 + segments 数组；保持 cp 标记、count_distr/单调约束、
  `(curr_iv - last_iv + 1e-8)/(last_iv + 1e-8) > min_iv_inc` 接受条件、
  tie 取最小索引、off-by-one 行为、breaks 提取语义完全一致。
- **ChiMerge**：向量化 2×2 Yates 修正 χ²（与 scipy `chi2_contingency(correction=True)`
  数值一致；边界期望为 0 → 0）；保留 min_chi2/count_distr/n_bins 三分支决策、
  idx 修正规则、增量维护与折叠正则语义。先 O(k²) 向量化，不达标再考虑
  heap/链表（不改变输出）。
- **Rule**：保留 cut_binning/foil/lift/fisher_exact/min_hit_samples 逻辑，
  向量化累计计数；fisher_exact 只对通过前置条件的候选调用。

### 3.4 ComposedWOEBin 缓存（Phase 6）

首级（quantile/hist）完成后缓存 BinCountTable，tree/chi2/rule 复用；
不破坏单独调用 `woebin(dtm, breaks)` 的行为。

### 3.5 Numba 后端（Phase 7）

- 内核：定长连续数组 + 标量入参，输出段尾索引；不在内核内碰 pandas/object/字符串；
  分类变量在边界层编码为整数；`fastmath=False` 强制。
- `engine='auto'|'numpy'|'numba'`；auto 按 initial_bins/样本量/变量数选择；
  numba import 失败自动 fallback；小数据默认 NumPy；不与 mp 混用；
  不在模块导入时强制编译。

## 4. 兼容性影响

- 公共 API（woebin、woebin_ply、woebin_breaks、sc_bins_to_df、make_scorecard、
  WOEBinFactory、WOEBin 及子类）签名与返回类型保持兼容；
- 默认分箱输出不变：golden 快照默认必须逐位一致；B-10 属方法学修正，
  `build_scorecard` 默认筛选数据集从 OOT 改为 valid（新增参数可显式回退，
  有 warning），集成用例相应更新并在 W2 报告说明；
- `woebin`/`woebin_ply` 默认 `no_cores` 行为从"自动并行"改为 1（B-11），
  这是缺陷修复的一部分，显式传 `no_cores>1` 仍可并行。

## 5. 实施步骤

- [ ] Phase 0：基线确认全绿；bench before 入库；文档勘误；开分支
- [ ] Phase 1：B-1..B-13 逐项修复（先改测试，再改实现，逐项回归）
- [ ] Phase 2：BinCountTable + 聚合向量化
- [ ] Phase 3：Tree NumPy 等价 + `test/test_binning_equivalence.py`（旧实现参考拷贝 oracle）
- [ ] Phase 4：ChiMerge NumPy 等价
- [ ] Phase 5：Rule 向量化
- [ ] Phase 6：ComposedWOEBin 缓存
- [ ] Phase 7：Numba 后端 + 等价性
- [ ] Phase 8：golden/差分/bench after/CI 全绿
- [ ] Phase 9：W2 报告、W1 状态更新、按建议拆分提交、PR

提交拆分：`fix(pandas3)` B-2/B-4/B-5/B-9；`fix(models)` B-3；`fix(factory)` B-6/B-7/B-8；
`fix(methodology)` B-10；`fix(parallel)` B-11；`refactor(binning)`；`perf(tree/chimerge/rule/composed/numba)`；
`test(binning)`；`bench`；`docs`。

## 6. 测试计划

- 每个 Bug：xfail(strict=True) → 正向断言；集成用例移除 xfail/witness；
- `test/test_binning_equivalence.py`：内置旧实现参考拷贝（仅测试用）作 oracle，
  随机差分覆盖数值/类别、缺失/特殊值、ib=20/100/500、bin_num_limit=3/5、
  count_distr_limit、monotonic on/off、并列候选；固定种子；
  NumPy vs 旧参考、Numba vs NumPy 双链路；
- golden：现有快照保持不变；新增 B-4/B-5 修复后的 special_values 边界 golden；
- 所有测试/bench 一律 `no_cores=1`、`PYTHONHASHSEED=0`。

## 7. 风险与回滚

| 风险 | 缓解 |
| --- | --- |
| 向量化实现与旧实现存在浮点/并列 tie 差异 | 参考拷贝差分测试 + golden 逐位比对；tie 一律取最小索引，与旧实现相同 |
| Numba 浮点误差导致 tie 不稳定 | 以 NumPy 参考实现为准，收紧比较逻辑；fastmath=False |
| B-9/B-10 行为变化影响下游 | 正常场景输出不变；变化点新增针对性测试并在报告中显式说明 |
| 性能目标不达标 | 分阶段验证（NumPy 先行，Numba 增量）；heap/链表作为后备优化 |
| 回滚 | 每 Task 独立 commit；`git revert` 单提交即可回退；golden 全绿是合并门槛 |

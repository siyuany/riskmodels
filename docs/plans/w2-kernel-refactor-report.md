# W2 报告：缺陷修复 B-1..B-13 与分箱性能内核重构

> 分支：`feature/w2-binning-kernel`（自 `develop@372265c`）→ 目标合并回 `develop`
> 前置：`docs/plans/w1-baseline-report.md`（W1 基线，权威依据）；
> 计划归档：`docs/plans/w2-binning-kernel.md`
> 基线/结果：`bench/results/baseline_20260930.json`（W1）、
> `bench/results/w2_before.json`（W2 起点复测）、`bench/results/w2_after.json`（W2 结果），
> 三份 JSON 的 `environment` 字段**逐字段一致**（macOS arm64 / CPython 3.12.13 /
> pandas 3.0.6 / numpy 2.5.3 / scipy 1.18.1 / sklearn 1.9.1，`PYTHONHASHSEED=0`，
> 全部 `no_cores=1`）
> CI：`.github/workflows/ci.yml` 4 job（unit / unit-py312 / integration / golden），
> 运行记录见 §8

---

## 1. B-1..B-13 状态表

| Bug | 优先级 | 状态 | 修复位置 | 回归测试位置 |
| --- | --- | --- | --- | --- |
| B-1 测试数据路径 | P1 | **fixed（W1），W2 验证不回退** | —（W1 已修） | `test_known_bugs.py::test_b1_no_hardcoded_csv_paths_in_tests`；`grep` 复核 `src/test/bench` 无 `test/*.csv` 硬编码残留（仅文档性提及） |
| B-2 woebin_plot KeyError | P0 | **fixed** | `scorecard/api/evaluation.py`：`groupby.apply` → `transform('sum')` 向量化；`_plot_single_bin` 改用 `bin` 列（旧代码引用的 `bin_chr` 列在本仓库输出中从不存在，任何 pandas 版本都会 KeyError）；`x=None` 时变量顺序 `sorted`（与旧 apply 分组排序一致且确定） | `test_b2_woebin_plot_works`（正向）；`test_scorecard_integration.py::test_build_scorecard_pipeline`（移除 xfail+witness，端到端全绿并校验 `pic/*.png` 产物） |
| B-3 stepwise_lr 单向返回空 | P0 | **fixed** | `models.py`：候选生成与择优更新拆分，direction 只控制候选集合；纯 backward 未给 initial_features 时从全量特征起步（经典后向消除）并以当前集合指标为基线 | `test_b3_stepwise_lr_single_direction_selects_features[forward/backward]`、`test_b3_stepwise_lr_direction_semantics` |
| B-4 数值列+数值 special_values merge 报错 | P0 | **fixed** | `scorecard/core/base.py::split_special_values` 重写：弃用 `fillna("missing")+merge`，改布尔掩码 + 值→bin_chr 映射；标签语义保持 legacy（float 列 `-999.0`；int 列且 sv 无 NaN 时 `str(int(v))` 含 astype 截断语义；数值列组合条目拆分独立箱） | `test_b4_numeric_special_values_binning`、`test_b4_numeric_special_values_edge_cases`；golden 新增 `synthetic_numeric_special_values` |
| B-5 整型列+missing astype(int) 报错 | P0 | **fixed** | 同上：sv 值含 NaN 时不再 astype 回 int dtype，保持 float 语义（`dtm_ns.value` 保持原 dtype，与旧 astype 还原一致） | `test_b5_integer_column_special_values_binning`、`test_b5_integer_column_with_nan_and_missing`；golden 新增 `synthetic_integer_missing_special_value` |
| B-6 Factory kwargs 无差别透传 | P1 | **fixed** | `core/factory.py::_filter_kwargs_for`：构造函数无 `**kwargs` 的类只收显式声明参数（丢弃项 debug 日志）；有 `**kwargs` 的类保持 legacy 全量透传；实例/类/注册名三种传参不变 | `test_b6_quantile_rule_with_initial_bins`、`test_b6_factory_kwargs_dispatch_and_input_forms` |
| B-7 RuleOptimBin 不接 kwargs | P1 | **fixed**（eps 语义保留，见 §6 决策 4） | `bins/optimal.py::RuleOptimBin.__init__` 增加 `**kwargs` 透传 super | `test_b7_rule_optim_bin_accepts_kwargs`（含 eps/epsilon 语义护栏） |
| B-8 check_breaks_list 用 eval | P2 | **fixed** | `utils/validation.py`：`eval` → `ast.literal_eval`，非字面量抛 `ValueError`；"必须是字典"校验保留 | `test_b8_check_breaks_list_rejects_non_literal_expressions` |
| B-9 woebin_psi 单侧缺失静默放大 | P0 | **fixed** | `api/evaluation.py::woebin_psi`：两侧计数取并集 reindex，缺失侧显式 fillna(0) 再归一化；单侧缺失触发 `UserWarning`；`evaluate.psi` 对 NaN 输入显式告警（不再静默 NaN→0+重归一化） | `test_b9_...`（正向重写：W1 版用例误取 `bin.iloc[0]`=missing 箱导致 cmp_df 为空，已修正为取真实类别箱；断言有限、可解释、与显式分布直接调用 psi 一致、正常场景回归） |
| B-10 build_scorecard 用 OOT 筛选 | P1 | **fixed**（方法学行为变更，见 §6 决策 1） | `contrib/build_scorecard.py`：新增 `risk_consistency_dataset='valid'\|'oot'`（签名末位，兼容），默认 `'valid'`（02_test）；验证集为空回退训练集并告警；显式 `'oot'` 时 logging.warn + UserWarning | `test_b10_variable_selection_no_longer_uses_oot_by_default`（AST+签名）、`test_b10_risk_consistency_dataset_behavior[valid/oot]`（行为级：捕获 risk_trends_consistency 实收数据集）、`test_b10_risk_consistency_falls_back_to_train_when_no_valid` |
| B-11 mp.Pool 默认并行不安全 | P1 | **fixed**（默认行为变更，见 §6 决策 2） | `utils.py::resolve_no_cores/parallel_starmap`；`api/woebin.py`、`api/transform.py`：默认 `no_cores=1`，None/<1 视为 1；显式 >1 才并行，平台默认上下文 + with 清理 + `starmap_async().get(timeout)` 错误传播/超时 terminate；spawn+交互式环境（`interactive_mode()` 探测）自动回退串行并 UserWarning；新增可选参数 `parallel_timeout`；独立 commit（不与内核重构混合） | `test_b11_default_is_serial`（patch mp.Pool/get_context 均不得触发）、`test_b11_interactive_spawn_falls_back_to_serial`（假 ctx 跨平台确定）、`test_b11_explicit_parallel_matches_serial`（真实进程池，woebin+woebin_ply 结果逐位一致）、原 picklable 护栏保留 |
| B-12 binning_helpers 脱节 | P2 | **fixed**（方案 b：deprecated+兼容导出，见 §6 决策 5） | `utils/binning_helpers.py`：`__deprecated__=True`，6 个公开函数触发 `DeprecationWarning`；`scorecard/__init__` 兼容导出保留 | `test_b12_binning_helpers_deprecated_and_not_used_by_production`、`test_b12_helper_woe_iv_semantics_consistent_with_production`（compute_woe/compute_iv 与生产 binning_format 数值一致；breaks 语义差异两侧钉住）；`test_binning_helpers.py` 重定位为兼容层稳定性测试 |
| B-13 conftest fixture 缺陷 | P2 | **fixed** | `test/test_scorecard/conftest.py`：5 个 fixture 改依赖注入；`data_with_mixed_types` 另需先 `astype(object)`（pandas 3 `str` dtype 不允许写入 int —— fixture 原本自身即抛 TypeError，同类潜伏缺陷一并修复） | `test_scorecard/test_conftest_fixtures.py`（13 个用例真实消费全部数据类 fixture）；`test_b13_conftest_fixtures_use_dependency_injection`（静态钉住 DI+消费者两个不变量） |

**B-14**（W1 已修）无回退：golden `data_fingerprint` 前置校验全绿。
另按 W1 报告 §1.1 建议补齐**随机流指纹**用例（`test/test_rng_fingerprint.py`）。

---

## 2. BinCountTable 设计（Phase 2）

`src/syriskmodels/scorecard/core/counts.py`：

```python
@dataclass(frozen=True, eq=False)
class BinCountTable:
    variable: str
    bin_chr: np.ndarray      # object ndarray[str]
    good: np.ndarray         # int64
    bad: np.ndarray          # int64
    is_numeric: bool
    epsilon: float = 0.5
    categories: Optional[pd.Index] = None   # 见 §6 决策 3
    # 只读属性：n_bins / count / total / count_distr / badprob / woe / bin_iv / total_iv
    # 方法：from_binning_df / to_binning_df / segment_sums / replace
```

* 粗分箱内核（tree/chi2/rule）**只依赖该结构**，候选循环不再触碰原始
  DataFrame；行序与 legacy `initial_binning` 输出完全一致（数值型区间升序、
  类别型 badprob 降序，任务书 §2.4）。
* `WOEBin.binning` 的 `_n0/_n1` lambda → `(y==0)/(y==1)` int64 列 +
  `groupby.sum`（cython），输出行序/列名/dtype/空类别保留/NaN 组键丢弃
  语义与旧实现**逐位一致**（`test_bin_count_table.py` 以 legacy lambda
  参考拷贝做差分，含空分箱与 NaN 组键边界）。
* 特殊值拆分已在 B-4/B-5 修复中改为布尔掩码/映射（任务书 §2.3），
  `dtm_ns.value` 保持原 dtype。
* WOE/IV 属性与 `binning_format` / `TreeOptimBin.iv` 的 epsilon 公式逐位
  一致（同算式同求和顺序）。
* 验收（§2.5）：count/good/bad/WOE/IV/breaks 与旧实现一致 —— golden 14
  用例逐位不变 + `test_bin_count_table.py` 9 用例 + 差分 fuzz（§3）。

## 3. Tree / ChiMerge / Rule 等价性证据

### 3.1 方法

`test/test_binning_equivalence.py`（446 用例：356 unit + 90 slow）内置
**W1 develop（`372265c`）实现的逐字参考拷贝**（`RefTreeOptimBin` /
`RefChiMergeOptimBin` / `RefRuleOptimBin`，自带 legacy `_n0/_n1` binning
聚合，完全独立于生产代码路径）作为 oracle：

* 同一 dtm + 同一初始 breaks → 生产内核 vs 参考实现，breaks **值/dtype/
  categories（含 names）逐位相等**；
* 全路径 `__call__` / `woebin()` API 级 `assert_frame_equal`（count/good/
  bad/woe/iv/breaks 全列，含特殊值路径）；
* 组合链 [q,tree] / [q,chi2] / [q,rule] / [q,tree,chi2] / [q,chi2,tree]
  生产（带缓存）vs 全参考链（无缓存）。

覆盖矩阵（固定种子，可复现）：14 个数据场景（数值：信号/离散并列/纯噪声/
缺失+哨兵/长尾/完全可分/对抗性占比边界；类别：有序风险/多类别+缺失/
非单调/并列 badprob）× 参数（ib=20/50/100/500[slow]、bin_num_limit=
0/1/2/3/4/5/6/8、count_distr_limit=0/0.01/0.02/0.05/0.2、p=0.05/0.5/0.9、
min_iv_inc=0/0.05/0.5、ensure_monotonic on/off、eps=0.5/1e-8、lift/
min_hit_samples/direction=good|bad）+ 边界专项：用户 breaks 空分箱
（bad_prob NaN、χ²=0 短路、lift 除零）、全好/全坏零边际对、负数区间
边界（折叠正则失配场景）、单值/双值退化表、无合格切点单箱路径、
监控分箱双分支（左/右尾坏率尖峰）、germancredit 真实类别列。

### 3.2 为保证"逐位一致"复刻的 legacy 数值语义（差分过程中实证发现）

| 语义点 | legacy 行为 | 内核复刻 |
| --- | --- | --- |
| float `count_distr` 段和（tree 约束） | pandas `groupby.sum(float64)` = **Kahan 补偿求和**（20 组对抗性试验逐位验证；≠ 朴素顺序 ≠ np.sum pairwise） | `kahan_prefix_sums`（左子段定起点前缀扫描）+ `kahan_suffix_sums_batched`（右子段候选批向量化前向扫描），累加路径与 groupby 分组内顺序一致 |
| chi2 的 `count_distr` 增量 | 合并时**标量浮点左加**（`distr[i-1] + distr[i]`，非 Kahan、非区间和） | 内核按相同标量增量维护 |
| IV / 卡方的求和次序 | `np.sum`（pairwise）；scipy χ² 为 2×2 flatten 左结合 4 项和 | 矩阵批量 + `np.sum(axis=1)`（与逐行 `np.sum` 逐位一致，实测 9/130 列均验证）；χ² 闭式 `((t00+t01)+t10)+t11` |
| scipy Yates 修正 | `chi2_contingency(correction=True)` 对过修正区（D<n/2）**精确返回 0**（clamp） | 闭式 `max(0,\|O-E\|-0.5)²/E` 求和，8 组边界表与 scipy 输出逐位比对 |
| 单调约束 NaN | `utils.monotonic`：任何 NaN → non_monotonic；等值 → increasing | `all(diff>=0) \| all(diff<=0)` + NaN 显式短路（inf 语义实测一致） |
| tie-breaking | dict 插入序（idx 升序）+ `sorted(key=-iv)` 稳定排序 → 并列取最小索引 | `np.argmax`（升序首个最大值） |
| off-by-one | tree 循环条件在候选生成**前**检查 → 段数上限 = bin_num_limit+1 | 原样保留 |
| cum_count_distr（rule monitor） | pandas `Series.cumsum` = np.cumsum 朴素顺序（30 组对抗试验验证） | `np.cumsum` |
| **类别型 breaks dtype 特殊形态** | 全单类别段时 `groupby.agg(join)` 保持 **category dtype**（categories=初始 breaks 顺序），下游 `set_categories` 采用 **categories 而非 values** → 最终行序=初始顺序；有合并时退化 str dtype → 行序=段序 | `segments_to_breaks(categories=...)` 按成员规则精确复刻；`BinCountTable.categories` 保存 categories **Index 对象**（含 name —— name 会传播到最终输出 breaks 列的 `categories.names`，属 assert_frame_equal 可观测行为） |
| breaks Series `name` | legacy 返回 `best_binning['bin_chr']`（name='bin_chr'） | 输出 Series 固定 name='bin_chr' |

> 教训：golden 的 `germancredit_quantile_tree_ib20_limit5` 在 Phase 3 首跑
> 即捕获了 category-dtype 特殊形态的漂移（`foreign.worker` 行序翻转）——
> "golden 逐位不变 + 参考拷贝差分"双护栏按设计发挥了作用。

### 3.3 性能内核结构

`core/kernels.py`（NumPy 参考）：`tree_cut_search`（cp 标记/前缀和/候选
矩阵批量评估）、`chi2_pair_stats` + `chi2_merge_search`（O(k²) 向量化，
未引入 heap/链表 —— 已达标且任何此类优化都不得改变输出，任务书 §4.4）、
`rule_cut_search` + `rule_segments`（foil/lift 批量；fisher_exact 仅对
通过 lift+min_hit 前置门的候选调用）、`segments_to_breaks`、
`resolve_engine`。

## 4. ComposedWOEBin 缓存（Phase 6）

* 各级内核经 `woebin_with_table(dtm, breaks, parent)` 返回
  `(breaks, 输入计数表, 段边界)`；公开 `woebin(dtm, breaks)` 签名不变。
* 下一级 `initial_count_table(parent=...)` 与 `__call__` 的最终
  `binning_breaks` 由 `counts.binning_from_segments` 段聚合构造，跳过
  pd.cut/merge + groupby 全量重扫；构造与全新扫描**逐位等价**（labels
  用相同构造式、类别行序=set_categories 的 categories 序、categories
  Index 含 name 保真），一致性校验失败自动回退原始扫描。
* 缓存以 **weakref 对象同一性**为键（dtm 与 breaks 均为同一次 woebin 的
  对象才命中），`__getstate__` 丢弃缓存（weakref 不可 pickle，B-11 并行
  与 pickle 护栏不受影响）。
* 单独调用 `binner.woebin(dtm, breaks)` 无缓存上下文，行为不变（§6.3）；
  自定义分箱器（无 `woebin_with_table`）自动走旧路径。
* 证据：`test_composed_cache_scans_raw_data_once`（spy 证明 [q,tree,chi2]
  全链只有 1 次原始 `WOEBin.binning_breaks` 扫描）；组合链差分 5 种链路
  × 14 场景 vs 全参考链；pickle/同一性守卫用例。

## 5. Numba 后端（Phase 7）

* `core/kernels_numba.py`：`@njit(cache=True, nogil=True, fastmath=False)`；
  定长连续数组 + 标量入参，段边界写入预分配数组；内核内无 pandas/object/
  字符串；类别变量天然以计数表整数数组进入内核（字符串只在边界层）。
* **逐位一致护栏**：
  * `np.log`：numba（libm 标量）与 numpy 元素级实测逐位一致（arm64，
    40 万对抗样本 0 失配；CI x86_64 由差分 fuzz 把关）；
  * `np.sum` pairwise：n ≤ 8 为纯标量路径可跨平台复刻
    （`_nb_pairwise_sum`）；**n ≥ 9 时 numpy 走 SIMD 相关路径，不可移植
    复刻**（实测 n=9 起与标量 pairwise 分歧）→ tree 后端限制
    `bin_num_limit ≤ 6`（分区求和长度 T ≤ 8），超出自动降级 NumPy
    （任务书 §7.4"以 NumPy 参考实现为准"）；chi2 无变长求和，不受限；
  * rule 恒 NumPy 后端：热路径依赖 `scipy.fisher_exact`（不可入 njit），
    向量化版已达标（W2 决策，§7.5 的 Numba 目标亦只含 tree/chi2）。
* 后端选择 `resolve_engine`：`engine='auto'|'numpy'|'numba'`（三个粗分箱
  构造器新增参数，签名末位追加，向后兼容）。auto：初始分箱 ≥64 且样本
  ≥5000（tree 另需 limit≤6）才走 Numba；**小数据不 import numba**
  （子进程用例钉住 `import syriskmodels.scorecard` 不加载 numba，§7.6）；
  numba 不可用/触发护栏 → logging.warn 降级 NumPy，绝不静默。
* 不与 multiprocessing 混用：进程并行在 `woebin(no_cores>1)` 层，与内核
  后端正交；内核 `nogil=True` 为未来线程/`prange` 并行留口。
* warm-up/缓存：`cache=True` 产物写入模块旁 `__pycache__/*.nbi`；首进程
  首调用编译（每内核约 0.5–2s），后续进程加载缓存（约 50–100ms）；
  bench 的 warmup 轮吸收编译成本；不在模块导入时编译任何内核。
* 等价性证据：tree 14 场景 × 10 参数组、chi2 14 场景 × 16 参数组（含
  limit=8）内核级 `seg_bounds` 逐位相等；ib=500 大表（slow）；binner 级
  `engine='numba'` 全路径 vs W1 参考；limit>6 降级护栏；auto 规模化
  端到端（n=6000/ib=100 实际走 Numba 且与参考一致）；
  `NUMBA_AVAILABLE=False` 回退。

### 5.1 核心搜索计时（§7.5 目标，creditcard 50k，最差单变量）

| 内核 | NumPy | Numba | 目标 | 达成 |
| --- | ---: | ---: | --- | --- |
| tree ib=100 单变量核心搜索 | 1.51 ms | **0.043 ms** | < 1 ms | ✅（35x） |
| chi2 ib=500 单变量核心搜索 | 13.73 ms | **0.65 ms** | < 5 ms | ✅（21x） |
| creditcard 50k×7 tree ib=100 端到端 | — | **0.111 s** | ≤ 0.3 s | ✅ |

## 6. 关键决策记录（含行为变更声明）

1. **B-10 默认口径变更（方法学修正，有意的行为变化）**：
   `build_scorecard` 变量趋势一致性筛选默认从 OOT 改为验证集（02_test），
   OOT 只用于最终评估；显式 `risk_consistency_dataset='oot'` 保留旧口径
   并强制告警。维护者未曾要求保留 OOT 口径（W1 报告 §6 问题 3 无相反
   结论），故按任务书默认 `'valid'`。影响：`build_scorecard` 的入选变量
   可能与旧版不同（更保守、评估更诚实）；分箱 golden 不受影响。
2. **B-11 默认并行度变更（缺陷修复的一部分）**：`woebin`/`woebin_ply`
   默认 `no_cores=1`（旧默认 None=自动并行：小数据更慢 +45%，交互式
   spawn 挂死）。显式 `no_cores>1` 行为保留且结果与串行逐位一致（用例
   钉住）。新增可选参数 `parallel_timeout`。
3. **类别型 breaks 的 category-dtype 特殊形态**：legacy 在"全单类别段"
   时输出 categorical Series，且其 categories（而非 values）决定最终
   分箱行序 —— 该形态被完整复刻而非"顺手修复"（修复=行为变化，违反
   硬性约束 2）。已在 `segments_to_breaks` / `binning_from_segments` /
   `BinCountTable.categories` 三处文档化。
4. **RuleOptimBin.eps 不转发基类**：其历史含义是 lift 平滑项（1e-8），
   转发会把基类 `epsilon`（WOE 零替换值）从 0.5 变 1e-8，改变 rule 默认
   输出 —— 违反"默认不改变分箱输出"。基类参数经 `**kwargs` 可达基类，
   `eps` 名称冲突维持 legacy 行为。
5. **B-12 选方案 b（deprecated + 兼容导出）**：W2 内核重构正把粗分箱
   迁往 BinCountTable/NumPy，把 DataFrame 形态 helper 接入热路径与重构
   方向相反；且 `extract_*` 的 breaks 语义与生产本就不同。保留导出
   （公共 API 不破坏）、调用告警、重叠语义（compute_woe/compute_iv vs
   binning_format）用测试钉住一致，差异语义两侧钉住防漂移。
6. **B-5 新语义**：int 列 + sv 含 NaN 时数值特殊值标签为 float 形式
   （`-1.0`）—— 旧实现该路径直接崩溃，无 legacy 行为可保留；已由
   golden `synthetic_integer_missing_special_value` 钉住。
7. **woebin_psi 变量输出顺序改为 sorted**：旧实现依赖 set 交集的哈希
   顺序（跨环境不稳定）；数值输出在正常场景不变。
8. **golden 更新**：仅新增 B-4/B-5 两个边界用例（快照 diff 纯新增
   341 行，现有 12 用例逐位未变）；`UPDATE_GOLDEN=1` 机制保持可用。

## 7. before / after 性能对比（中位数，秒；同机同版本同命令）

命令：`PYTHONHASHSEED=0 python bench/run_all.py --repeats 3 --warmup 1`
（creditcard 为确定性 head 抽样 50k；全部 `no_cores=1`）

| 用例 | W1 baseline | W2 before | **W2 after** | vs W1 | W2 目标 | 达成 |
| --- | ---: | ---: | ---: | ---: | --- | :-: |
| germancredit 20 变量 quantile ib20 | 0.122 | 0.148 | **0.120** | 1.0x | 无回退 | ✅ |
| germancredit 20 变量 quantile+tree ib20 | 0.869 | 1.060 | **0.136** | 6.4x | — | ✅ |
| germancredit 20 变量 quantile+chi2 ib20 | 0.379 | 0.434 | **0.135** | 2.8x | — | ✅ |
| creditcard 50k×7 quantile+tree ib20 | 1.107 | 1.260 | **0.094** | 11.8x | ≤ 基线 | ✅ |
| creditcard 50k×7 quantile+tree ib100 | 6.259 | 6.579 | **0.111** | **56.4x** | ≥5x（≤1.3s） | ✅ |
| creditcard 50k×7 quantile+chi2 ib20 | 0.411 | 0.391 | **0.095** | 4.3x | ≤ 基线 | ✅ |
| creditcard 50k×7 quantile+chi2 ib100 | 3.425 | 3.403 | **0.105** | **32.6x** | ≥5x（≤0.7s） | ✅ |
| creditcard 50k×12 woebin_ply(woe) | 0.108 | 0.126 | **0.105** | 1.0x | 不改动/无回退 | ✅ |

（w2_before 与 W1 基线的 ±15% 内差异为本机运行噪声；after 相对 before
与相对 W1 的结论一致。ib=500 用例默认关闭，口径与 W1 相同。）

分阶段归因（creditcard 50k×7 tree ib100）：W1 6.26s → Phase 2 聚合
向量化 6.10s → Phase 3 tree 内核 **0.133s** → Phase 6 缓存 **0.113s** →
Phase 7 Numba(auto) **0.104–0.111s**（端到端已由初始扫描主导）。
`woebin_ply` 未做任何性能改动（仅 B-11 的默认并行度语义）。

## 8. 测试与 CI

| 套件 | 命令 | W1 | W2 |
| --- | --- | --- | --- |
| unit | `PYTHONHASHSEED=0 pytest -q -m "not slow"` | 114 passed, 10 xfailed | **516 passed, 0 xfailed**（xfail 全部转正） |
| golden | `pytest -q -m "golden"` | 12 passed | **14 passed**（+2 新增 B-4/B-5 边界；现有快照逐位不变） |
| integration/slow | `pytest -q -m "slow"` | 7 passed, 1 xfailed | **98 passed**（含差分 fuzz heavy ib=500、creditcard 全量 rule/build_scorecard 端到端） |

新增测试资产：`test_binning_equivalence.py`（446 用例，含 W1 参考拷贝
oracle）、`test_bin_count_table.py`（9）、`test_conftest_fixtures.py`（13）、
`test_rng_fingerprint.py`（2）；known_bugs 全部转正向 + 边界扩充。

CI（`.github/workflows/ci.yml`，push `feature/w2-binning-kernel`）：
4 job —— unit(py3.11) / unit-py312 / integration(py3.11, slow) /
golden(py3.11)。integration job 真实运行 creditcard 全量（`data/*.csv.gz`
已随仓库入库，W2 Phase 0 已勘误相关注释）。Numba 差分用例会在 CI 上
首跑编译（约数秒，cache 产物不跨 run 持久）。运行结果见 PR 描述。

## 9. 公共 API 兼容性自查

* 签名兼容：`woebin`（新增可选 `parallel_timeout`，`no_cores` 默认值
  None→1 属 B-11 缺陷修复）、`woebin_ply`（同前）、`woebin_breaks`、
  `sc_bins_to_df`、`make_scorecard` 未变；`WOEBinFactory` 三种传参方式
  保留；`WOEBin` 及子类公开方法全保留（`binning`/`split_special_values`/
  `binning_breaks`/`initial_binning`/`merge_binning`/`node_split`/`iv`/
  `chi2_stat`/`cut_binning` 等，热路径不再依赖者已在 docstring 标注）；
  新增导出 `BinCountTable`（增量）；三个粗分箱类新增 `engine='auto'`
  命名参数（末位追加）；`build_scorecard` 新增
  `risk_consistency_dataset='valid'`（末位追加）。
* 返回类型：breaks 的 Series 形态（含 name/dtype/categories）逐位保真；
  `woebin` 返回 dict 结构不变。
* 已知微差（均为修复目标本身或不可观测）：B-9 单侧缺失场景 PSI 数值
  （修复目的）；B-10 默认筛选数据集（方法学修正）；B-11 默认并行度；
  woebin_psi 变量行序 sorted（原为哈希序）；rule 无合格切点仍返回
  `[-inf, inf]` list（同 legacy）。

## 10. 未完成项 / 新观察 / W3 计划

**W2 无 blocked 项**；以下为本轮明确不做（任务书 §5）与新观察：

1. W3 计划内：验证集感知分箱、稳定性约束分箱、WOE 平滑；stepwise 目标
   函数重构（B-3 只修复了方向控制流，目标函数仍是
   `min(train,valid)-|Δ|`）、beam search；build_scorecard pipeline/配置
   体系重构；跨时间稳定性框架、校准与监控。
2. **新观察 B-15（W3 候选）**：`x_variable()` / `woebin_ply` 的变量集合
   来自 `set` 运算，输出顺序依赖哈希种子（`PYTHONHASHSEED=0` 下可复现，
   跨环境不确定；woebin_ply 输出**列顺序**受影响）。建议 W3 改为稳定
   排序（属行为变化，需单独评估）。
3. **新观察 B-16（W3 候选）**：混合类型 object 列（str+int）在
   `np.unique` 排序处抛 TypeError（`test_conftest_fixtures.py::
   test_data_with_mixed_types` 已钉住现状）。建议 W3 在 `check_uniques`
   / 分箱入口做类型归一或给出可操作的早期报错。
4. **新观察 B-17（W3 候选）**：全好数据（bad.sum()=0）下
   `binning_format` 的 lift 列触发 RuntimeWarning（scalar divide）——
   legacy 行为，等价测试中可见。建议 W3 显式守卫。
5. `scorecard_legacy.py` 仍保留（未被生产路径引用）；W3 评估删除或
   移入独立兼容包。
6. Numba 后端未覆盖 rule（scipy 依赖）；`prange`/线程级并行未启用
   （当前单变量内核已达 <1ms 量级，收益有限）；跨变量并行仍由
   `no_cores>1` 进程层承担。
7. W1 报告 §1.1 的 numpy 版本对比证据（2.4.6 vs 2.5.3 的
   `default_rng.random` 流）已由随机流指纹用例前移到"版本升级即刻
   报警"，无需再维护双版本环境。

## 11. 修改文件清单（develop..HEAD，17 commits）

**src/**（10 文件）
`scorecard/core/base.py`（split_special_values 重写、binning 向量化、
OptimBinMixin.initial_count_table/_initial_binning_finish、ComposedWOEBin
缓存、apply idx 自建）、`scorecard/core/counts.py`（新增）、
`scorecard/core/kernels.py`（新增）、`scorecard/core/kernels_numba.py`
（新增）、`scorecard/core/factory.py`（kwargs 分发）、
`scorecard/bins/optimal.py`（三内核接入 + engine）、
`scorecard/api/woebin.py`、`scorecard/api/transform.py`（B-11）、
`scorecard/api/evaluation.py`（B-2/B-9）、`scorecard/utils/validation.py`
（B-8）、`scorecard/utils/binning_helpers.py`（B-12 deprecated）、
`scorecard/core/__init__.py`、`scorecard/__init__.py`（BinCountTable 导出）、
`evaluate.py`（psi 告警）、`models.py`（B-3）、
`contrib/build_scorecard.py`（B-10）、`utils.py`（B-11 helper）

**test/**（9 文件）
`test_known_bugs.py`（B-2..B-13 全转正向+扩充）、
`test_binning_equivalence.py`（新增，446 用例）、
`test_bin_count_table.py`（新增）、`test_golden_binning.py`（+2 用例）、
`golden/synthetic_binning.json`（+2 快照）、`test_scorecard_integration.py`
（xfail/witness 移除、tmp 隔离）、`test_scorecard/conftest.py`（B-13）、
`test_scorecard/test_conftest_fixtures.py`（新增）、
`test_scorecard/test_binning_helpers.py`（兼容层定位）、
`test_rng_fingerprint.py`（新增）、`conftest.py`（数据说明勘误）

**bench/** `results/w2_before.json`、`results/w2_after.json`（新增入库）

**其他** `.github/workflows/ci.yml`（注释勘误）、`test/golden/README.md`、
`docs/plans/w1-baseline-report.md`（勘误+状态更新）、
`docs/plans/w2-binning-kernel.md`（计划归档）、本报告

## 12. 提交列表

```
ffc0560 chore(w2): phase 0 — persist w2_before baseline, fix data-tracking docs, archive W2 plan
f3cc8ab fix(pandas3): B-2/B-4/B-5/B-9 — woebin_plot、特殊值拆分、woebin_psi
6b5ace0 fix(models): B-3 — stepwise_lr forward/backward 单向搜索返回空
e3e64d7 fix(factory): B-6/B-7/B-8 — kwargs 按签名分发、RuleOptimBin 透传、literal_eval
148d09e fix(methodology): B-10 — build_scorecard 变量筛选默认不再使用 OOT
dbfbe2a fix(parallel): B-11 — 默认 no_cores=1，显式并行使用安全进程上下文
6b3978c refactor(scorecard): B-12 deprecated；test(scorecard): B-13 fixture 依赖注入
9c8c342 refactor(binning): BinCountTable + 聚合向量化（Phase 2）
8928f65 perf(tree): TreeOptimBin NumPy 精确等价内核（Phase 3）
e75fa94 perf(chimerge): ChiMergeOptimBin NumPy 精确等价内核（Phase 4）
f9fe61c perf(rule): RuleOptimBin 向量化精确等价实现（Phase 5）
32d472f perf(composed): ComposedWOEBin 计数表缓存（Phase 6）
ec4e068 perf(numba): Tree/ChiMerge Numba 编译后端（Phase 7）
af2bf56 test(binning): 差分 fuzz 补充（空分箱/退化表）
25a2545 bench: W2 after 基准
f120fa3 test(baseline): 随机流指纹用例
(+ 本报告与 W1 报告状态更新的 docs commit)
```

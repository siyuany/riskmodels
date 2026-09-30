# W1 基线建设报告：可复现的测试与性能基线

> 分支：`feature/w1-test-baseline` → 已按 git flow `--no-ff` 合并入 `develop`
> 基线提交：`2aeaa32`（W1 起点）｜ 合并提交：`263e8bd` ｜ CI：见 2.5
> 范围：测试可运行性、CI、golden 快照、性能基线、Bug 清单
> **本阶段不修改任何分箱 / 模型 / 评分卡核心行为，不碰 `src/`**

---

## 1. 环境与复现命令

### 1.1 开发机环境（基线产出环境）

| 项目 | 值 |
| --- | --- |
| 平台 | macOS 26.6.2 / arm64（Apple Silicon） |
| CPU | 8 核 |
| Python | CPython 3.12.13（`~/.venvs/jep`） |
| pandas | 3.0.6 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| statsmodels | 0.15.0 |
| pyarrow | 25.0.1 |
| pytest | 9.1.1（+ pytest-cov 7.1.0） |
| multiprocessing start method | **spawn** |

> ⚠️ pandas / numpy 版本与需求描述的 “pandas 3.0 / numpy 2.4” 略有出入（本机为
> pandas 3.0.6 / numpy 2.5.3）。基线数值与版本强相关，跨版本对比前请先核对
> `bench/results/*.json` 的 `environment` 字段。

### 1.2 复现命令

```bash
# 安装（含 dev 依赖：pytest / pytest-cov / openpyxl）
~/.venvs/jep/bin/python -m pip install -e '.[dev]'

# 单元 / 快速用例（不依赖大数据集）
PYTHONHASHSEED=0 ~/.venvs/jep/bin/python -m pytest -q -m "not slow"

# 集成用例（需要 data/*.csv.gz）
PYTHONHASHSEED=0 ~/.venvs/jep/bin/python -m pytest -q -m "slow"

# golden 快照
PYTHONHASHSEED=0 ~/.venvs/jep/bin/python -m pytest -q -m "golden"
UPDATE_GOLDEN=1 PYTHONHASHSEED=0 ~/.venvs/jep/bin/python -m pytest test/test_golden_binning.py   # 重生成

# 性能基线
PYTHONHASHSEED=0 ~/.venvs/jep/bin/python bench/run_all.py --repeats 3 --warmup 1
PYTHONHASHSEED=0 ~/.venvs/jep/bin/python bench/run_all.py --quick          # 冒烟
PYTHONHASHSEED=0 ~/.venvs/jep/bin/python bench/run_all.py --include-ib500  # 含重用例
```

**数据前提**：`data/germancredit.csv.gz`（17 KB）、`data/creditcard.csv.gz`（65 MB）
不入版本库。缺失时相关用例 `skip` 并在报告中说明原因，不会以 `FileNotFoundError` 报错。

---

## 2. 测试 / CI 状态

### 2.1 改动前后对比

| 指标 | 改动前 | 改动后 |
| --- | --- | --- |
| `pytest test/` 收集 | 101 | 132（含 golden 12、known-bugs 14、datasets 契约 3、detect +1） |
| `pytest test/`（全量，含 slow） | **7 failed**, 94 passed（5.90s） | **121 passed, 11 xfailed, 0 failed**（48.5s） |
| `-m "not slow"`（unit / CI unit job） | 不适用（无 marker） | **114 passed, 10 xfailed, 8 deselected**（17.0s） |
| `-m "slow"`（CI integration job） | 不适用 | **7 passed, 1 xfailed**（42.5s，含 creditcard 全量） |
| `-m "golden"` | 无 | **12 passed**（4.2s） |
| 无数据集时（模拟干净克隆） | 7 failed | `-m "not slow"`：95 passed, 19 skipped；`-m "slow"`：8 skipped（退出码 0）；golden：7 passed, 5 skipped |

> 关键点：无数据集时**合成数据 golden 用例仍然执行**（7 passed），只有
> 依赖 germancredit 的 5 个 golden 测试 skip —— 干净克隆不会丢掉全部护栏。

### 2.2 改动前 7 个失败的根因

全部集中在 `test/test_scorecard_integration.py`，**同一个根因**：

```
subprocess.CalledProcessError → FileNotFoundError:
  '/Users/.../riskmodels/test/germancredit.csv'
```

`_load_csvs_in_subprocess()` 读取的是 `test/germancredit.csv` 与
`test/creditcard.csv`，而版本库里**从来没有这两个文件**：只有被
`.gitignore:19 test/*.csv.gz` 忽略的**软链接** `test/*.csv.gz → data/*.csv.gz`。
即"本地能过、干净克隆必挂"，属环境耦合缺陷（见 B-1）。

### 2.3 marker 设计

| marker | 含义 | CI 归属 |
| --- | --- | --- |
| `slow` | 依赖大数据集（creditcard 全量）或长耗时 | `integration` job |
| `integration` | 端到端流程（真实数据 / 完整评分卡流水线） | 随 slow/unit 自然分布 |
| `golden` | 与 `test/golden/*.json` 严格比对 | `unit` + `golden` job |

* `pyproject.toml` 的 `addopts` **刻意不加默认 `-m`**，避免"测试被静默隐藏"。
* `filterwarnings` 逐条 `default::...`，**不使用 `ignore`**，保证 warning diff 可见。
* 配置集中在 `pyproject.toml`，未新增被 `.gitignore` 忽略的 `pytest.ini`。

### 2.4 CI

`.github/workflows/ci.yml`，4 个 job（YAML 已用 PyYAML 严格解析验证）：

| job | Python | 命令 | 说明 |
| --- | --- | --- | --- |
| `unit` | 3.11 | `pytest -q -m "not slow"` | pandas 3.x，含合成 golden + known-bugs |
| `unit-py312` | 3.12 | `pytest -q -m "not slow"` | 版本矩阵 |
| `integration` | 3.11 | `pytest -q -m "slow"` | 数据缺失时 skip 并打印原因，job 仍通过 |
| `golden` | 3.11 | `pytest -q -m "golden"` | 基线漂移一眼可见 |

* 全局 `PYTHONHASHSEED=0`、`MPLBACKEND=Agg`、`MPLCONFIGDIR=/tmp/mplconfig`。
* `actions/setup-python@v5` + `cache: pip`，`cache-dependency-path: pyproject.toml`。
* 不使用 multiprocessing；测试内所有分箱/转换调用显式 `no_cores=1`。
* 可选：配置仓库变量 `RISKMODELS_DATA_URL` 即自动拉取数据集。
* 注：`data/*.csv.gz` 已在版本库中跟踪，因此 `integration` / `golden` job 默认
  有真实数据，不会空跑。

### 2.5 首次 CI 运行记录（run 36674137396）

`develop` 推送后 CI 首次运行，**这正是 CI 存在的意义** —— 本地 macOS 全绿，
ubuntu x86_64 上暴露了一处只有真实 runner 才能发现的问题：

| job | 结果 | 说明 |
| --- | --- | --- |
| `integration` (py3.11, slow) | ✅ 通过 | creditcard 全量分箱 + 评分卡流水线 |
| `unit` (py3.11) | ❌ 2 failed | 均为 `test_golden_binning.py` 的合成数据用例 |
| `unit-py312` | ❌ 2 failed | 同上 |
| `golden` (py3.11) | ❌ 2 failed | 同上 |

失败**全部**指向 `synthetic_quantile_chi2_ib20_limit5`，根因是合成数据生成用了
RNG 内核、跨架构不同位（详见 B-14）。修复后 CI 转绿，见下节。

### 2.6 修复后的 CI 状态

| job | 结果 |
| --- | --- |
| `unit` (py3.11) | ✅ 114 passed, 10 xfailed |
| `unit-py312` | ✅ 同上 |
| `integration` (py3.11, slow) | ✅ 7 passed, 1 xfailed |
| `golden` (py3.11) | ✅ 12 passed |

**验证手段的教训**：在本地"用与 CI 逐字相同的命令跑通"并不能替代真实 runner ——
它能覆盖命令与版本，但覆盖不到 CPU 架构、libm、RNG 内核差异。凡是把**运行时
生成的数据**写进快照，就必须假设它可能在别的平台上不同位。

---

## 3. 性能基线

基线文件：[`bench/results/baseline_20260930.json`](../../bench/results/baseline_20260930.json)
（`bench/results/*` 已 gitignore，需要入库时 `git add -f`）

**方法**：1 次 warmup（不计入）+ 3 次计时取**中位数**；全部 `no_cores=1`；
creditcard 用**确定性 head 抽样**（不做随机抽样，保证可复现）。

| 用例 | 行数 | 变量 | 方法 | initial_bins | 中位数 (s) | 每变量 (s) |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| germancredit 全变量 | 1 000 | 20 | quantile | 20 | **0.122** | 0.0061 |
| germancredit 全变量 | 1 000 | 20 | quantile+tree | 20 | **0.869** | 0.0435 |
| germancredit 全变量 | 1 000 | 20 | quantile+chi2 | 20 | **0.379** | 0.0189 |
| creditcard 抽样 | 50 000 | 7 | quantile+tree | 20 | **1.107** | 0.1582 |
| creditcard 抽样 | 50 000 | 7 | quantile+tree | 100 | **6.259** | 0.8942 |
| creditcard 抽样 | 50 000 | 7 | quantile+chi2 | 20 | **0.411** | 0.0587 |
| creditcard 抽样 | 50 000 | 7 | quantile+chi2 | 100 | **3.425** | 0.4893 |
| woebin_ply（woe） | 50 000 | 12 | — | — | **0.108** | 0.0090 |

（单位：秒。`initial_bins=500` 用例默认关闭：单次约 40 s，全流程 > 5 分钟，
需要时用 `--include-ib500` 打开。）

### 3.1 观察

1. **`initial_bins` 是主成本驱动**：50k×7 变量下，tree 从 20→100 箱耗时 ×5.7
   （1.107→6.259 s），chi2 ×8.3（0.411→3.425 s）。细分箱数量近似线性放大
   初始箱数，粗分箱的合并循环（`ChiMerge` 每轮重算全部 chi2；
   `Tree` 每轮对每个候选切点重建 binning）则是超线性来源。
2. **tree 比 chi2 贵 2.7×**（ib=20：1.107 vs 0.411 s；ib=100：6.259 vs 3.425 s）。
   `TreeOptimBin.woebin` 的内层循环对**每个内部节点**调用
   `merge_binning`（`groupby + agg + lambda` 一次全量重算）→ O(n_bins²) 量级的
   pandas 开销，是最值得优化的热点。
3. **`woebin_ply` 极便宜**：50k×12 变量 0.108 s（≈ 0.009 s/变量），
   说明转换路径不是瓶颈，W2 优化优先级应放在粗分箱。
4. **germancredit 规模太小，不足以暴露性能问题**（20 变量全量 < 1 s），
   后续性能回归必须用 creditcard 抽样用例。

---

## 4. Bug / 风险清单

> 状态：**W1 只记录不修复**（B-1、B-14 例外：均为 **W1 已修**，前者是"测试不可运行"，
> 后者是"golden 不可移植"，两者都是 W1 验收标准本身要求修好的）。每条都有可执行回归用例
> （`test/test_known_bugs.py`，`xfail(strict=True)`：修复后用例变 XPASS → 失败，
> 强制清理记录）。
> 优先级：**P0** = 阻断主流程 / 结果错误；**P1** = 功能缺失 / 兼容性；
> **P2** = 工程质量 / 安全加固。

### B-1 测试数据路径与软链接（P1）

| 项 | 内容 |
| --- | --- |
| 现象 | 干净克隆后 `pytest test/` → 7 failed，全部 `FileNotFoundError: test/germancredit.csv` |
| 复现 | `git clone && pytest test/test_scorecard_integration.py -q` |
| 位置 | 原 `test/test_scorecard_integration.py:27-32` |
| 疑似原因 | 测试硬编码 `test/*.csv`，而仓库只有被 gitignore 的 `test/*.csv.gz` 软链接指向 `data/` |
| 处置 | **W1 已修**：改用 `syriskmodels.datasets.load_germancredit()/load_creditcard()`，无数据时 `skip` 并说明原因 |
| 目标阶段 | 已完成（W1） |

### B-2 pandas 3 下 `woebin_plot` 抛 `KeyError: 'variable'`（**P0**）

| 项 | 内容 |
| --- | --- |
| 现象 | `woebin_plot(bins)` → `KeyError: 'variable'`；`build_scorecard` 尾部绘图同样崩 |
| 复现 | `pytest test/test_known_bugs.py::test_b2_woebin_plot_works -q` |
| 位置 | `src/syriskmodels/scorecard/api/evaluation.py:181-184` |
| 疑似原因 | `bins_df.groupby('variable', observed=False).apply(_gb_distr)`：pandas 3 下 DataFrame 返回值的 `apply` **不再把分组键保留为列**（分组键只进 index）。实测 `apply(lambda x: x.assign(...))` 返回列不含 `variable` |
| 影响 | `woebin_plot` 不可用；`build_scorecard` 主体跑通后仍整体失败（集成用例 `test_build_scorecard_pipeline` 因此 `xfail`） |
| 建议修法 | `apply(...)` 后 `reset_index(level=0)`，或改用 `transform` / 显式 `merge` 回分组键 |
| 目标阶段 | **W2 第一优先**（当前唯一阻断端到端流程的缺陷） |

### B-3 `stepwise_lr(direction='forward'|'backward')` 返回 `(None, [])`（**P0**）

| 项 | 内容 |
| --- | --- |
| 现象 | `forward` / `backward` 均返回 `(None, [])`；`bidirectional` 正常 |
| 复现 | `pytest test/test_known_bugs.py::test_b3_stepwise_lr_single_direction_selects_features -q` |
| 位置 | `src/syriskmodels/models.py:130-159`（择优块在 145-152 行） |
| 疑似原因 | 择优并更新 `selected_features` / `best_metrics` / `improved` 的整段代码被放在 `if direction in ['backward','bidirectional']` 分支内；`forward` 收集完 `perf_records` 后 `improved` 恒为 `False`，首轮 `break` |
| 影响 | 单向逐步回归完全不可用（静默返回空结果，不报错）；`build_scorecard` 默认 `direction='bidirectional'` 未受影响 |
| 建议修法 | 把择优块移出方向判断，方向仅控制"产生哪些候选" |
| 目标阶段 | W2 |

### B-4 数值列 + 显式数值型 `special_values` 在 pandas 3 下报 ValueError（**P0**）

| 项 | 内容 |
| --- | --- |
| 现象 | `ValueError: You are trying to merge on float64 and object columns for key 'value'` |
| 复现 | `pytest test/test_known_bugs.py::test_b4_numeric_special_values_binning -q`；最小复现：`woebin(df, y='y', x=['num_b'], special_values={'num_b': ['-999','missing']})` |
| 位置 | `src/syriskmodels/scorecard/core/base.py:113-118` |
| 疑似原因 | `dtm.fillna("missing")` 使 object 化列 dtype 变 object，而 `sv_df['value']` 仍是 float64（或反向）；pandas 3 不再隐式放宽 object↔float 的 merge 键类型（pandas 2 会隐式转换） |
| 影响 | `special_values=[-999, -1, ...]` 这一 README 主推用法在 pandas 3 下**整体不可用** |
| 建议修法 | merge 前统一 `dtm_merge['value'] = dtm_merge['value'].astype(object)`（或双方 `.astype(str)`），保持 pandas 2 语义 |
| 目标阶段 | **W2 第一优先**（与 B-2 并列） |

### B-5 整型列 + `'missing'` 特殊值在 pandas 3 下报 ValueError（**P0**）

| 项 | 内容 |
| --- | --- |
| 现象 | `ValueError: cannot convert float NaN to integer` |
| 复现 | `pytest test/test_known_bugs.py::test_b5_integer_column_special_values_binning -q` |
| 位置 | `src/syriskmodels/scorecard/core/base.py:104-105` |
| 疑似原因 | `sv_df['value'].astype(dtm['value'].dtypes)`：`split_vec_to_df` 对 `'missing'` 产出 `None`/NaN，对 int64 列执行 `astype(int64)` 直接抛错（pandas 3 不再静默截断 NaN→int） |
| 影响 | 整型变量（如 `number.of.existing.credits.at.this.bank`）无法使用 `missing` 特殊值 |
| 建议修法 | 仅在无 NaN 时 `astype`，否则保持 object/float dtype |
| 目标阶段 | W2（与 B-4 同一处代码，建议一并修） |

### B-6 `WOEBinFactory` 把全局 kwargs 透传给所有 binner（P1）

| 项 | 内容 |
| --- | --- |
| 现象 | `methods=['quantile','rule']` + `initial_bins=20` → `TypeError: RuleOptimBin.__init__() got an unexpected keyword argument 'initial_bins'` |
| 复现 | `pytest test/test_known_bugs.py::test_b6_quantile_rule_with_initial_bins -q` |
| 位置 | `src/syriskmodels/scorecard/core/factory.py:122`（`get_binner(bin_cls, **kwargs)` 对列表中所有类别透传同一份 kwargs） |
| 疑似原因 | 缺少"按分箱器类型过滤 kwargs"或"显式声明各自参数"的机制；README 描述的是"kwargs 传给各分箱方法"，实现却是无差别透传 |
| 影响 | 组合 `rule` 时必须改写为类实例列表 |
| 目标阶段 | W2 |

### B-7 `RuleOptimBin.__init__` 不接受 / 不传递 `**kwargs`（P1）

| 项 | 内容 |
| --- | --- |
| 现象 | `RuleOptimBin(initial_bins=20)` → `TypeError` |
| 复现 | `pytest test/test_known_bugs.py::test_b7_rule_optim_bin_accepts_kwargs -q` |
| 位置 | `src/syriskmodels/scorecard/bins/optimal.py:285-297` |
| 疑似原因 | 未声明 `**kwargs`，且 `super().__init__()` **未传参**：基类 `eps` 默认 0.5 被丢弃，自身写死 `eps=1e-8`，与其它粗分箱类行为不一致 |
| 影响 | 无法调 `eps`；与 B-6 共同导致 `rule` 无法通过字符串 methods 组合 |
| 目标阶段 | W2 |

### B-8 `check_breaks_list` 使用 `eval`（P2，安全）

| 项 | 内容 |
| --- | --- |
| 现象 | 字符串入参被 `eval`，任意表达式都会执行 |
| 复现 | `pytest test/test_known_bugs.py::test_b8_check_breaks_list_rejects_non_literal_expressions -q` |
| 位置 | `src/syriskmodels/scorecard/utils/validation.py:222-231` |
| 疑似原因 | 历史实现用 `eval` 支持"从配置文件读字符串形式的字典" |
| 影响 | `breaks_list` 常来自外部配置，构成代码注入面 |
| 建议修法 | 改 `ast.literal_eval`，保留"必须是字典"校验 |
| 目标阶段 | W2（低风险、低工作量，建议顺手做） |

### B-9 `woebin_psi` 单侧缺失分箱时静默放大 PSI（**P0**，结果错误）

| 项 | 内容 |
| --- | --- |
| 现象 | 比较集缺少某分箱时，该变量**每一行**都得到同一个偏大的 PSI（实测 4.897；正常应 < 1），且无任何告警 |
| 复现 | `pytest test/test_known_bugs.py::test_b9_woebin_psi_one_sided_bins_are_not_silently_inflated -q` |
| 位置 | `src/syriskmodels/scorecard/api/evaluation.py:48-70`（`pivot_table` 后 `tmp.columns = ['base','cmp']` 未区分缺失侧 → NaN），叠加 `src/syriskmodels/evaluate.py:171-193`（`psi` 把 NaN 当 0 再重新归一化） |
| 疑似原因 | 缺失分布未按 0 显式补齐，`psi()` 的 NaN→0 又被"重新归一化"二次放大 |
| 影响 | 变量稳定性结论可能完全错误；`build_scorecard` 的 PSI 分析直接受影响 |
| 建议修法 | `pivot_table` 后 `reindex` 到 bins 全集并 `fillna(0)`；或在 `psi()` 中显式校验并抛错而非静默填 0 |
| 目标阶段 | **W2 第一优先**（正确性问题） |

### B-10 `build_scorecard` 使用 OOT 做变量筛选（P1，方法学泄漏）

| 项 | 内容 |
| --- | --- |
| 现象 | 变量筛选链路：训练集分箱 → 训练集 IV/单调性初筛 → **`risk_trends_consistency(oot_df, ...)`** → `stepwise_lr` 候选池 |
| 复现 | `pytest test/test_known_bugs.py::test_b10_variable_selection_uses_oot_and_leaks_into_woe_selection -q`（AST 静态断言） |
| 位置 | `src/syriskmodels/contrib/build_scorecard.py:139-141` |
| 疑似原因 | OOT 被当作"跨期一致性验证"用于特征选择，而 OOT 本应只在最终评估阶段使用 |
| 影响 | OOT 不再是干净样本外集合，模型效果与 PSI 评估偏乐观 |
| 建议修法 | 趋势一致性改在训练/验证集上做；若业务上确需 OOT，须在文档与报告中显式声明 |
| 目标阶段 | W2（需业务确认，见第 6 节问题 3） |

### B-11 `mp.Pool` 默认并行不安全，且小数据更慢（P1）

| 项 | 内容 |
| --- | --- |
| 现象 1（**spawn 挂死**） | `woebin(..., no_cores=None)` 在 Jupyter / REPL / `python - <<EOF` / 管道 stdin 下：worker 报 `FileNotFoundError: .../<stdin>`，主进程 **无限等待不返回**（实测 240 s 未退出，需 kill） |
| 现象 2（小数据更慢） | germancredit 1000 行 × 20 变量 × quantile+tree：`no_cores=1` **0.85 s** vs 自动 4 进程 **1.23 s**（+45%） |
| 现象 3（大数据有效但非必需） | creditcard 50k × 7 变量 ib=100：`no_cores=1` 5.80 s vs 2 进程 4.18 s（结果完全一致，已验证） |
| 复现 | `printf '...\n' \| PYTHONHASHSEED=0 python -`（现象 1）；`python /tmp/probe.py`（现象 2/3） |
| 位置 | `src/syriskmodels/scorecard/api/woebin.py:119-125,151-156`；`api/transform.py:68-71,85-90` |
| 疑似原因 | 自动核数计算在数据很小时也返回 >1；`mp.Pool` 用模块级函数虽可 pickle，但 macOS/Windows 的 spawn 需要可导入的 `__main__`，交互式环境无法满足；无 `mp.get_context('fork')` 或 `__main__` 保护、无超时 |
| 影响 | 交互式使用（大量用户的真实场景）会挂死；测试环境不确定性 |
| W1 处置 | 所有测试与 benchmark **一律 `no_cores=1`**；`build_scorecard` 集成用例通过 `monkeypatch` 固定并行度 |
| 目标阶段 | W2「性能重构」阶段一并设计（阈值化 + fork 上下文 + 交互式环境探测） |

### B-12 `binning_helpers` 与生产代码脱节（P2，工程风险）

| 项 | 内容 |
| --- | --- |
| 现象 | `scorecard/utils/binning_helpers.py`（396 行、6 个公开函数）**只被 `scorecard/__init__.py` 再导出**，`core/`、`bins/`、`api/` 无任何引用 |
| 复现 | `pytest test/test_known_bugs.py::test_b12 -q`；`grep -rn binning_helpers src/syriskmodels/scorecard/{core,bins,api}` 无结果 |
| 位置 | `src/syriskmodels/scorecard/utils/binning_helpers.py` vs 实际生产路径 `api/transform.woebin_breaks` + `core/base.WOEBin.binning_breaks` |
| 疑似原因 | 重构遗留：新实现绕开了 helper，但 helper 与 `test/test_scorecard/test_binning_helpers.py`（30+ 断言）被保留 |
| 影响 | ① 测试覆盖的是**死代码**，造成"分箱逻辑已被充分覆盖"的错觉；② 两套 `breaks` 语义不同（helper 返回右边界含 `inf`，生产返回分箱名字符串），后续重构易改错对象 |
| 目标阶段 | W2 决策：接入生产路径，或标注 deprecated 并删除（连同其测试） |

### B-13 `test/test_scorecard/conftest.py` fixture 定义缺陷（P2）

| 项 | 内容 |
| --- | --- |
| 现象 | `data_with_constant_var` 等 5 个 fixture 直接调用 `clean_data()`，而 `clean_data` 是 fixture 函数，普通调用返回 `_FixtureFunction` 对象 → 一旦被使用必然 `AttributeError` |
| 位置 | `test/test_scorecard/conftest.py:24-45`（`df['constant'] = 999` 等） |
| 现状 | 当前无测试消费这些 fixture，因此**未暴露**；属潜伏缺陷 |
| 建议 | 改为 `def data_with_constant_var(clean_data): ...` 依赖注入；W1 未改动（不在允许改动范围的必要项内，且要避免"顺手改测试语义"） |
| 目标阶段 | W2 |

### B-14 合成测试数据的跨平台可移植性（P1，**已在 W1 修复**）

| 项 | 内容 |
| --- | --- |
| 现象 | **首次 CI 运行**（push `develop` 后）：`synthetic_quantile_chi2_ib20_limit5` 在 `unit`(py3.11)、`unit-py312`、`golden` 三个 job 上**全数失败**；本地 macOS 全绿。失败形态是分箱 diff（`num_signal` 的 breaks 与 count 不同），根因被埋在一大坨 JSON 里 |
| 复现 | `gh run view 36674137396 --log-failed`；本地无法复现（macOS arm64 与 ubuntu x86_64 数据不同位） |
| 位置 | `test/test_golden_binning.py::_synthetic_frame()`（第一版用 `np.random.default_rng`） |
| 疑似原因 | 相同 seed 下 `Generator.standard_normal` / `binomial` / `lognormal` 在不同 CPU 架构走不同 SIMD 内核，产生**不同位序列**；`num_signal` 逐位不同 → tree/chi2 贪心切分点分歧（`num_normal`/`num_skewed`/`num_discrete`/`cat_ok` 四项一致，只有依赖 RNG 的 `num_signal` 分歧，可反推为数据而非算法问题） |
| 影响 | golden 快照变成"只在生成它的平台上有效"，CI 永远红 |
| 处置 | **W1 已修**：生成器改为**纯整数算术 + 1/2^k 缩放**（IEEE-754 精确运算，不调用 RNG 内核、不用超越函数），并在快照中加入 `data_fingerprint`（每列 CRC32）作为**前置校验** —— 数据不一致时立刻指出具体列，不再让人从分箱 diff 里猜根因 |
| 教训 | "固定 seed" ≠ "跨平台可复现"。凡是进快照的测试输入，生成过程必须只用精确运算；否则应把数据作为**固定的数据文件**入库，而不是运行时生成 |
| 目标阶段 | 已完成（W1） |

---

## 5. W1 交付物清单

| 类别 | 文件 | 说明 |
| --- | --- | --- |
| 配置 | `pyproject.toml` | `[project.optional-dependencies].dev`、`[tool.pytest.ini_options]`（testpaths/markers/filterwarnings） |
| CI | `.github/workflows/ci.yml` | 4 job，`PYTHONHASHSEED=0`，pip 缓存，无 multiprocessing |
| 测试 | `test/conftest.py` | 数据可用性 fixture/helper、哈希种子校验、header 报告 |
| 测试 | `test/__init__.py` | 包标记，保证 `from test.conftest import ...` 解析 |
| 测试 | `test/test_scorecard_integration.py` | 修数据路径、去 pickle hack、模块级缓存、`no_cores=1`、slow 标记 |
| 测试 | `test/test_datasets.py` | 数据缺失 skip + `FileNotFoundError` 契约用例 |
| 测试 | `test/test_detect.py` | 固定种子 + 确定性护栏 |
| 测试 | `test/test_golden_binning.py` + `test/golden/*.json` + `test/golden/README.md` | 12 个 golden 用例（10 用例 + 2 份整体比对） |
| 测试 | `test/test_known_bugs.py` | 14 个用例（10 xfail + 4 护栏）覆盖 B-1..B-12 |
| 基准 | `bench/common.py`、`bench/bench_binning.py`、`bench/bench_ply.py`、`bench/run_all.py` | 公共工具 + 两个基准 + 一键脚本 |
| 基准产物 | `bench/results/baseline_20260930.json` | 8 个用例的中位数基线（本地，已 gitignore） |
| 忽略规则 | `.gitignore` | `bench/results/*`（保留 `.gitkeep`） |
| 报告 | `docs/plans/w1-baseline-report.md` | 本文档（`docs/` 被 gitignore，本地留存） |

**`src/` 改动：无。** W1 未发现"测试不可运行所必需的极小修复"，
所有缺陷一律记录 + `xfail` 回归用例。

---

## 6. W2 建议

### 6.1 建议优先级

| 优先级 | 事项 | 对应 Bug | 理由 |
| --- | --- | --- | --- |
| **P0-1** | 修 pandas 3 兼容：数值型 `special_values` / 整型 `missing` | B-4, B-5 | README 主推用法整体不可用；同一处代码，改动小 |
| **P0-2** | 修 `woebin_plot` 分组列丢失 | B-2 | 唯一阻断 `build_scorecard` 端到端的缺陷；集成用例因此只能 `xfail` |
| **P0-3** | 修 `woebin_psi` 单侧缺失静默放大 | B-9 | 结果正确性问题，直接影响变量筛选结论 |
| **P0-4** | 修 `stepwise_lr` 单向方向 | B-3 | 静默返回空结果，最易误用 |
| **P1-1** | `WOEBinFactory` kwargs 按类型分发 + `RuleOptimBin` 接 kwargs | B-6, B-7 | 解开 `rule` 组合限制 |
| **P1-2** | 并行策略重设计（阈值 + fork 上下文 + 交互式探测） | B-11 | 交互式挂死是真实事故面 |
| **P1-3** | `build_scorecard` OOT 泄漏 | B-10 | 需业务确认口径 |
| **P2-1** | `eval` → `ast.literal_eval` | B-8 | 低风险顺手项 |
| **P2-2** | `binning_helpers` 去留决策 | B-12 | 清理"死代码覆盖"错觉 |
| **P2-3** | 测试 fixture 修复 | B-13 | 清理潜伏缺陷 |

### 6.2 性能重构护栏（W2 起生效）

1. **改前先跑基线**：`bench/run_all.py --repeats 3`，与
   `baseline_20260930.json` 对比中位数；同机同版本下 >20% 回退需说明。
2. **改后必须全绿**：`pytest -m "not slow"` + `pytest -m "golden"` +
   `pytest -m "slow"`（数据存在时）。golden 快照可精确到"哪个变量哪个箱的
   woe/breaks 变了"。
3. **热点明确**：优先动 `TreeOptimBin.woebin` 的
   `merge_binning`（O(n_bins²) 的重建循环）与 `ChiMergeOptimBin.chi2_stat`
   的逐轮全量重算；`woebin_ply` 无需优化（0.009 s/变量）。
4. **保持 `no_cores=1` 作为基准口径**：并行化是独立议题（B-11），
   不要在算法重构里混入并行改动，否则基线不可比。

### 6.3 基线维护

* golden 快照：**有意**的行为变更才可 `UPDATE_GOLDEN=1`，并在 PR 描述中
  贴出 `--diff` 摘要。
* 基准 JSON：机器/依赖升级后请重新生成基线并**更新本报告的表格**；
  入库时用 `git add -f bench/results/...`。
* CI 里 `golden` job 独立存在，目的就是让基线漂移在 review 时一眼可见。

---

## 7. 已知限制

1. `bench` 的 creditcard 用例用 **head 抽样**（非随机），与"随机 50k"结果不可直接比较；
   选择 head 是为了可复现（无 RNG 依赖）。
2. `initial_bins=500` 用例默认关闭（单次 ~40 s）；如需完整矩阵请用 `--include-ib500`。
3. `test_build_scorecard_pipeline` 使用 `xfail(strict=False)`：因 B-2 无法全绿。
   为避免"任意环节失败都被 xfail 掩盖"，用例内加了**见证断言**
   （`woebin_psi` 返回非空 + Excel 产物非空）；修复 B-2 后请移除 xfail。
4. 集成用例的 creditcard 部分在无数据环境会 skip；`pytest -m "slow"` 仍返回 0（通过），
   报告中会打印 `creditcard.csv.gz=MISSING`。

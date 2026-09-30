# -*- encoding: utf-8 -*-
"""已知 Bug / 风险回归用例（W1 只记录不修复）。

本文件是 ``docs/plans/w1-baseline-report.md`` 中 Bug 清单的**可执行副本**：
每个用例对应清单里的一条，断言的是"当前（有缺陷的）行为"。

约定
----
* 一律 ``@pytest.mark.xfail(strict=True, reason='B-x: ...')``：
  缺陷被修复后用例会变成 ``XPASS`` → **失败**，从而强制清理这条记录
  （修复者应把用例改成正确的正向断言，并从清单中划掉）。
* 只做**只读**断言，不修改 ``src/`` 下任何实现。
* 所有分箱调用 ``no_cores=1``；需要数据集时缺失即 skip。
"""
import pickle

import numpy as np
import pandas as pd
import pytest

from syriskmodels.scorecard import (
    ComposedWOEBin,
    QuantileInitBin,
    RuleOptimBin,
    TreeOptimBin,
    woebin,
    woebin_plot,
    woebin_psi,
)
from syriskmodels.models import stepwise_lr

# --------------------------------------------------------------------------- #
# 公共小数据
# --------------------------------------------------------------------------- #

FEATURE_COLS = ['num_a', 'num_b', 'num_c']


def _synthetic(n: int = 600, seed: int = 20240501) -> pd.DataFrame:
    """固定种子小数据，含数值/类别/缺失/特殊值。"""
    rng = np.random.default_rng(seed)
    num_a = rng.normal(size=n)
    y = rng.binomial(1, 1 / (1 + np.exp(-(1.2 * num_a + 0.4))))
    df = pd.DataFrame({
        'num_a': np.round(num_a, 6),
        'num_b': np.round(rng.lognormal(0, 1.0, n), 6),
        'num_c': rng.integers(0, 5, n),
        'cat_a': rng.choice(['p', 'q', 'r', 's'], n),
        'target': y,
    })
    df.loc[df.index[:40], 'num_a'] = np.nan
    df.loc[df.index[40:70], 'num_b'] = -999.0
    df.loc[df.index[70:100], 'cat_a'] = np.nan
    return df


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return _synthetic()


def _bins(frame: pd.DataFrame, x, **kwargs):
    return woebin(
        frame, y='target', x=x, methods=['quantile', 'tree'],
        initial_bins=20, bin_num_limit=5, no_cores=1, **kwargs
    )


# --------------------------------------------------------------------------- #
# B-1 测试数据路径 / 软链接导致干净克隆下集成测试不可用
# --------------------------------------------------------------------------- #

def test_b1_no_hardcoded_csv_paths_in_tests():
    """B-1：test/ 下不应再出现指向 ``*.csv``（非 .csv.gz）的硬编码数据路径。

    历史实现读取 ``test/germancredit.csv`` / ``test/creditcard.csv``，而这两个
    文件在版本库中**从未存在**（只有被 .gitignore 忽略的 ``test/*.csv.gz``
    软链接），干净克隆后 7 个集成用例必然失败（见报告 B-1）。
    """
    from pathlib import Path
    test_dir = Path(__file__).resolve().parent
    # 这些文件是"合法消费者"：它们只在文档/错误信息里提到 .csv.gz
    allowed = {
        'test_known_bugs.py',            # 本文件（用例说明文字）
        'test_datasets.py',
        'test_scorecard_integration.py',
        'test_golden_binning.py',
        'conftest.py',
    }
    offenders = []
    for path in sorted(test_dir.rglob('*.py')):
        if path.name in allowed:
            continue
        text = path.read_text(encoding='utf-8')
        # 允许 *.csv.gz；只在出现裸 '.csv' / ".csv" 字面量时记录
        if ".csv'" in text or '.csv"' in text:
            offenders.append(path.name)
    assert not offenders, f'仍存在硬编码 .csv 数据路径: {offenders}'


# --------------------------------------------------------------------------- #
# B-2 pandas 3 下 woebin_plot 抛 KeyError: 'variable'
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(
    strict=True,
    reason="B-2: pandas 3 的 groupby.apply 不再把分组键保留为列，"
           "woebin_plot 中 bins_df['variable'] 抛 KeyError",
)
def test_b2_woebin_plot_works(synthetic_df):
    """B-2：``woebin_plot`` 应能对 woebin 结果生成图像。

    现象：``KeyError: 'variable'``
    位置：``src/syriskmodels/scorecard/api/evaluation.py:181-184``
    根因：``bins_df.groupby('variable', observed=False).apply(_gb_distr)``
          返回的 DataFrame 在 pandas 3 下丢失分组列（分组键只进 index），
          后续 ``bins_df['variable']`` 失败。
          已实测：``pandas=3.0.6`` 下 ``apply(lambda x: x.assign(...))``
          返回列不含 'variable'。
    影响：``woebin_plot`` 与 ``build_scorecard`` 尾部绘图全流程不可用。
    """
    frame = _synthetic()
    bins = _bins(frame, ['num_a', 'num_b', 'cat_a'])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    try:
        plots = woebin_plot(bins)
    finally:
        plt.close('all')

    assert isinstance(plots, dict)
    assert set(plots) == {'num_a', 'num_b', 'cat_a'}


# --------------------------------------------------------------------------- #
# B-3 stepwise_lr direction='forward' / 'backward' 返回 (None, [])
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(
    strict=True,
    reason="B-3: 择优循环被写在 if direction in ['backward','bidirectional'] "
           "分支内，forward/backward 方向永不更新 best_metrics",
)
@pytest.mark.parametrize('direction', ['forward', 'backward'])
def test_b3_stepwise_lr_single_direction_selects_features(synthetic_df, direction):
    """B-3：单向逐步回归应至少选出非空特征集合。

    现象：``stepwise_lr(direction='forward')`` 与 ``'backward'`` 均返回
          ``(None, [])``；``'bidirectional'`` 正常返回 ``(float, [...])``。
    位置：``src/syriskmodels/models.py:130-159``
    根因：``perf_records`` 在 forward 分支收集，但"择优并更新
          ``selected_features`` / ``best_metrics`` / ``improved``"的整个代码块
          被包在 ``if direction in ['backward', 'bidirectional']`` 内
          （models.py:145-152）。forward/backward 方向下 ``improved`` 恒为
          ``False``，首轮即 ``break``，返回初始的 ``(None, [])``。
    """
    frame = _synthetic()
    x = ['num_a', 'num_b', 'num_c']

    for col in x:
        frame[col + '_woe'] = frame[col].fillna(frame[col].median())

    best_metrics, selected = stepwise_lr(
        frame, y='target', x=[c + '_woe' for c in x], cv=2, direction=direction
    )

    assert best_metrics is not None
    assert isinstance(best_metrics, float)
    assert len(selected) > 0


# --------------------------------------------------------------------------- #
# B-4 / B-5 pandas 3 下 split_special_values 的 merge dtype 冲突
# --------------------------------------------------------------------------- #

def _numeric_special_values_frame() -> pd.DataFrame:
    """数值列 + 显式数值型特殊值（触发 B-4 的最小数据）。"""
    rng = np.random.default_rng(11)
    n = 400
    y = rng.binomial(1, 0.3, n)
    return pd.DataFrame({
        'num_b': np.where(rng.random(n) < 0.1, -999.0, rng.normal(size=n)),
        'target': y,
    })


@pytest.mark.xfail(
    strict=True,
    reason="B-4: pandas 3 下 pd.merge(object, float64) 直接 ValueError，"
           "数值列 + 显式数值型 special_values 无法分箱",
)
def test_b4_numeric_special_values_binning():
    """B-4：数值列传入数值型 ``special_values``（如 ``['-999']``）应能分箱。

    现象：``ValueError: You are trying to merge on float64 and object columns
          for key 'value'.``
    位置：``src/syriskmodels/scorecard/core/base.py:113-118``
    根因：``dtm.fillna("missing")`` 把 NaN 引入后使 object 化列的 dtype 变为
          object，而 ``sv_df['value']`` 仍是 float64（或反之）；pandas 3 不再
          隐式放宽 object/float 的 merge 键类型（pandas 2 可隐式转换）。
    影响：``woebin(special_values=[-999, -1, ...])`` 这一**文档化的主推用法**
          在 pandas 3 下整体不可用。
    """
    frame = _numeric_special_values_frame()
    result = woebin(
        frame, y='target', x=['num_b'], methods=['quantile', 'tree'],
        initial_bins=20, bin_num_limit=5,
        special_values={'num_b': ['-999', 'missing']}, no_cores=1,
    )
    bins = result['num_b']
    assert isinstance(bins, pd.DataFrame)
    assert bool(bins['is_special_values'].any())


@pytest.mark.xfail(
    strict=True,
    reason="B-5: pandas 3 下 astype(int) 遇到 'missing' 填出的 NaN 抛 "
           "ValueError: cannot convert float NaN to integer",
)
def test_b5_integer_column_special_values_binning():
    """B-5：整型列传入数值型特殊值应能分箱。

    现象：``ValueError: cannot convert float NaN to integer``
    位置：``src/syriskmodels/scorecard/core/base.py:104-105``
    根因：``sv_df['value'].astype(dtm['value'].dtypes)`` 中，若
          ``special_values`` 含 ``'missing'``，``split_vec_to_df`` 会产出
          ``None``/NaN，对 int64 列执行 ``astype(int64)`` 直接抛错
          （pandas 3 不再允许 NaN → int 的静默截断）。
    影响：整型变量（如 ``number.of.existing.credits``）无法使用 ``missing``
          特殊值。
    """
    rng = np.random.default_rng(12)
    n = 400
    frame = pd.DataFrame({
        'num_c': rng.integers(0, 5, n),
        'target': rng.binomial(1, 0.3, n),
    })
    result = woebin(
        frame, y='target', x=['num_c'], methods=['quantile', 'tree'],
        initial_bins=20, bin_num_limit=5,
        special_values={'num_c': ['-1', 'missing']}, no_cores=1,
    )
    assert isinstance(result['num_c'], pd.DataFrame)


# --------------------------------------------------------------------------- #
# B-6 WOEBinFactory 把全局 kwargs 透传给所有 binner
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(
    strict=True,
    reason="B-6: WOEBinFactory.build 把 **kwargs 透传给每个 binner 构造函数，"
           "methods 含 rule 时 initial_bins 触发 TypeError",
)
def test_b6_quantile_rule_with_initial_bins(synthetic_df):
    """B-6：``methods=['quantile','rule']`` 配合 ``initial_bins`` 应能分箱。

    现象：``TypeError: RuleOptimBin.__init__() got an unexpected keyword
          argument 'initial_bins'``
    位置：``src/syriskmodels/scorecard/core/factory.py:122``
          （``cls.get_binner(bin_cls, **kwargs)`` 对列表内**所有**类别透传同一份
          kwargs），配合 ``bins/optimal.py:285-291``（``RuleOptimBin.__init__``
          没有 ``**kwargs``）。
    影响：README 中"首个细分箱 + 后续粗分箱共享 kwargs"的组合方式对
          ``rule`` 失效；用户必须改用类实例列表（``[QuantileInitBin(20),
          RuleOptimBin()]``）绕过。
    """
    frame = _synthetic()
    result = woebin(
        frame, y='target', x=['num_a', 'num_b'], methods=['quantile', 'rule'],
        initial_bins=20, no_cores=1,
    )
    assert isinstance(result['num_a'], pd.DataFrame)


# --------------------------------------------------------------------------- #
# B-7 RuleOptimBin.__init__ 不接受 / 不传递 **kwargs
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(
    strict=True,
    reason="B-7: RuleOptimBin.__init__ 不接收也不传递 **kwargs（含 eps）",
)
def test_b7_rule_optim_bin_accepts_kwargs():
    """B-7：``RuleOptimBin`` 应像其他粗分箱类一样接受并传递 ``**kwargs``。

    现象：``RuleOptimBin(initial_bins=20, eps=0.5)`` →
          ``TypeError: unexpected keyword argument``
    位置：``src/syriskmodels/scorecard/bins/optimal.py:285-297``
    对比：``TreeOptimBin.__init__`` / ``ChiMergeOptimBin.__init__`` 均有
          ``**kwargs`` 并调用 ``super().__init__(**kwargs)``。
    影响：① 无法通过 kwargs 调整 ``eps``（基类默认 0.5，RuleOptimBin 自己
          写死 ``eps=1e-8`` 且不传给父类，行为与基类不一致）；
          ② 与 ``WOEBinFactory`` 的 kwargs 透传机制不兼容（见 B-6）。
    """
    binner = RuleOptimBin(initial_bins=20)
    assert isinstance(binner, RuleOptimBin)


# --------------------------------------------------------------------------- #
# B-8 check_breaks_list 使用 eval
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(
    strict=True,
    reason="B-8: check_breaks_list 使用 eval 而非 ast.literal_eval，"
           "字符串入参可执行任意表达式（代码注入面）",
)
def test_b8_check_breaks_list_rejects_non_literal_expressions():
    """B-8：``check_breaks_list`` 不应求值非字面量表达式（安全）。

    现象：入参为字符串时直接 ``eval(breaks_list)``，任意表达式都会被执行。
    位置：``src/syriskmodels/scorecard/utils/validation.py:222-231``
    风险：``breaks_list`` 常来自配置文件 / 外部输入，``eval`` 使配置具备
          代码执行能力，属于注入面。
    建议：W2 改为 ``ast.literal_eval``，并保留"必须是字典"的校验。

    用例设计：断言"含函数调用的字符串被拒绝"。当前实现会成功求值 →
    xfail；改为 ``literal_eval`` 后抛 ``ValueError`` → 用例通过（XPASS 消失），
    此时请把本用例改成不带 xfail 的正向断言并更新 B-8 记录。
    """
    from syriskmodels.scorecard.utils.validation import check_breaks_list

    # 合法用法必须继续可用（字典直传）
    assert check_breaks_list({'age': [20, 30]}) == {'age': [20, 30]}

    with pytest.raises(Exception):
        check_breaks_list("{'age': [len('abcd')]}")


# --------------------------------------------------------------------------- #
# B-9 woebin_psi 在一侧缺失分箱时静默给出错误 PSI
# --------------------------------------------------------------------------- #

@pytest.mark.xfail(
    strict=True,
    reason="B-9: 一侧缺失分箱时 cmp_distr 为 NaN，psi() 用 0 填充后重归一化，"
           "同一变量的所有分箱都得到同一个偏大的 PSI 值",
)
def test_b9_woebin_psi_one_sided_bins_are_not_silently_inflated(synthetic_df):
    """B-9：比较集缺少某个分箱时，PSI 不应在所有行上静默放大。

    现象：比较集里只出现 ``cat_a`` 的一个类别时，
          ``cmp_distr`` 对该变量的多数分箱为 ``NaN``；
          ``evaluate.psi`` 内部 ``np.where(np.isnan(x), 0, x)`` 把 NaN 当 0
          处理再重新归一化，导致该变量**每一行**都得到同一个偏大的 PSI
          （实测约 4.897，而正常量级应 < 1）。
    位置：``src/syriskmodels/scorecard/api/evaluation.py:48-70``
          （``pd.pivot_table`` 后 ``tmp.columns = ['base','cmp']`` 未区分列名，
          缺失一侧直接变成 NaN）
          与 ``src/syriskmodels/evaluate.py:171-193``（``psi`` 的 NaN→0）。
    风险：变量稳定性结论完全错误，且没有任何告警。
    """
    frame = _synthetic(n=800)
    bins = _bins(frame, ['cat_a', 'num_a'])

    base = frame.iloc[:400]
    tail = frame.iloc[400:].copy()
    keep = bins['cat_a']['bin'].iloc[0].split('%,%')
    cmp_df = tail[tail['cat_a'].isin(keep)]
    assert len(cmp_df) > 0

    result = woebin_psi(base, cmp_df, bins)
    cat_psi = result[result['variable'] == 'cat_a']

    # 期望：每个分箱都有自己的分布，PSI 是变量级稳定的有限值
    assert cat_psi['cmp_distr'].notna().all(), 'PSI 明细中不应出现 NaN 分布'
    assert cat_psi['psi'].nunique() == 1, '同一个变量的 PSI 应只有一个取值'
    assert cat_psi['psi'].iloc[0] < 1.0, (
        f"单侧缺失分箱导致 PSI 被放大到 {cat_psi['psi'].iloc[0]:.4f}"
    )


# --------------------------------------------------------------------------- #
# B-10 build_scorecard 用 OOT 做变量筛选（泄漏）
# --------------------------------------------------------------------------- #

def _build_scorecard_source() -> str:
    import pathlib
    return (
        pathlib.Path(__file__).resolve().parent.parent
        / 'src' / 'syriskmodels' / 'contrib' / 'build_scorecard.py'
    ).read_text(encoding='utf-8')


@pytest.mark.xfail(
    strict=True,
    reason="B-10: build_scorecard 用 OOT（oot_df）做风险趋势一致性筛选，"
           "样本外信息进入特征选择环节",
)
def test_b10_variable_selection_uses_oot_and_leaks_into_woe_selection():
    """B-10：``build_scorecard`` 的变量筛选依赖 OOT 数据（信息泄漏）。

    现象（静态定位）：``build_scorecard.py`` 的变量筛选流程为

        1. ``bins = woebin(train_df, ...)``        —— 只在训练集上分箱（正确）
        2. ``selected_variables = iv_df[...]``     —— 用训练集 IV + 单调性初筛
        3. ``risk_trends_consistency(oot_df, ...)``—— **用 OOT 做趋势一致性筛选**
        4. ``stepwise_lr(woebin_ply(train_df, ...))`` —— 用第 3 步的结果作为
           候选池决定哪些变量进入模型

    第 3 步把 OOT（样本外）信息带入了**特征选择**环节：OOT 通过
    ``train_filter`` / ``oot_filter`` 划分，本应只在最终评估阶段使用。
    结果是 OOT 不再是干净的样本外验证集，模型效果与 PSI 评估均偏乐观。

    说明：这是流程设计问题而非崩溃 bug，因此本用例做**静态断言**（廉价、确定），
    要求该调用在 W2 显式处置（改为在训练集/验证集上做趋势筛选，或在文档中
    明确声明 OOT 参与筛选）。处置后本用例会 XPASS → 失败，请更新记录。
    保持 xfail 直到流程被修改，避免"记录悄悄失效"。
    """
    import ast

    tree = ast.parse(_build_scorecard_source())
    build_fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == 'build_scorecard'
    )

    leak_calls = []
    for node in ast.walk(build_fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, 'id', None) or getattr(func, 'attr', None)
        arg_names = [
            getattr(arg, 'id', None) or getattr(arg, 'attr', None)
            for arg in node.args
        ]
        if name == 'risk_trends_consistency' and 'oot_df' in arg_names:
            leak_calls.append(f'line {node.lineno}: risk_trends_consistency({arg_names})')

    assert not leak_calls, (
        'build_scorecard 仍使用 OOT 做变量筛选（B-10 未处置）：' + '; '.join(leak_calls)
    )


# --------------------------------------------------------------------------- #
# B-11 multiprocessing 默认路径：picklability 与 spawn 安全性
# --------------------------------------------------------------------------- #

def test_b11_woebin_returns_identical_result_for_single_core(synthetic_df):
    """B-11（护栏）：分箱器对象可 pickle，且 ``no_cores=1`` 结果确定。

    ``woebin`` / ``woebin_ply`` 在 ``no_cores=None`` 时会自行计算核数并走
    ``mp.Pool``（``api/woebin.py:119-125,151-156``、
    ``api/transform.py:68-71,85-90``）。macOS / Windows 默认 spawn 语义，
    Pool 需要 pickle 任务与 binner 对象；同时小数据下并行与进程启动开销
    相比并无收益（见报告 B-11 的实测数据）。

    本用例只做**廉价**的不变量检查（不影响 CI 时长）：
    1. ``ComposedWOEBin`` 及其子分箱器实例可 pickle 往返（spawn 的前提）；
    2. 同一输入两次运行结果完全一致（确定性护栏）。
    """
    frame = _synthetic()
    binners = ComposedWOEBin([QuantileInitBin(initial_bins=20), TreeOptimBin(bin_num_limit=5)])

    payload = pickle.dumps(binners)
    restored = pickle.loads(payload)
    assert isinstance(restored, ComposedWOEBin)
    assert repr(restored) == repr(binners)

    first = _bins(frame, ['num_a', 'num_b', 'cat_a'])
    second = _bins(frame, ['num_a', 'num_b', 'cat_a'])
    for variable in first:
        pd.testing.assert_frame_equal(first[variable], second[variable])


# --------------------------------------------------------------------------- #
# B-12 binning_helpers 与生产代码脱节
# --------------------------------------------------------------------------- #

def test_b12_binning_helpers_not_used_by_production_path():
    """B-12：``scorecard.utils.binning_helpers`` 当前只被再导出，未被生产路径使用。

    现象：``src/syriskmodels/scorecard/{core,bins,api}`` 中没有任何模块 import
          ``binning_helpers``（仅 ``scorecard/__init__.py`` 与测试引用）。
          真实切分点提取走 ``api/transform.woebin_breaks`` 与
          ``core/base.WOEBin.binning_breaks``，两套实现并存。
    风险：① ``test/test_scorecard/test_binning_helpers.py`` 的 30+ 断言覆盖的是
          **未被使用**的代码，给人"分箱逻辑已充分覆盖"的错觉；
          ② 后续重构若误以为它是生产路径，会改错地方。
    说明：断言"当前无生产引用"，用于在 W2 明确处置（接入或删除）后强制更新记录。
    """
    import pathlib
    src = pathlib.Path(__file__).resolve().parent.parent / 'src' / 'syriskmodels' / 'scorecard'
    referencing = sorted(
        str(path.relative_to(src))
        for path in src.rglob('*.py')
        if 'binning_helpers' in path.read_text(encoding='utf-8')
    )
    assert referencing == ['__init__.py'], (
        f'binning_helpers 的引用发生了变化：{referencing}；'
        f'请同步更新 W1 报告中的 B-12 记录'
    )


def test_b12_binning_helpers_semantics_do_not_match_production():
    """B-12（补充）：两套切分点提取对同一分箱给出不同的 ``breaks`` 语义。

    ``binning_helpers.extract_breaks_from_binning`` 返回的是**分箱右边界**
    （含 ``inf``），而生产路径 ``woebin_breaks`` 返回的是分箱名（类别型）或
    右边界字符串（数值型）。两者不能互换使用。
    """
    from syriskmodels.scorecard.utils.binning_helpers import (
        extract_breaks_from_binning,
    )

    numeric_binning = pd.DataFrame({
        'bin_chr': ['[-inf, 20)', '[20, 40)', '[40, inf)'],
    })
    helper_breaks = list(extract_breaks_from_binning(numeric_binning, is_numeric=True))
    assert helper_breaks == [20.0, 40.0, np.inf]

    frame = _synthetic()
    bins = _bins(frame, ['num_a'])
    production_breaks = bins['num_a']['breaks'].tolist()
    assert production_breaks != [str(b) for b in helper_breaks], (
        '两套实现的 breaks 语义已一致，请复核 B-12 记录是否仍然成立'
    )

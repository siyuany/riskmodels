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
# B-2 pandas 3 下 woebin_plot 抛 KeyError: 'variable'（W2 已修复）
# --------------------------------------------------------------------------- #

def test_b2_woebin_plot_works(synthetic_df):
    """B-2（已修复）：``woebin_plot`` 应能对 woebin 结果生成图像。

    历史缺陷：``bins_df.groupby('variable', observed=False).apply(_gb_distr)``
    在 pandas 3 下不再把分组键保留为列（分组键只进 index），后续
    ``bins_df['variable']`` 抛 ``KeyError``；且 ``_plot_single_bin`` 引用了
    woebin 输出中不存在的 ``bin_chr`` 列（实际列名为 ``bin``）。

    W2 修复：改用 ``groupby(...).transform('sum')`` 向量化计算
    good_distr/bad_distr（不再依赖 apply 的分组列行为），x 轴刻度改用
    ``bin`` 列。图形语义不变。
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
    for fig in plots.values():
        assert fig is not None


# --------------------------------------------------------------------------- #
# B-3 stepwise_lr direction='forward' / 'backward' 返回 (None, [])
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('direction', ['forward', 'backward'])
def test_b3_stepwise_lr_single_direction_selects_features(synthetic_df, direction):
    """B-3（已修复）：单向逐步回归应至少选出非空特征集合。

    历史缺陷：``stepwise_lr(direction='forward')`` 与 ``'backward'`` 均返回
    ``(None, [])``；``'bidirectional'`` 正常。根因是"择优并更新
    ``selected_features`` / ``best_metrics`` / ``improved``"的整段代码被包在
    ``if direction in ['backward', 'bidirectional']`` 分支内，且候选生成只在
    ``forward/bidirectional`` 分支 —— forward 收集完候选后 ``improved`` 恒为
    ``False`` 首轮即 break；backward 则根本不产生候选。

    W2 修复：把"生成候选"与"择优更新"拆开，direction 只控制候选集合
    （forward=加入候选；backward=剔除候选；bidirectional=两者），择优更新
    对三种方向一致生效；纯 backward 未指定 ``initial_features`` 时从全量
    特征集起步（经典后向消除），并以当前集合的指标为基线，只有严格改进
    才接受剔除。
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
    assert set(selected) <= {c + '_woe' for c in x}


def test_b3_stepwise_lr_direction_semantics(synthetic_df):
    """B-3 补充：三种 direction 的语义约束。

    * bidirectional 保持修复前行为（返回非空合理结果）；
    * backward + initial_features：结果是 initial 的子集（只剔除不加入）；
    * forward：结果是候选池子集（只加入不剔除，从空集起步）。
    """
    frame = _synthetic()
    x = ['num_a', 'num_b', 'num_c']
    xw = [c + '_woe' for c in x]
    for col in x:
        frame[col + '_woe'] = frame[col].fillna(frame[col].median())

    m_bi, sel_bi = stepwise_lr(frame, y='target', x=xw, cv=2,
                               direction='bidirectional')
    assert isinstance(m_bi, float) and len(sel_bi) > 0

    m_bw, sel_bw = stepwise_lr(frame, y='target', x=xw, cv=2,
                               direction='backward', initial_features=xw[:2])
    assert isinstance(m_bw, float)
    assert set(sel_bw) <= set(xw[:2])
    assert len(sel_bw) > 0

    m_fw, sel_fw = stepwise_lr(frame, y='target', x=xw, cv=2,
                               direction='forward')
    assert isinstance(m_fw, float)
    assert set(sel_fw) <= set(xw)
    assert len(sel_fw) > 0


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


def test_b4_numeric_special_values_binning():
    """B-4（已修复）：数值列传入数值型 ``special_values``（如 ``['-999']``）应能分箱。

    历史缺陷：``ValueError: You are trying to merge on float64 and object
    columns for key 'value'``。根因是 ``split_special_values`` 中
    ``dtm.fillna("missing")`` 与 ``sv_df`` 的 merge 键 dtype 不一致
    （pandas 3 不再隐式放宽 object↔float 的 merge 键类型）。

    W2 修复：特殊值拆分不再经过 fillna+merge，改为布尔掩码 + 值→bin_chr
    映射；NaN、特殊值、数值区间分箱语义与 pandas 2 一致。
    修复后的输出边界另由 golden 快照钉住（见 test_golden_binning.py 的
    ``synthetic_numeric_special_values`` 用例）。
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

    sv_bins = bins[bins['is_special_values']]
    # -999 哨兵值单独成箱；数据中无 NaN，'missing' 条目不产生分箱
    assert sv_bins['bin'].tolist() == ['-999.0']
    assert int(sv_bins['count'].iloc[0]) == int((frame['num_b'] == -999.0).sum())
    # 非特殊值部分正常区间分箱，且总数守恒
    ns_bins = bins[~bins['is_special_values']]
    assert len(ns_bins) > 0
    assert int(bins['count'].sum()) == len(frame)


def test_b4_numeric_special_values_edge_cases():
    """B-4 边界：多特殊值 / 组合特殊值 / float 列含 NaN。"""
    rng = np.random.default_rng(23)
    n = 400
    value = np.where(rng.random(n) < 0.1, -999.0, rng.normal(size=n))
    value[10:20] = -1.0          # 第二个哨兵值
    value[20:25] = np.nan        # 缺失
    frame = pd.DataFrame({'v': value, 'target': rng.binomial(1, 0.3, n)})

    # 多特殊值 + missing：各自单独成箱
    result = woebin(
        frame, y='target', x=['v'], methods=['quantile'],
        initial_bins=10, special_values={'v': ['-999', '-1', 'missing']},
        no_cores=1,
    )
    bins = result['v']
    sv = bins[bins['is_special_values']]
    assert set(sv['bin']) == {'missing', '-999.0', '-1.0'}
    assert int(sv.loc[sv['bin'] == 'missing', 'count'].iloc[0]) == 5
    assert int(sv.loc[sv['bin'] == '-999.0', 'count'].iloc[0]) == int(
        (frame['v'] == -999.0).sum())
    assert int(sv.loc[sv['bin'] == '-1.0', 'count'].iloc[0]) == 10
    assert int(bins['count'].sum()) == n

    # 组合特殊值在数值列上按 legacy 语义拆分为独立分箱（'0%,%1' → '0.0' 与 '1.0'）
    frame2 = pd.DataFrame({
        'v2': rng.integers(0, 9, n).astype(float),
        'target': rng.binomial(1, 0.3, n),
    })
    result2 = woebin(
        frame2, y='target', x=['v2'], methods=['quantile'],
        initial_bins=5, special_values={'v2': ['0%,%1']}, no_cores=1,
    )
    sv2 = result2['v2'][result2['v2']['is_special_values']]
    assert set(sv2['bin']) == {'0.0', '1.0'}


def test_b5_integer_column_special_values_binning():
    """B-5（已修复）：整型列传入 ``missing`` 等特殊值应能分箱。

    历史缺陷：``ValueError: cannot convert float NaN to integer``。根因是
    ``sv_df['value'].astype(原 int dtype)`` 对 ``'missing'`` 产出的 NaN 执行
    int 转换（pandas 3 不再允许 NaN → int）。

    W2 修复：特殊值仅在**不含 NaN** 时才 astype 回原整型 dtype；含 NaN 时
    保持 float 语义（整型列 + 'missing' 时数值特殊值标签为 float 形式，
    如 ``-1.0``；该新行为由 golden 快照
    ``synthetic_integer_missing_special_value`` 钉住）。
    """
    rng = np.random.default_rng(12)
    n = 400
    value = rng.integers(0, 5, n)
    value[30:40] = -1
    frame = pd.DataFrame({
        'num_c': value,
        'target': rng.binomial(1, 0.3, n),
    })
    # 整型列本身无 NaN，但 special_values 含 'missing' → sv 值集合含 NaN
    result = woebin(
        frame, y='target', x=['num_c'], methods=['quantile', 'tree'],
        initial_bins=20, bin_num_limit=5,
        special_values={'num_c': ['-1', 'missing']}, no_cores=1,
    )
    bins = result['num_c']
    assert isinstance(bins, pd.DataFrame)
    sv = bins[bins['is_special_values']]
    assert sv['bin'].tolist() == ['-1.0']
    assert int(sv['count'].iloc[0]) == 10
    assert int(bins['count'].sum()) == n


def test_b5_integer_column_with_nan_and_missing():
    """B-5 边界：含 NaN 的数值列 + 'missing' 特殊值。"""
    rng = np.random.default_rng(34)
    n = 300
    value = rng.integers(0, 6, n).astype(float)
    value[:20] = np.nan
    frame = pd.DataFrame({'v': value, 'target': rng.binomial(1, 0.25, n)})
    result = woebin(
        frame, y='target', x=['v'], methods=['quantile', 'tree'],
        initial_bins=10, bin_num_limit=4,
        special_values={'v': ['missing']}, no_cores=1,
    )
    bins = result['v']
    sv = bins[bins['is_special_values']]
    assert sv['bin'].tolist() == ['missing']
    assert int(sv['count'].iloc[0]) == 20
    assert int(bins['count'].sum()) == n


# --------------------------------------------------------------------------- #
# B-6 WOEBinFactory 把全局 kwargs 透传给所有 binner
# --------------------------------------------------------------------------- #

def test_b6_quantile_rule_with_initial_bins(synthetic_df):
    """B-6（已修复）：``methods=['quantile','rule']`` 配合 ``initial_bins`` 应能分箱。

    历史缺陷：``WOEBinFactory.build`` 把 ``**kwargs`` 无差别透传给每个 binner
    构造函数，``RuleOptimBin.__init__`` 没有 ``**kwargs`` →
    ``TypeError: unexpected keyword argument 'initial_bins'``。

    W2 修复：``get_binner`` 按构造函数签名过滤 kwargs —— 构造函数不接受
    ``**kwargs`` 的类只收到其显式声明的参数；接受 ``**kwargs`` 的类保持
    legacy 的全量透传（多余参数由基类收纳）。叠加 B-7（RuleOptimBin 接收
    ``**kwargs``）后，字符串 methods 组合对 rule 生效。
    """
    frame = _synthetic()
    result = woebin(
        frame, y='target', x=['num_a', 'num_b'], methods=['quantile', 'rule'],
        initial_bins=20, no_cores=1,
    )
    assert isinstance(result['num_a'], pd.DataFrame)
    assert isinstance(result['num_b'], pd.DataFrame)


def test_b6_factory_kwargs_dispatch_and_input_forms():
    """B-6 回归：kwargs 按构造签名过滤；实例/类/注册名三种传参方式保留。"""
    from syriskmodels.scorecard import (
        ChiMergeOptimBin,
        ComposedWOEBin,
        QuantileInitBin,
        TreeOptimBin,
        WOEBin,
        WOEBinFactory,
    )

    # 不接受 **kwargs 的严格构造类：无关 kwargs 应被过滤而不是抛 TypeError
    class StrictBin(WOEBin):
        def __init__(self, bin_num_limit=5):
            super().__init__()
            self.bin_num_limit = bin_num_limit

        def woebin(self, dtm, breaks=None):
            return [-np.inf, np.inf]

    binner = WOEBinFactory.get_binner(
        StrictBin, bin_num_limit=3, initial_bins=20, whatever=1)
    assert binner.bin_num_limit == 3

    # 接受 **kwargs 的类：保持 legacy 全量透传（基类收纳多余参数）
    tree = WOEBinFactory.get_binner(TreeOptimBin, bin_num_limit=4,
                                    initial_bins=20)
    assert tree.bin_num_limit == 4
    assert tree.kwargs.get('initial_bins') == 20

    # 三种传参方式：注册名 / 类 / 实例
    composed = WOEBinFactory.build(['quantile', 'tree'], initial_bins=20,
                                   bin_num_limit=5)
    assert isinstance(composed, ComposedWOEBin)
    assert isinstance(composed.bins[0], QuantileInitBin)
    assert composed.bins[0].n_bins == 20
    assert isinstance(composed.bins[1], TreeOptimBin)
    assert composed.bins[1].bin_num_limit == 5

    composed_cls = WOEBinFactory.build([QuantileInitBin, ChiMergeOptimBin],
                                       initial_bins=10)
    assert isinstance(composed_cls.bins[0], QuantileInitBin)
    assert composed_cls.bins[0].n_bins == 10
    assert isinstance(composed_cls.bins[1], ChiMergeOptimBin)

    inst_a, inst_b = QuantileInitBin(initial_bins=7), TreeOptimBin(bin_num_limit=2)
    composed_inst = WOEBinFactory.build([inst_a, inst_b], initial_bins=99)
    assert composed_inst.bins[0] is inst_a   # 实例原样使用，kwargs 不覆盖
    assert composed_inst.bins[1] is inst_b

    with pytest.raises(KeyError):
        WOEBinFactory.get_binner('no_such_method')


# --------------------------------------------------------------------------- #
# B-7 RuleOptimBin.__init__ 不接受 / 不传递 **kwargs
# --------------------------------------------------------------------------- #

def test_b7_rule_optim_bin_accepts_kwargs():
    """B-7（已修复）：``RuleOptimBin`` 应像其他粗分箱类一样接受 ``**kwargs``。

    历史缺陷：``RuleOptimBin(initial_bins=20, eps=0.5)`` →
    ``TypeError: unexpected keyword argument``；且 ``super().__init__()``
    未传参，任意基类参数都被吞掉。

    W2 修复：``__init__`` 接收 ``**kwargs`` 并传给 ``super().__init__``。
    注意语义决策（记录于 W2 报告）：``RuleOptimBin.eps`` 保持**历史含义**
    （lift 平滑项，默认 1e-8），不转发给基类 —— 若转发会把基类
    ``epsilon``（WOE 零计数替换值）从 0.5 变为 1e-8，**改变 rule 分箱的
    默认输出**，违反 W2「默认不改变分箱输出」的硬性约束。基类参数可经
    ``**kwargs`` 链到达基类（``eps`` 名称冲突除外，维持 legacy 行为）。
    """
    binner = RuleOptimBin(initial_bins=20)
    assert isinstance(binner, RuleOptimBin)
    # 多余 kwargs 由基类收纳
    assert binner.kwargs.get('initial_bins') == 20
    # 默认语义不变：lift 平滑 eps=1e-8；基类 epsilon 保持默认 0.5
    assert binner._eps == 1e-8
    assert binner.epsilon == 0.5
    # 显式 eps 仍按历史语义作用于 lift 平滑，不影响基类 epsilon
    binner2 = RuleOptimBin(eps=0.1, lift=5)
    assert binner2._eps == 0.1
    assert binner2._min_lift == 5
    assert binner2.epsilon == 0.5


# --------------------------------------------------------------------------- #
# B-8 check_breaks_list 使用 eval
# --------------------------------------------------------------------------- #

def test_b8_check_breaks_list_rejects_non_literal_expressions():
    """B-8（已修复）：``check_breaks_list`` 不应求值非字面量表达式（安全）。

    历史缺陷：入参为字符串时直接 ``eval(breaks_list)``，任意表达式都会被
    执行（``breaks_list`` 常来自配置文件，构成代码注入面）。

    W2 修复：改用 ``ast.literal_eval``，保留"必须是字典"的校验；非法表达式
    抛 ``ValueError`` 而不是被执行。
    """
    from syriskmodels.scorecard.utils.validation import check_breaks_list

    # 合法用法必须继续可用（字典直传 / 字面量字符串）
    assert check_breaks_list({'age': [20, 30]}) == {'age': [20, 30]}
    assert check_breaks_list("{'age': [20, 30]}") == {'age': [20, 30]}
    assert check_breaks_list(None) == {}

    # 非字面量表达式：抛 ValueError，且表达式不被执行
    with pytest.raises(ValueError):
        check_breaks_list("{'age': [len('abcd')]}")
    with pytest.raises(ValueError):
        check_breaks_list("__import__('os').getcwd()")

    # 字面量但不是字典：保留"必须是字典"校验
    with pytest.raises(Exception):
        check_breaks_list("[20, 30]")


# --------------------------------------------------------------------------- #
# B-9 woebin_psi 在一侧缺失分箱时静默给出错误 PSI
# --------------------------------------------------------------------------- #

def test_b9_woebin_psi_one_sided_bins_are_not_silently_inflated(synthetic_df):
    """B-9（已修复）：比较集缺少某个分箱时，PSI 应有限、可解释。

    历史缺陷：``pd.pivot_table`` 后 ``tmp.columns = ['base','cmp']`` 未区分
    缺失侧 → ``cmp_distr`` 出现 NaN；``evaluate.psi`` 把 NaN 当 0 再重新
    归一化，同一变量的所有分箱得到同一个被静默放大的 PSI。
    （另注：W1 版本的本用例误取 ``bin.iloc[0]``——那是 'missing' 特殊值箱，
    导致 cmp_df 为空、断言路径根本到不了 PSI；W2 改为取真实类别箱。）

    W2 修复：``woebin_psi`` 对两侧分箱计数取**并集** reindex，缺失侧显式
    fillna(0) 后归一化；正常场景（两侧分箱齐全）输出与修复前一致；
    ``evaluate.psi`` 收到 NaN 时给出显式告警。
    """
    from syriskmodels.evaluate import psi as psi_fn

    frame = _synthetic(n=800)
    bins = _bins(frame, ['cat_a', 'num_a'])

    base = frame.iloc[:400]
    tail = frame.iloc[400:].copy()

    # 取一个非特殊值的真实类别箱，让比较集只覆盖它 → 其余分箱单侧缺失
    cat_bins = bins['cat_a']
    keep_bin = cat_bins.loc[~cat_bins['is_special_values'], 'bin'].iloc[0]
    keep = keep_bin.split('%,%')
    cmp_df = tail[tail['cat_a'].isin(keep)]
    assert len(cmp_df) > 0

    result = woebin_psi(base, cmp_df, bins)
    cat_psi = result[result['variable'] == 'cat_a']

    # 缺失侧显式补 0：明细中不应出现 NaN 分布
    assert cat_psi['cmp_distr'].notna().all(), 'PSI 明细中不应出现 NaN 分布'
    assert cat_psi['base_distr'].notna().all()
    # 单侧缺失的分箱 cmp_distr 应为 0（补零），而非 NaN
    missing_bins = set(cat_bins['bin']) - {keep_bin}
    zero_rows = cat_psi[cat_psi['bin'].isin(missing_bins)]
    assert len(zero_rows) > 0
    assert (zero_rows['cmp_distr'] == 0).all()
    # 同一个变量只有一个 PSI 取值，且有限
    assert cat_psi['psi'].nunique() == 1
    psi_value = float(cat_psi['psi'].iloc[0])
    assert np.isfinite(psi_value)
    # 可解释：与"对表中显式分布直接调用 psi()"的结果一致（无隐藏的二次放大）
    expected = psi_fn(cat_psi['base_distr'].to_numpy(),
                      cat_psi['cmp_distr'].to_numpy())
    assert psi_value == pytest.approx(float(expected), rel=1e-12)
    # 极端单侧缺失应提示显著不稳定（超过 0.25 的常用警戒线）
    assert psi_value > 0.25

    # 正常场景回归：两侧分箱齐全时 PSI 应为小值（分布来自同一生成过程）
    perm = np.random.default_rng(7).permutation(len(frame))
    base_n = frame.iloc[perm[:400]]
    cmp_n = frame.iloc[perm[400:]]
    result_n = woebin_psi(base_n, cmp_n, bins)
    cat_psi_n = result_n[result_n['variable'] == 'cat_a']
    assert cat_psi_n['cmp_distr'].notna().all()
    assert (cat_psi_n['cmp_distr'] > 0).all(), '随机对半分割下两侧分箱应齐全'
    assert float(cat_psi_n['psi'].iloc[0]) < 0.25


# --------------------------------------------------------------------------- #
# B-10 build_scorecard 用 OOT 做变量筛选（泄漏）
# --------------------------------------------------------------------------- #

def _build_scorecard_source() -> str:
    import pathlib
    return (
        pathlib.Path(__file__).resolve().parent.parent
        / 'src' / 'syriskmodels' / 'contrib' / 'build_scorecard.py'
    ).read_text(encoding='utf-8')


def test_b10_variable_selection_no_longer_uses_oot_by_default():
    """B-10（已修复，静态断言）：``build_scorecard`` 默认不用 OOT 做变量筛选。

    历史缺陷：变量筛选链路为 训练集分箱 → 训练集 IV/单调性初筛 →
    ``risk_trends_consistency(oot_df, ...)`` → ``stepwise_lr`` 候选池。
    第 3 步把 OOT（样本外）信息带入了**特征选择**环节，OOT 不再是干净的
    样本外验证集，模型效果与 PSI 评估均偏乐观。

    W2 修复：新增 ``risk_consistency_dataset='valid'|'oot'`` 参数，默认
    ``'valid'``（02_test 验证集；验证集为空时回退训练集并告警）；显式选择
    ``'oot'`` 时给出明确 warning 说明报告偏乐观。OOT 只用于最终评估。
    行为验证见 ``test_b10_risk_consistency_dataset_behavior``。
    """
    import ast
    import inspect

    from syriskmodels.contrib.build_scorecard import build_scorecard

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
        'build_scorecard 仍直接把 OOT 传入变量筛选：' + '; '.join(leak_calls)
    )

    sig = inspect.signature(build_scorecard)
    assert 'risk_consistency_dataset' in sig.parameters, (
        '缺少显式的 risk_consistency_dataset 参数'
    )
    assert sig.parameters['risk_consistency_dataset'].default == 'valid', (
        "risk_consistency_dataset 默认值必须是 'valid'（OOT 只做最终评估）"
    )


def _b10_sample_df(n: int = 600, seed: int = 5) -> pd.DataFrame:
    """B-10 行为测试用的小样本（1 个信号变量 + 2 个噪声变量 + Time）。"""
    rng = np.random.default_rng(seed)
    f1 = np.round(rng.normal(size=n), 6)
    p = 1 / (1 + np.exp(-(1.5 * f1 - 0.5)))
    return pd.DataFrame({
        'f1': f1,
        'f2': np.round(rng.normal(size=n), 6),
        'f3': rng.integers(0, 4, n).astype(float),
        'Class': rng.binomial(1, p),
        'Time': rng.integers(0, 200000, n),
    })


@pytest.mark.parametrize('option,expected_flag', [
    (None, '02_test'),   # 默认 → 验证集
    ('oot', '03_oot'),   # 显式 OOT → 允许，但必须告警
])
def test_b10_risk_consistency_dataset_behavior(
        tmp_path, monkeypatch, option, expected_flag):
    """B-10 行为验证：趋势一致性筛选收到的是哪个数据集。"""
    import warnings as _warnings

    from syriskmodels.contrib import build_scorecard as _bs_module

    captured = {}

    def fake_rtc(df, sc_bins, target):
        captured['df'] = df
        return {v: 1.0 for v in sc_bins}

    monkeypatch.setattr(_bs_module, 'risk_trends_consistency', fake_rtc)
    monkeypatch.chdir(tmp_path)

    kwargs = {} if option is None else {'risk_consistency_dataset': option}
    sample_df = _b10_sample_df()

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter('always')
        _bs_module.build_scorecard(
            sample_df,
            features=['f1', 'f2', 'f3'],
            target='Class',
            train_filter=lambda x: x['Time'] <= 140000,
            oot_filter=lambda x: x['Time'] > 140000,
            output_excel_file=str(tmp_path / 'sc.xlsx'),
            cv=2,
            binning_kwargs={'no_cores': 1},
            **kwargs,
        )

    assert 'df' in captured, 'risk_trends_consistency 未被调用'
    flags = captured['df']['_train_test_flag_'].unique().tolist()
    assert flags == [expected_flag], (
        f'趋势一致性筛选应只使用 {expected_flag} 数据，实际: {flags}'
    )

    if option == 'oot':
        b10_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning) and 'OOT' in str(w.message)
        ]
        assert b10_warnings, "显式选择 'oot' 时必须给出明确的 UserWarning"


def test_b10_risk_consistency_falls_back_to_train_when_no_valid(
        tmp_path, monkeypatch):
    """B-10 边界：random_test_set=0（无验证集）时回退训练集。"""
    from syriskmodels.contrib import build_scorecard as _bs_module

    captured = {}

    def fake_rtc(df, sc_bins, target):
        captured['df'] = df
        return {v: 1.0 for v in sc_bins}

    monkeypatch.setattr(_bs_module, 'risk_trends_consistency', fake_rtc)
    monkeypatch.chdir(tmp_path)

    _bs_module.build_scorecard(
        _b10_sample_df(),
        features=['f1', 'f2', 'f3'],
        target='Class',
        train_filter=lambda x: x['Time'] <= 140000,
        oot_filter=lambda x: x['Time'] > 140000,
        output_excel_file=str(tmp_path / 'sc.xlsx'),
        cv=2,
        random_test_set=0,
        binning_kwargs={'no_cores': 1},
    )

    flags = captured['df']['_train_test_flag_'].unique().tolist()
    assert flags == ['01_train'], (
        f'无验证集时应回退训练集，实际: {flags}'
    )


# --------------------------------------------------------------------------- #
# B-11 multiprocessing 默认路径：picklability 与 spawn 安全性
# --------------------------------------------------------------------------- #

def test_b11_woebin_returns_identical_result_for_single_core(synthetic_df):
    """B-11（护栏）：分箱器对象可 pickle，且 ``no_cores=1`` 结果确定。

    ``woebin`` / ``woebin_ply`` 历史上在 ``no_cores=None``（默认）时会自行
    计算核数并走 ``mp.Pool``（``api/woebin.py`` / ``api/transform.py``）。
    macOS / Windows 默认 spawn 语义，Pool 需要 pickle 任务与 binner 对象；
    同时小数据下并行与进程启动开销相比并无收益（见报告 B-11 的实测数据）。

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


def _b11_frame(n_vars: int = 6) -> pd.DataFrame:
    """B-11 用数据：变量数 >5，legacy 自动核数公式在默认路径会给出 >1。"""
    rng = np.random.default_rng(77)
    n = 500
    data = {f'v{i}': np.round(rng.normal(size=n), 6) for i in range(n_vars)}
    data['target'] = rng.binomial(1, 0.3, n)
    return pd.DataFrame(data)


def test_b11_default_is_serial(monkeypatch):
    """B-11（已修复）：默认不创建进程池 —— no_cores 缺省即串行。

    历史缺陷：``no_cores=None``（默认）时按 ``ceil(len(xs)/5)`` 自动开启
    ``mp.Pool``：小数据更慢（+45%），且 spawn 语义下 Jupyter/REPL/stdin
    会因 worker 无法导入 ``__main__`` 而**无限挂起**。

    W2 修复：``woebin`` / ``woebin_ply`` 默认 ``no_cores=1``；None/<1 一律
    视为 1（串行）；只有显式传入 ``no_cores>1`` 才启用并行。
    """
    import multiprocessing

    def _no_pool(*args, **kwargs):
        raise AssertionError('默认路径不应创建 multiprocessing 进程池')

    monkeypatch.setattr(multiprocessing, 'Pool', _no_pool)
    monkeypatch.setattr(multiprocessing, 'get_context', _no_pool)

    frame = _b11_frame()
    x = [f'v{i}' for i in range(6)]

    bins_default = woebin(frame, y='target', x=x, methods=['quantile', 'tree'],
                          initial_bins=20, bin_num_limit=5)
    bins_serial = woebin(frame, y='target', x=x, methods=['quantile', 'tree'],
                         initial_bins=20, bin_num_limit=5, no_cores=1)
    for v in x:
        pd.testing.assert_frame_equal(bins_default[v], bins_serial[v])

    from syriskmodels.scorecard import woebin_ply
    ply_default = woebin_ply(frame[x], bins_default, value='woe')
    ply_serial = woebin_ply(frame[x], bins_serial, value='woe', no_cores=1)
    pd.testing.assert_frame_equal(
        ply_default.sort_index(axis=1), ply_serial.sort_index(axis=1))


def test_b11_interactive_spawn_falls_back_to_serial(monkeypatch):
    """B-11（已修复）：交互式环境 + spawn 语义 → 自动回退串行并告警。

    spawn 需要 worker 重新导入 ``__main__``；Jupyter/REPL/管道 stdin 下
    ``__main__`` 不可导入，历史实现会**无限等待不返回**（实测 240s 未退出）。
    修复后：显式 ``no_cores>1`` 且（spawn 上下文 + 交互式环境）时，回退
    ``no_cores=1`` 串行执行并发出 ``UserWarning``，结果与串行一致。
    """
    import multiprocessing

    from syriskmodels import utils as sy_utils

    class _SpawnLikeCtx:
        def get_start_method(self):
            return 'spawn'

        def Pool(self, *args, **kwargs):
            raise AssertionError('交互式 spawn 环境不应创建进程池')

    monkeypatch.setattr(multiprocessing, 'get_context',
                        lambda *a, **k: _SpawnLikeCtx())
    monkeypatch.setattr(sy_utils, 'interactive_mode', lambda: True)

    frame = _b11_frame()
    x = [f'v{i}' for i in range(6)]

    with pytest.warns(UserWarning, match='交互式'):
        bins_par = woebin(frame, y='target', x=x,
                          methods=['quantile', 'tree'],
                          initial_bins=20, bin_num_limit=5, no_cores=2)
    bins_serial = woebin(frame, y='target', x=x, methods=['quantile', 'tree'],
                         initial_bins=20, bin_num_limit=5, no_cores=1)
    for v in x:
        pd.testing.assert_frame_equal(bins_par[v], bins_serial[v])


def test_b11_explicit_parallel_matches_serial():
    """B-11 验收：显式 ``no_cores=2`` 与 ``no_cores=1`` 结果完全一致。

    在非交互（pytest 脚本）环境下真实创建进程池执行；worker 异常必须
    传播回主进程（``starmap_async().get()``），进程池用 with 上下文管理清理。
    """
    frame = _b11_frame()
    x = [f'v{i}' for i in range(6)]

    bins_serial = woebin(frame, y='target', x=x, methods=['quantile', 'tree'],
                         initial_bins=20, bin_num_limit=5, no_cores=1)
    bins_parallel = woebin(frame, y='target', x=x, methods=['quantile', 'tree'],
                           initial_bins=20, bin_num_limit=5, no_cores=2)
    for v in x:
        pd.testing.assert_frame_equal(bins_parallel[v], bins_serial[v])

    from syriskmodels.scorecard import woebin_ply
    ply_serial = woebin_ply(frame[x], bins_serial, value='woe', no_cores=1)
    ply_parallel = woebin_ply(frame[x], bins_serial, value='woe', no_cores=2)
    pd.testing.assert_frame_equal(
        ply_parallel.sort_index(axis=1), ply_serial.sort_index(axis=1))


# --------------------------------------------------------------------------- #
# B-12 binning_helpers 与生产代码脱节（W2 处置：标记 deprecated + 兼容导出）
# --------------------------------------------------------------------------- #

def test_b12_binning_helpers_deprecated_and_not_used_by_production():
    """B-12（已处置）：``binning_helpers`` 标记 deprecated，生产路径不依赖。

    W2 决策（方案 b，见 W2 报告 §B-12）：不把 helper 接入生产路径 ——
    W2 性能内核重构正把粗分箱迁移到 BinCountTable/NumPy 实现，把
    DataFrame 形态的 helper 塞进热路径与重构方向相反；改为：

    1. 模块标记 ``__deprecated__``，6 个公开函数调用时触发
       ``DeprecationWarning``；
    2. ``scorecard/__init__.py`` 兼容导出保留（公共 API 不破坏）；
    3. 生产路径（core/bins/api）依旧**零引用**；
    4. ``test_binning_helpers.py`` 明确定位为"兼容层稳定性"测试。
    """
    import pathlib

    import pytest as _pytest

    from syriskmodels.scorecard.utils import binning_helpers as bh

    # 1. 弃用标记 + 调用告警
    assert getattr(bh, '__deprecated__', False) is True
    with _pytest.warns(DeprecationWarning):
        bh.compute_woe(np.array([80, 60]), np.array([20, 40]), 0.5)
    with _pytest.warns(DeprecationWarning):
        bh.extract_numeric_breaks(
            pd.DataFrame({'bin_chr': ['[-inf, 20)', '[20, inf)']}))

    # 2. 兼容导出保留
    import syriskmodels.scorecard as sc
    for name in ('extract_numeric_breaks', 'format_numeric_bin_names',
                 'extract_breaks_from_binning', 'compute_woe', 'compute_iv',
                 'merge_adjacent_bins'):
        assert hasattr(sc, name), f'兼容导出缺失: {name}'
        assert name in sc.__all__

    # 3. 生产路径零引用（仅 __init__.py 再导出）
    src = (pathlib.Path(__file__).resolve().parent.parent
           / 'src' / 'syriskmodels' / 'scorecard')
    referencing = sorted(
        str(path.relative_to(src))
        for path in src.rglob('*.py')
        if 'binning_helpers' in path.read_text(encoding='utf-8')
    )
    assert referencing == ['__init__.py', 'utils/binning_helpers.py'], (
        f'binning_helpers 的生产引用发生了变化：{referencing}；'
        f'请同步更新 W2 报告中的 B-12 记录'
    )


def test_b12_helper_woe_iv_semantics_consistent_with_production():
    """B-12（补充）：helper 与生产路径**重叠语义**必须一致。

    ``compute_woe`` / ``compute_iv``（epsilon=0.5）与生产路径
    ``WOEBin.binning_format`` 的 woe / total_iv 数学定义相同 ——
    这是两套实现唯一语义重叠的部分，用真实分箱结果钉住数值一致。
    ``extract_*`` 的 breaks 语义与生产路径**有意不同**（float 右边界 vs
    字符串），差异已在弃用说明中记录，本用例同时钉住两侧行为防漂移。
    """
    import pytest as _pytest

    from syriskmodels.scorecard.utils import binning_helpers as bh

    frame = _synthetic()
    bins = _bins(frame, ['num_a', 'cat_a'])

    for var in ('num_a', 'cat_a'):
        res = bins[var]
        good = res['good'].to_numpy()
        bad = res['bad'].to_numpy()
        eps = 0.5

        with _pytest.warns(DeprecationWarning):
            woe = bh.compute_woe(good, bad, epsilon=eps)
        assert np.allclose(woe, res['woe'].to_numpy(), rtol=0, atol=1e-12)

        sub0 = lambda a: np.where(a == 0, eps, a)  # noqa: E731
        with _pytest.warns(DeprecationWarning):
            iv = bh.compute_iv(woe, sub0(good), sub0(bad))
        assert iv == _pytest.approx(float(res['total_iv'].iloc[0]), rel=0,
                                    abs=1e-12)

    # breaks 语义差异（弃用说明的一部分）：helper 返回 float 右边界，
    # 生产路径返回字符串（数值型为右边界字符串，类别型为分箱名）
    numeric_binning = pd.DataFrame({
        'bin_chr': ['[-inf, 20)', '[20, 40)', '[40, inf)'],
    })
    with _pytest.warns(DeprecationWarning):
        helper_breaks = list(
            bh.extract_breaks_from_binning(numeric_binning, is_numeric=True))
    assert helper_breaks == [20.0, 40.0, np.inf]
    assert all(isinstance(b, float) for b in helper_breaks)

    production_breaks = bins['num_a']['breaks'].tolist()
    assert all(isinstance(b, str) for b in production_breaks), (
        '生产路径数值型 breaks 应为右边界字符串，语义已变化，请复核 B-12 记录'
    )


# --------------------------------------------------------------------------- #
# B-13 test_scorecard/conftest.py fixture 定义缺陷（W2 已修复）
# --------------------------------------------------------------------------- #

def test_b13_conftest_fixtures_use_dependency_injection():
    """B-13（已修复）：test_scorecard/conftest.py 数据 fixture 依赖注入且有消费者。

    历史缺陷：``data_with_constant_var`` 等 5 个 fixture 直接调用
    ``clean_data()``（fixture 函数的普通调用返回包装对象而非 DataFrame），
    一旦被使用必然 ``AttributeError``；且当时**没有任何测试消费这些
    fixture**，缺陷一直潜伏。

    W2 修复：
    1. 5 个 fixture 全部改为依赖注入形式（``def ...(clean_data): ...``）；
       ``data_with_mixed_types`` 另需先 ``astype(object)``（pandas 3 的
       ``str`` dtype 不允许写入非字符串值，fixture 原本在 pandas 3 下
       自身即抛 TypeError —— 同类潜伏缺陷）；
    2. 新增 ``test/test_scorecard/test_conftest_fixtures.py`` 真实消费
       全部数据类 fixture（含 dtm_* 与 binning_result_*）；
    3. 本用例静态钉住两个不变量：① fixture 签名含 ``clean_data``；
       ② 每个 fixture 至少被一个测试函数消费。
    """
    import ast
    import inspect
    import pathlib

    from test.test_scorecard import conftest as sc_conftest

    fixture_names = [
        'data_with_constant_var',
        'data_with_too_many_categories',
        'data_with_special_values',
        'data_with_mixed_types',
        'data_all_nan',
    ]

    # ① 依赖注入：fixture 原函数签名必须含 clean_data
    for name in fixture_names:
        fn = getattr(sc_conftest, name)
        wrapped = getattr(fn, '__wrapped__', fn)
        params = inspect.signature(wrapped).parameters
        assert 'clean_data' in params, (
            f'{name} 应通过依赖注入接收 clean_data fixture，'
            f'当前签名: {list(params)}'
        )

    # ② 至少一个测试真实消费每个 fixture
    test_dir = pathlib.Path(__file__).resolve().parent / 'test_scorecard'
    consumed = set()
    for path in sorted(test_dir.glob('test_*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name.startswith('test_')):
                consumed.update(a.arg for a in node.args.args)
    missing = [n for n in fixture_names if n not in consumed]
    assert not missing, f'以下 fixture 仍无测试消费（潜伏缺陷风险）: {missing}'

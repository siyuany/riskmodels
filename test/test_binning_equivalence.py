# -*- encoding: utf-8 -*-
"""分箱内核差分/等价测试（W2 Phase 3–5、7–8）。

结构
----
* ``Ref*`` 类：W1 develop（合并基线 ``372265c``）pandas 实现的**逐字参考
  拷贝**（仅测试用 oracle），自带旧版 ``binning``（_n0/_n1 lambda）、
  ``initial_binning``、搜索循环与 breaks 提取 —— 不依赖任何生产代码路径，
  生产侧任何改动都不会"污染" oracle；
* 差分断言：同一 dtm + 同一初始 breaks 下，生产实现与参考实现的 breaks
  **逐位相等**（``assert_series_equal`` 含 dtype），并经 ``__call__`` 全
  路径比对最终分箱 DataFrame（count/good/bad/woe/iv/breaks 全列）；
* 覆盖矩阵（任务书 §3.4/§4.5/§5.4）：数值/类别 × 缺失/特殊值 ×
  ib=20/100/500 × bin_num_limit=0/1/3/5/8 × count_distr_limit 变化 ×
  monotonic on/off × 并列候选（离散值制造 IV tie）× 空分箱（用户 breaks）
  × 全好/全坏段（epsilon 路径）× 对抗性 count_distr 边界（Kahan 和恰在
  limit 上）；固定种子，可复现。

Numba 链路（Phase 7）：``test_engine_numba_matches_numpy_*`` 用例在
numba 可用时对比 Numba 与 NumPy 参考内核输出，浮点 tie 不稳定时以
NumPy 参考实现为准（任务书 §7.4）。
"""
import re

import numpy as np
import pandas as pd
import pytest

from syriskmodels.scorecard import (
    ComposedWOEBin,
    QuantileInitBin,
    WOEBin,
    WOEBinFactory,
    woebin,
)
from syriskmodels.scorecard.bins.optimal import (
    ChiMergeOptimBin,
    RuleOptimBin,
    TreeOptimBin,
)
from syriskmodels.scorecard.core.base import OptimBinMixin

# ============================================================================ #
# W1 参考拷贝（oracle）—— 以下代码逐字取自 W1 develop，禁止"顺手修复"
# ============================================================================ #


class _LegacyBinningMixin:
    """W1 develop ``core/base.py::WOEBin.binning`` 的逐字拷贝。"""

    @classmethod
    def binning(cls, dtm: pd.DataFrame, bin_chr: pd.Series) -> pd.DataFrame:
        def _n0(x):
            return np.sum(x == 0)

        def _n1(x):
            return np.sum(x == 1)

        bin_chr = bin_chr.rename(index='bin_chr')
        binning = dtm.groupby(['variable', bin_chr], observed=False)['y'].agg(
            good=_n0, bad=_n1)
        binning = binning.reset_index()

        return binning


class _LegacyOptimBinMixin(OptimBinMixin):
    """initial_binning 与生产实现一致（W1 至今未变），仅为可读性显式命名。"""


class RefTreeOptimBin(_LegacyBinningMixin, WOEBin, _LegacyOptimBinMixin):
    """W1 develop ``bins/optimal.py::TreeOptimBin`` 的逐字参考拷贝。"""

    def __init__(self,
                 bin_num_limit: int = 5,
                 min_iv_inc: float = 0.05,
                 count_distr_limit: float = 0.02,
                 ensure_monotonic: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.bin_num_limit = bin_num_limit
        self.min_iv_inc = min_iv_inc
        self.count_distr_limit = count_distr_limit
        self.ensure_monotonic = ensure_monotonic

    def woebin(self, dtm, breaks=None):
        from syriskmodels.utils import monotonic

        assert breaks is not None, \
            f"使用{self.__class__.__name__}类进行分箱，需要传入初始分箱（细分箱）结果"
        binning_tree = self.initial_binning(dtm, breaks)
        binning_tree['node_id'] = 0
        binning_tree['cp'] = False  # cut point flag
        binning_tree.loc[len(binning_tree) - 1, 'cp'] = True

        last_iv = 0

        while len(binning_tree['node_id'].unique()) <= self.bin_num_limit:
            cut_idx_iv = {}
            for idx in binning_tree.index[~binning_tree['cp']]:
                new_node_ids = self.node_split(
                    binning_tree['node_id'], idx)
                new_binning = self.merge_binning(
                    binning_tree, new_node_ids)
                if self.ensure_monotonic:
                    monotonic_type = monotonic(new_binning['bad_prob'])
                    if monotonic_type in ('increasing', 'decreasing'):
                        monotonic_constrain = True
                    else:
                        monotonic_constrain = False
                else:
                    monotonic_constrain = True

                if (np.all(
                        new_binning['count_distr'] > self.count_distr_limit
                    ) and monotonic_constrain):
                    curr_iv = new_binning['total_iv'].iloc[0]
                    if ((curr_iv - last_iv + 1e-8) /
                            (last_iv + 1e-8)) > self.min_iv_inc:
                        cut_idx_iv[idx] = curr_iv

            if len(cut_idx_iv) > 0:
                sorted_cut_idx_iv = sorted(
                    cut_idx_iv.items(), key=lambda x: -x[1])
                best_cut_idx = sorted_cut_idx_iv[0][0]
                last_iv = sorted_cut_idx_iv[0][1]
                binning_tree['node_id'] = self.node_split(
                    binning_tree['node_id'], best_cut_idx)
                binning_tree.loc[best_cut_idx, 'cp'] = True
            else:
                break

        best_binning = self.merge_binning(
            binning_tree, binning_tree['node_id'])

        if pd.api.types.is_numeric_dtype(dtm['value']):
            best_binning['bin_chr'] = best_binning['bin_chr'].apply(
                lambda x: re.sub(r',[.\d]+\)%,%\[[.\d]+,', ',', x))
            _pattern = re.compile(r"^\[(.*), *(.*)\)")
            breaks = best_binning['bin_chr'].apply(
                lambda x: _pattern.match(x)[2])
            breaks = pd.to_numeric(breaks)
        else:
            breaks = best_binning['bin_chr']

        return breaks

    def merge_binning(self, binning, node_ids):
        # yapf: disable
        new_binning = binning.groupby([
            'variable',
            node_ids,
        ]).agg(
            bin_chr=('bin_chr', lambda x: '%,%'.join(x.tolist())),
            count=('count', 'sum'),
            count_distr=('count_distr', 'sum'),
            good=('good', 'sum'),
            bad=('bad', 'sum')
        ).assign(
            bad_prob=lambda x: x['bad'] / x['count'],
            total_iv=lambda x: self.iv(x['good'], x['bad']))
        # yapf: enable

        return new_binning

    @staticmethod
    def node_split(node_ids, idx):
        new_node_ids = np.where(
            node_ids.index <= idx, node_ids, node_ids + 1)
        return new_node_ids

    def iv(self, good, bad):
        good = np.asarray(good)
        bad = np.asarray(bad)
        # substitute 0 by self.epsilon
        good = np.where(good == 0, self.epsilon, good)
        bad = np.where(bad == 0, self.epsilon, bad)
        good_distr = good / good.sum()
        bad_distr = bad / bad.sum()
        iv = (good_distr - bad_distr) * np.log(good_distr / bad_distr)
        return iv.sum()


class RefChiMergeOptimBin(_LegacyBinningMixin, WOEBin, _LegacyOptimBinMixin):
    """W1 develop ``bins/optimal.py::ChiMergeOptimBin`` 的逐字参考拷贝。"""

    def __init__(self,
                 bin_num_limit: int = 5,
                 p: float = 0.05,
                 count_distr_limit: float = 0.02,
                 ensure_monotonic: bool = False,
                 **kwargs):
        from scipy.stats import chi2 as _chi2_dist
        super().__init__(**kwargs)
        self.bin_num_limit = bin_num_limit
        self.p = p
        self.count_distr_limit = count_distr_limit
        self.ensure_monotonic = ensure_monotonic
        self.chi2_limit = _chi2_dist.isf(p, df=1)

    @staticmethod
    def chi2_stat(binning):
        from scipy.stats import chi2_contingency

        binning['good_lag'] = binning['good'].shift(1)
        binning['bad_lag'] = binning['bad'].shift(1)

        def chi2_cont_tbl(arr):
            if np.any(np.isnan(arr)):
                return np.nan
            elif np.any(np.sum(arr, axis=1) == 0) or np.any(np.sum(arr, axis=0) == 0):
                return 0.0
            else:
                return chi2_contingency(arr, correction=True)[0]

        binning['chi2'] = binning.apply(
            lambda x: chi2_cont_tbl([[x['good'], x['bad']],
                                      [x['good_lag'], x['bad_lag']]]),
            axis=1
        )
        del binning['good_lag']
        del binning['bad_lag']

        return binning

    def woebin(self, dtm, breaks=None):
        assert breaks is not None, \
            f"使用{self.__class__.__name__}类进行分箱，需要传入初始分箱（细分箱）结果"

        binning = self.initial_binning(dtm, breaks)
        binning_chi2 = self.chi2_stat(binning)
        binning_chi2['bin_chr'] = binning_chi2['bin_chr'].astype('str')

        # Start merge loop
        while True:
            min_chi2 = binning_chi2['chi2'].min()
            min_count_distr = binning_chi2['count_distr'].min()
            n_bins = len(binning_chi2)

            if min_chi2 < self.chi2_limit:
                # 分箱坏占比差异不显著
                idx = binning_chi2[binning_chi2['chi2'] == min_chi2].index[0]
            elif min_count_distr < self.count_distr_limit:
                # 分箱占比过少
                idx = binning_chi2[
                    binning_chi2['count_distr'] == min_count_distr
                ].index[0]
                if idx == 0 or (idx < len(binning_chi2) - 1 and
                               (binning_chi2['chi2'][idx]
                                > binning_chi2['chi2'][idx + 1])):
                    idx = idx + 1
            elif n_bins > self.bin_num_limit:
                # 分箱数太多
                idx = binning_chi2[binning_chi2['chi2'] == min_chi2].index[0]
            else:
                # 结束合并操作
                break

            # 合并分箱
            binning_chi2.loc[idx - 1, 'bin_chr'] = '%,%'.join([
                binning_chi2.loc[idx - 1, 'bin_chr'],
                binning_chi2.loc[idx, 'bin_chr']
            ])
            binning_chi2.loc[idx - 1, 'count'] = (
                binning_chi2.loc[idx - 1, 'count'] +
                binning_chi2.loc[idx, 'count'])
            binning_chi2.loc[idx - 1, 'count_distr'] = (
                binning_chi2.loc[idx - 1, 'count_distr'] +
                binning_chi2.loc[idx, 'count_distr'])
            binning_chi2.loc[idx - 1, 'good'] = (
                binning_chi2.loc[idx - 1, 'good'] +
                binning_chi2.loc[idx, 'good'])
            binning_chi2.loc[idx - 1, 'bad'] = (
                binning_chi2.loc[idx - 1, 'bad'] +
                binning_chi2.loc[idx, 'bad'])

            if pd.api.types.is_numeric_dtype(dtm['value']):
                # 数值类型分箱合并: [a,b)%,%[b,c) -> [a,c)
                binning_chi2['bin_chr'] = binning_chi2['bin_chr'].apply(
                    lambda x: re.sub(r',[.\d]+\)%,%\[[.\d]+,', ',', x))

            index = binning_chi2.index.tolist()
            index.remove(idx)
            binning_chi2 = binning_chi2.iloc[
                index,
            ].reset_index(drop=True)
            binning_chi2 = self.chi2_stat(binning_chi2)
        # End of loop

        # 切分点提取
        if pd.api.types.is_numeric_dtype(dtm['value']):
            _pattern = re.compile(r"^\[(.*), *(.*)\)")
            breaks = binning_chi2['bin_chr'].apply(
                lambda x: _pattern.match(x)[2])
            breaks = pd.to_numeric(breaks)
        else:
            breaks = binning_chi2['bin_chr']

        return breaks


# ============================================================================ #
# 数据生成（固定种子；覆盖任务书要求的边界）
# ============================================================================ #

def _dtm(value, y, variable='x'):
    return pd.DataFrame({'variable': variable, 'y': y, 'value': value})


def _numeric_cases():
    """数值型 dtm 场景集合：(名称, dtm 构造参数)。"""
    cases = {}

    rng = np.random.default_rng(101)
    n = 800
    v = np.round(rng.normal(size=n), 6)
    y = rng.binomial(1, 1 / (1 + np.exp(-(1.3 * v))))
    cases['signal'] = _dtm(v, y)

    # 大量并列值（离散化 → IV tie / 相同候选）
    rng = np.random.default_rng(102)
    v = rng.integers(0, 9, n).astype(float)
    y = rng.binomial(1, np.where(v >= 6, 0.7, 0.15))
    cases['ties_discrete'] = _dtm(v, y)

    # 纯噪声（iv≈0，接受条件走 last_iv=0 路径）
    rng = np.random.default_rng(103)
    cases['noise'] = _dtm(np.round(rng.normal(size=n), 6),
                          rng.binomial(1, 0.3, n))

    # 缺失 + 哨兵值（-999 走 special_values；NaN 走隐式 missing）
    rng = np.random.default_rng(104)
    v = np.round(rng.normal(size=n), 6)
    v[50:90] = -999.0
    v[100:130] = np.nan
    y = rng.binomial(1, 0.25, n)
    cases['missing_sentinel'] = _dtm(v, y)

    # 长尾（分位数极不均匀 → count_distr 边界）
    rng = np.random.default_rng(105)
    v = np.round(rng.lognormal(0, 2.5, n), 6)
    y = rng.binomial(1, 0.2, n)
    cases['lognormal'] = _dtm(v, y)

    # 全坏段风险：y 与阈值完全可分
    rng = np.random.default_rng(106)
    v = np.round(rng.uniform(-3, 3, n), 6)
    y = (v > 0).astype(int)
    cases['separable'] = _dtm(v, y)

    # 对抗性 count_distr 边界：n=500，10 个等宽离散值 → 每箱占比恰 0.02 附近
    rng = np.random.default_rng(107)
    v = (rng.integers(0, 10, 500) * 1.0)
    # 强制每个值恰好 50 个（占比恰为 0.1；两段合并占比恰为 0.02*10 组合）
    v = np.repeat(np.arange(10, dtype=float), 50)
    y = rng.binomial(1, 0.3, 500)
    cases['exact_ratio_bounds'] = _dtm(v, y)

    return cases


def _categorical_cases():
    cases = {}

    rng = np.random.default_rng(201)
    n = 900
    cats = rng.choice(['a', 'b', 'c', 'd', 'e', 'f', 'g'], n)
    p = pd.Series({'a': .1, 'b': .2, 'c': .3, 'd': .4, 'e': .5,
                   'f': .6, 'g': .7})
    y = rng.binomial(1, pd.Series(cats).map(p).to_numpy())
    cases['ordered_risk'] = _dtm(cats, y)

    # 大量类别 + 缺失
    rng = np.random.default_rng(202)
    cats = rng.choice([f'c{i}' for i in range(25)], n).astype(object)
    cats[rng.choice(n, 60, replace=False)] = np.nan
    y = rng.binomial(1, 0.35, n)
    cases['many_cats_missing'] = _dtm(cats, y)

    # 非单调 badprob（monotonic 约束生效）
    rng = np.random.default_rng(203)
    cats = rng.choice(['a', 'b', 'c', 'd', 'e'], n)
    p = pd.Series({'a': .1, 'b': .6, 'c': .2, 'd': .7, 'e': .3})
    y = rng.binomial(1, pd.Series(cats).map(p).to_numpy())
    cases['non_monotonic'] = _dtm(cats, y)

    # 并列 badprob（排序不稳定路径 + IV tie）
    v = np.array(['a', 'b'] * 400 + ['c', 'd'] * 50, dtype=object)
    y = np.tile([0, 1], 400 + 50)
    cases['tied_badprob'] = _dtm(v, y)

    return cases


def _all_dtm_cases():
    out = {}
    for name, df in _numeric_cases().items():
        out[f'num_{name}'] = df
    for name, df in _categorical_cases().items():
        out[f'cat_{name}'] = df
    return out


_CASE_IDS = sorted(_all_dtm_cases())


@pytest.fixture(scope='module')
def dtm_cases():
    return _all_dtm_cases()


# ============================================================================ #
# 差分工具
# ============================================================================ #

def _initial_breaks(dtm, initial_bins):
    """构造初始细分箱切分点。

    与生产路径保持同一不变量：``woebin(__call__)`` 会先把 NaN 拆入
    'missing' 特殊值箱，``QuantileInitBin`` 只见到无 NaN 的 ``dtm_ns``
    （直接对含 NaN 的 object 列跑 np.unique 会因 str/float 比较抛
    TypeError，含 NaN 的 float 列会得到 NaN 分位点）。
    """
    q = QuantileInitBin(initial_bins=initial_bins)
    return q.woebin(dtm[dtm['value'].notna()])


def _drop_nan(dtm):
    """树内核差分统一使用无 NaN 数据（生产路径的 dtm_ns 不变量）。"""
    return dtm[dtm['value'].notna()].reset_index(drop=True)


def _assert_breaks_identical(prod_breaks, ref_breaks):
    """breaks 逐位相等（值 + dtype + categories；索引无关）。

    dtype/categories 也属可观测行为：legacy 的类别型 breaks 在"无合并段"
    时为 category dtype（categories = 初始 breaks 顺序），下游
    ``set_categories`` 会采用 categories 而非 values 决定最终分箱行序
    （golden germancredit_quantile_tree_ib20_limit5 曾因此漂移）。
    """
    prod = pd.Series(prod_breaks).reset_index(drop=True)
    ref = pd.Series(ref_breaks).reset_index(drop=True)
    assert str(prod.dtype) == str(ref.dtype), (
        f'breaks dtype 不一致: prod={prod.dtype} ref={ref.dtype}')
    if str(ref.dtype) == 'category':
        assert prod.cat.categories.tolist() == ref.cat.categories.tolist(), (
            f'categories 不一致: prod={prod.cat.categories.tolist()} '
            f'ref={ref.cat.categories.tolist()}')
        assert bool(prod.cat.ordered) == bool(ref.cat.ordered)
        assert prod.tolist() == ref.tolist()
    elif pd.api.types.is_float_dtype(ref) or pd.api.types.is_integer_dtype(ref):
        prod_f = prod.astype('float64')
        ref_f = ref.astype('float64')
        np.testing.assert_array_equal(prod_f.to_numpy(), ref_f.to_numpy())
    else:
        assert prod.astype(str).tolist() == ref.astype(str).tolist(), (
            f'breaks 不一致:\nprod={prod}\nref={ref}')


def _run_tree_pair(dtm, breaks, **tree_kwargs):
    prod = TreeOptimBin(**tree_kwargs)
    ref = RefTreeOptimBin(**tree_kwargs)
    b_prod = prod.woebin(dtm, breaks)
    b_ref = ref.woebin(dtm, breaks)
    _assert_breaks_identical(b_prod, b_ref)
    return b_prod


# ============================================================================ #
# Tree：生产 NumPy 内核 vs W1 参考拷贝
# ============================================================================ #

TREE_PARAMS = [
    dict(bin_num_limit=bnl, count_distr_limit=cdl, ensure_monotonic=mono,
         min_iv_inc=mii, initial_bins=ib)
    for bnl in (3, 5)
    for cdl in (0.02, 0.0)
    for mono in (False, True)
    for mii in (0.05,)
    for ib in (20,)
]
# 追加专项组合：ib=100/500、limit 边界、min_iv_inc 变化、eps 变化
# ib=500 的参考拷贝为 O(k²) pandas 实现，单用例秒级 → 归入 slow marker
# （CI integration job 覆盖），保持 unit 套件时长可控。
TREE_PARAMS += [
    dict(bin_num_limit=0, count_distr_limit=0.02, ensure_monotonic=False,
         min_iv_inc=0.05, initial_bins=20),
    dict(bin_num_limit=1, count_distr_limit=0.05, ensure_monotonic=True,
         min_iv_inc=0.05, initial_bins=20),
    dict(bin_num_limit=8, count_distr_limit=0.0, ensure_monotonic=False,
         min_iv_inc=0.0, initial_bins=100),
    dict(bin_num_limit=5, count_distr_limit=0.2, ensure_monotonic=False,
         min_iv_inc=0.5, initial_bins=100),
    pytest.param(
        dict(bin_num_limit=3, count_distr_limit=0.02, ensure_monotonic=True,
             min_iv_inc=0.05, initial_bins=500, eps=1e-8),
        marks=pytest.mark.slow),
    pytest.param(
        dict(bin_num_limit=5, count_distr_limit=0.02, ensure_monotonic=False,
             min_iv_inc=0.05, initial_bins=500),
        marks=pytest.mark.slow),
]


def _tree_param_id(p):
    d = p if isinstance(p, dict) else p.values[0]
    return (f"lim{d['bin_num_limit']}_cdl{d['count_distr_limit']}"
            f"_mono{int(d['ensure_monotonic'])}_mii{d['min_iv_inc']}"
            f"_ib{d['initial_bins']}")


@pytest.mark.parametrize('case_id', _CASE_IDS)
@pytest.mark.parametrize('params', TREE_PARAMS,
                         ids=[_tree_param_id(p) for p in TREE_PARAMS])
def test_tree_kernel_matches_reference(dtm_cases, case_id, params):
    """Tree NumPy 内核与 W1 参考拷贝的差分（breaks 逐位相等）。"""
    params = dict(params)
    ib = params.pop('initial_bins')
    dtm = _drop_nan(dtm_cases[case_id])
    breaks = _initial_breaks(dtm, ib)
    if len(breaks) < 2:
        pytest.skip('初始分箱不足 2 个，无切分空间')
    _run_tree_pair(dtm, breaks, **params)


@pytest.mark.parametrize('case_id', _CASE_IDS)
def test_tree_full_call_matches_reference(dtm_cases, case_id):
    """全路径差分：woebin(__call__) 最终 DataFrame 全列一致（含特殊值路径）。"""
    dtm = dtm_cases[case_id]
    kwargs = dict(bin_num_limit=4, count_distr_limit=0.02,
                  ensure_monotonic=False, min_iv_inc=0.05)
    special = None
    if case_id == 'num_missing_sentinel':
        special = ['-999']

    prod = ComposedWOEBin(
        [QuantileInitBin(initial_bins=20), TreeOptimBin(**kwargs)])
    ref = ComposedWOEBin(
        [QuantileInitBin(initial_bins=20), RefTreeOptimBin(**kwargs)])

    res_prod = prod(dtm.copy(), special_values=special)
    res_ref = ref(dtm.copy(), special_values=special)

    if isinstance(res_ref, str):
        assert res_prod == res_ref
        return
    pd.testing.assert_frame_equal(res_prod, res_ref)


def test_tree_user_breaks_with_empty_bins():
    """用户指定 breaks 制造空分箱（count=0 → bad_prob NaN、distr=0）。"""
    rng = np.random.default_rng(301)
    n = 400
    v = np.round(rng.uniform(0, 1, n), 6)     # 数据只在 [0,1]
    y = rng.binomial(1, 0.3, n)
    dtm = _dtm(v, y)
    breaks = [-np.inf, -10.0, -5.0, 0.25, 0.5, 0.75, 5.0, 10.0, np.inf]

    for mono in (False, True):
        _run_tree_pair(dtm, breaks, bin_num_limit=5, count_distr_limit=0.0,
                       ensure_monotonic=mono)
        # count_distr_limit > 0 时空箱段必被拒绝
        _run_tree_pair(dtm, breaks, bin_num_limit=5, count_distr_limit=0.01,
                       ensure_monotonic=mono)


def test_tree_categorical_special_values_full_path():
    """类别 + 组合特殊值：sv 拆分与内核路径端到端一致。"""
    rng = np.random.default_rng(302)
    n = 600
    cats = rng.choice(['a', 'b', 'c', 'd', 'e', 'zz'], n).astype(object)
    cats[rng.choice(n, 40, replace=False)] = np.nan
    y = rng.binomial(1, 0.3, n)
    dtm = _dtm(cats, y)

    kwargs = dict(bin_num_limit=3, count_distr_limit=0.05)
    prod = ComposedWOEBin(
        [QuantileInitBin(initial_bins=20), TreeOptimBin(**kwargs)])
    ref = ComposedWOEBin(
        [QuantileInitBin(initial_bins=20), RefTreeOptimBin(**kwargs)])
    sv = ['zz%,%a', 'missing']
    pd.testing.assert_frame_equal(
        prod(dtm.copy(), special_values=sv),
        ref(dtm.copy(), special_values=sv))


def test_tree_woebin_api_end_to_end_identical():
    """公共 API 层面：methods=['quantile','tree'] 新旧实现输出一致。"""
    rng = np.random.default_rng(303)
    n = 700
    frame = pd.DataFrame({
        'v1': np.round(rng.normal(size=n), 6),
        'v2': rng.choice(['a', 'b', 'c', 'd'], n),
        'target': rng.binomial(1, 0.3, n),
    })
    frame.loc[10:25, 'v1'] = np.nan

    bins_prod = woebin(frame, y='target', x=['v1', 'v2'],
                       methods=['quantile', 'tree'],
                       initial_bins=20, bin_num_limit=5, no_cores=1)
    bins_ref = woebin(frame, y='target', x=['v1', 'v2'],
                      methods=[QuantileInitBin(initial_bins=20),
                               RefTreeOptimBin(bin_num_limit=5)],
                      no_cores=1)
    for v in ('v1', 'v2'):
        pd.testing.assert_frame_equal(bins_prod[v], bins_ref[v])


def test_tree_categorical_no_merge_keeps_legacy_category_dtype():
    """无合并段（每段=单一初始类别）：legacy breaks 为 category dtype。

    此时下游 ``set_categories`` 采用其 **categories**（初始 breaks 的
    字典序）而非 values（badprob 段序）决定最终分箱行序 —— W2 Phase 3
    开发中 golden ``germancredit_quantile_tree_ib20_limit5`` 正是因该
    pandas 语义漂移而被差分测试捕获。
    """
    rng = np.random.default_rng(401)
    n = 300
    cats = rng.choice(['alpha', 'beta', 'gamma'], n)
    p = pd.Series({'alpha': .2, 'beta': .5, 'gamma': .8})
    y = rng.binomial(1, pd.Series(cats).map(p).to_numpy())
    dtm = _dtm(cats, y)
    breaks0 = _initial_breaks(dtm, 20)

    kwargs = dict(bin_num_limit=5, min_iv_inc=0.0, count_distr_limit=0.0)
    prod = TreeOptimBin(**kwargs)
    ref = RefTreeOptimBin(**kwargs)
    b_prod = prod.woebin(dtm, breaks0)
    b_ref = ref.woebin(dtm, breaks0)

    # 命中"无合并"分支：参考实现必须产出 category dtype
    assert str(b_ref.dtype) == 'category', (
        f'测试前提不成立：参考实现 breaks dtype={b_ref.dtype}')
    _assert_breaks_identical(b_prod, b_ref)

    # 全路径：最终 DataFrame（含行序 = categories 字典序）一致
    prod_c = ComposedWOEBin([QuantileInitBin(initial_bins=20),
                             TreeOptimBin(**kwargs)])
    ref_c = ComposedWOEBin([QuantileInitBin(initial_bins=20),
                            RefTreeOptimBin(**kwargs)])
    out_prod = prod_c(dtm.copy())
    pd.testing.assert_frame_equal(out_prod, ref_c(dtm.copy()))
    assert out_prod['bin'].tolist() == ['alpha', 'beta', 'gamma']


def test_tree_germancredit_categorical_matches_reference():
    """golden 漂移场景的直接回归：germancredit 类别变量逐一差分。"""
    from test.conftest import GERMANCREDIT_FILE, require_data
    require_data(GERMANCREDIT_FILE)
    from syriskmodels.datasets import load_germancredit

    df = load_germancredit()
    cols = [
        'foreign.worker',
        'other.debtors.or.guarantors',
        'status.of.existing.checking.account',
        'purpose',
    ]
    param_sets = [
        dict(bin_num_limit=5),
        dict(bin_num_limit=8, min_iv_inc=0.0, count_distr_limit=0.0),
        dict(bin_num_limit=3, count_distr_limit=0.05, ensure_monotonic=True),
    ]
    for col in cols:
        dtm = pd.DataFrame({
            'variable': col,
            'y': df['creditability'],
            'value': df[col],
        })
        breaks0 = _initial_breaks(dtm, 20)
        for kwargs in param_sets:
            prod = TreeOptimBin(**kwargs)
            ref = RefTreeOptimBin(**kwargs)
            _assert_breaks_identical(prod.woebin(dtm, breaks0),
                                     ref.woebin(dtm, breaks0))
            prod_c = ComposedWOEBin(
                [QuantileInitBin(initial_bins=20), TreeOptimBin(**kwargs)])
            ref_c = ComposedWOEBin(
                [QuantileInitBin(initial_bins=20), RefTreeOptimBin(**kwargs)])
            pd.testing.assert_frame_equal(prod_c(dtm.copy()),
                                          ref_c(dtm.copy()))


# ============================================================================ #
# ChiMerge：生产 NumPy 内核 vs W1 参考拷贝
# ============================================================================ #

def _run_chi2_pair(dtm, breaks, **chi2_kwargs):
    prod = ChiMergeOptimBin(**chi2_kwargs)
    ref = RefChiMergeOptimBin(**chi2_kwargs)
    b_prod = prod.woebin(dtm, breaks)
    b_ref = ref.woebin(dtm, breaks)
    _assert_breaks_identical(b_prod, b_ref)
    return b_prod


# 参考实现每轮对全部相邻对调用 scipy（O(k²)），ib≥100 的组合归入 slow
CHI2_PARAMS_UNIT = [
    dict(bin_num_limit=5, count_distr_limit=0.02, p=0.05, initial_bins=20),
    dict(bin_num_limit=3, count_distr_limit=0.05, p=0.5, initial_bins=20),
    dict(bin_num_limit=1, count_distr_limit=0.02, p=0.9, initial_bins=20),
    dict(bin_num_limit=8, count_distr_limit=0.0, p=0.05, initial_bins=20),
]
CHI2_PARAMS_SLOW = [
    dict(bin_num_limit=5, count_distr_limit=0.02, p=0.05, initial_bins=100),
    dict(bin_num_limit=8, count_distr_limit=0.0, p=0.5, initial_bins=100),
    dict(bin_num_limit=5, count_distr_limit=0.2, p=0.05, initial_bins=100),
    dict(bin_num_limit=5, count_distr_limit=0.02, p=0.05, initial_bins=500),
]


def _chi2_id(p):
    return (f"lim{p['bin_num_limit']}_cdl{p['count_distr_limit']}"
            f"_p{p['p']}_ib{p['initial_bins']}")


@pytest.mark.parametrize('case_id', _CASE_IDS)
@pytest.mark.parametrize('params', CHI2_PARAMS_UNIT,
                         ids=[_chi2_id(p) for p in CHI2_PARAMS_UNIT])
def test_chi2_kernel_matches_reference(dtm_cases, case_id, params):
    """ChiMerge NumPy 内核与 W1 参考拷贝的差分（breaks 逐位相等）。"""
    params = dict(params)
    ib = params.pop('initial_bins')
    dtm = _drop_nan(dtm_cases[case_id])
    breaks = _initial_breaks(dtm, ib)
    if len(breaks) < 2:
        pytest.skip('初始分箱不足 2 个，无合并空间')
    _run_chi2_pair(dtm, breaks, **params)


@pytest.mark.slow
@pytest.mark.parametrize('case_id', _CASE_IDS)
@pytest.mark.parametrize('params', CHI2_PARAMS_SLOW,
                         ids=[_chi2_id(p) for p in CHI2_PARAMS_SLOW])
def test_chi2_kernel_matches_reference_heavy(dtm_cases, case_id, params):
    """ChiMerge 差分重用例（ib=100/500；参考实现 O(k²) scipy 调用）。"""
    params = dict(params)
    ib = params.pop('initial_bins')
    dtm = _drop_nan(dtm_cases[case_id])
    breaks = _initial_breaks(dtm, ib)
    if len(breaks) < 2:
        pytest.skip('初始分箱不足 2 个，无合并空间')
    _run_chi2_pair(dtm, breaks, **params)


@pytest.mark.parametrize('case_id', _CASE_IDS)
def test_chi2_full_call_matches_reference(dtm_cases, case_id):
    """全路径差分：chi2 的 woebin(__call__) 最终 DataFrame 全列一致。"""
    dtm = dtm_cases[case_id]
    kwargs = dict(bin_num_limit=4, count_distr_limit=0.02, p=0.05)
    special = ['-999'] if case_id == 'num_missing_sentinel' else None

    prod = ComposedWOEBin(
        [QuantileInitBin(initial_bins=20), ChiMergeOptimBin(**kwargs)])
    ref = ComposedWOEBin(
        [QuantileInitBin(initial_bins=20), RefChiMergeOptimBin(**kwargs)])

    res_prod = prod(dtm.copy(), special_values=special)
    res_ref = ref(dtm.copy(), special_values=special)

    if isinstance(res_ref, str):
        assert res_prod == res_ref
        return
    pd.testing.assert_frame_equal(res_prod, res_ref)


def test_chi2_germancredit_categorical_matches_reference():
    """germancredit 类别变量 ChiMerge 差分（golden 场景的直接回归）。"""
    from test.conftest import GERMANCREDIT_FILE, require_data
    require_data(GERMANCREDIT_FILE)
    from syriskmodels.datasets import load_germancredit

    df = load_germancredit()
    cols = [
        'foreign.worker',
        'other.debtors.or.guarantors',
        'status.of.existing.checking.account',
        'purpose',
    ]
    param_sets = [
        dict(bin_num_limit=5),
        dict(bin_num_limit=8, count_distr_limit=0.0),
        dict(bin_num_limit=2, p=0.5, count_distr_limit=0.05),
    ]
    for col in cols:
        dtm = pd.DataFrame({
            'variable': col,
            'y': df['creditability'],
            'value': df[col],
        })
        breaks0 = _initial_breaks(dtm, 20)
        for kwargs in param_sets:
            prod = ChiMergeOptimBin(**kwargs)
            ref = RefChiMergeOptimBin(**kwargs)
            _assert_breaks_identical(prod.woebin(dtm, breaks0),
                                     ref.woebin(dtm, breaks0))
            prod_c = ComposedWOEBin(
                [QuantileInitBin(initial_bins=20), ChiMergeOptimBin(**kwargs)])
            ref_c = ComposedWOEBin(
                [QuantileInitBin(initial_bins=20),
                 RefChiMergeOptimBin(**kwargs)])
            pd.testing.assert_frame_equal(prod_c(dtm.copy()),
                                          ref_c(dtm.copy()))


def test_chi2_numeric_negative_boundaries():
    """负数区间边界：折叠正则失配场景下 breaks 提取仍逐位一致。"""
    rng = np.random.default_rng(501)
    n = 500
    v = np.round(rng.normal(-50, 3, n), 6)   # 全负值区间
    y = rng.binomial(1, 1 / (1 + np.exp(-(v + 50) / 3)))
    dtm = _dtm(v, y)
    breaks0 = _initial_breaks(dtm, 30)
    for kwargs in (dict(bin_num_limit=5), dict(bin_num_limit=3, p=0.5)):
        _run_chi2_pair(dtm, breaks0, **kwargs)


def test_chi2_zero_marginal_pairs():
    """全好/全坏相邻箱：列边际为 0 → χ²=0.0 短路路径。"""
    n = 400
    v = np.repeat(np.arange(8, dtype=float), 50)
    y = np.concatenate([
        np.zeros(200, dtype=int),   # 前 4 箱全好
        np.ones(200, dtype=int),    # 后 4 箱全坏
    ])
    dtm = _dtm(v, y)
    breaks0 = _initial_breaks(dtm, 8)
    _run_chi2_pair(dtm, breaks0, bin_num_limit=5, count_distr_limit=0.0)
    _run_chi2_pair(dtm, breaks0, bin_num_limit=2, count_distr_limit=0.02,
                   p=0.5)


def test_composed_quantile_tree_chi2_matches_reference():
    """三级组合 [quantile, tree, chi2]：生产链 vs 全参考链输出一致。

    覆盖 tree 输出的 breaks（含 category-dtype 特殊形态）作为 chi2 输入
    的传播路径（initial_binning → set_categories → 行序 → badprob 排序）。
    """
    for seed, kind in ((601, 'num'), (602, 'cat')):
        rng = np.random.default_rng(seed)
        n = 600
        if kind == 'num':
            v = np.round(rng.normal(size=n), 6)
        else:
            v = rng.choice(['a', 'b', 'c', 'd', 'e', 'f'], n)
        y = rng.binomial(1, 0.3, n)
        dtm = _dtm(v, y)

        prod = ComposedWOEBin([
            QuantileInitBin(initial_bins=20),
            TreeOptimBin(bin_num_limit=8, count_distr_limit=0.0),
            ChiMergeOptimBin(bin_num_limit=4),
        ])
        ref = ComposedWOEBin([
            QuantileInitBin(initial_bins=20),
            RefTreeOptimBin(bin_num_limit=8, count_distr_limit=0.0),
            RefChiMergeOptimBin(bin_num_limit=4),
        ])
        pd.testing.assert_frame_equal(prod(dtm.copy()), ref(dtm.copy()))


class RefRuleOptimBin(_LegacyBinningMixin, WOEBin, _LegacyOptimBinMixin):
    """W1 develop ``bins/optimal.py::RuleOptimBin`` 的逐字参考拷贝。

    保留 W1 构造签名（无 ``**kwargs``；B-7 修复前形态），默认行为与
    生产版本一致。
    """

    def __init__(self,
                 lift: float = 3,
                 min_hit_samples=None,
                 pvalue: float = 0.05,
                 direction: str = 'bad',
                 eps: float = 1e-8):
        super().__init__()
        self._min_lift = lift
        self._min_hit_samples = min_hit_samples or 0
        self._p = pvalue
        self._eps = eps
        assert direction in ['good', 'bad'], '挖掘方向为good/bad两者之一'
        self._direction = direction

    def cut_binning(self, binning, idx):
        from scipy.stats import fisher_exact

        flag = np.where(binning.index <= idx, 'left', 'right')
        new_binning = binning.groupby([
            'variable',
            flag,
        ]).agg(
            bin_chr=('bin_chr', lambda x: '%,%'.join(x.tolist())),
            count=('count', 'sum'),
            count_distr=('count_distr', 'sum'),
            good=('good', 'sum'),
            bad=('bad', 'sum')).assign(
                bad_prob=lambda x: x['bad'] / x['count'],
                bad_prob_all=lambda x: (
                    x['bad'].sum() / x['count'].sum())).assign(
                    lift=lambda x: (
                        (x['bad_prob'] + self._eps) / x['bad_prob_all']))
        new_binning['foil'] = (
            new_binning['bad'] * np.log2(new_binning['lift']))

        lift_cond = (
            (self._direction == 'good') &
            (np.any(new_binning['lift'] < self._min_lift)) |
            ((self._direction == 'bad') &
             (np.any(new_binning['lift'] > self._min_lift))))

        # yapf: disable
        if not (lift_cond
            and new_binning['count'].min() > self._min_hit_samples
            and fisher_exact(
                new_binning[['good', 'bad']]).pvalue < self._p):
            new_binning['foil'] = 0
        # yapf: enable

        return new_binning.reset_index(drop=True)

    def woebin(self, dtm, breaks=None):
        assert breaks is not None, \
            f"使用{self.__class__.__name__}类进行分箱，需要传入初始分箱（细分箱）结果"
        binning = self.initial_binning(dtm, breaks)
        if binning.shape[0] < 2:
            return [-np.inf, np.inf]

        # 步骤1：寻找最优切点
        cut_idx_metric = {}
        for idx in range(binning.shape[0] - 1):
            cut_idx_metric[idx] = self.cut_binning(
                binning, idx)['foil'].max()
        sorted_cut_idx_metric = sorted(
            cut_idx_metric.items(), key=lambda x: -x[1])
        best_cut_idx = sorted_cut_idx_metric[0][0]
        best_cut_metric = sorted_cut_idx_metric[0][1]

        # 步骤2：设置监控分箱
        if best_cut_metric == 0:
            # 无法找到最优切点
            return [-np.inf, np.inf]
        else:
            new_binning = self.cut_binning(binning, best_cut_idx)
            binning['cum_count_distr'] = binning['count_distr'].cumsum()
            # yapf: disable
            if new_binning['bad_prob'].is_monotonic_decreasing:
                # 坏率下降，拒绝极小值
                reject_ratio = binning['count_distr'].iloc[best_cut_idx]
                monitor_cut_idx = binning.index[
                    binning['cum_count_distr'] >= min(
                        reject_ratio + 0.05, 1)].min()
                if np.isnan(monitor_cut_idx) or (
                        monitor_cut_idx > binning.shape[0] - 2):
                    monitor_cut_idx = np.inf
            else:
                # 坏率提升，拒绝极大值
                reject_ratio = (
                    1 - binning['cum_count_distr'].iloc[best_cut_idx])
                monitor_cut_idx = binning.index[
                    binning['cum_count_distr'] <= max(
                        1 - reject_ratio - 0.05, 0)].max()
                if np.isnan(monitor_cut_idx):
                    monitor_cut_idx = -np.inf
            # yapf: enable

        cut_idx = np.sort(
            np.unique([-np.inf, best_cut_idx, monitor_cut_idx, np.inf]))
        binning['grp'] = pd.cut(binning.index, cut_idx)
        best_binning = binning.groupby(
            ['variable', 'grp'], observed=False
        ).agg(
            bin_chr=('bin_chr', lambda x: '%,%'.join(x.tolist())))

        if pd.api.types.is_numeric_dtype(dtm['value']):
            best_binning['bin_chr'] = best_binning['bin_chr'].apply(
                lambda x: re.sub(r',[.\d]+\)%,%\[[.\d]+,', ',', x))
            _pattern = re.compile(r"^\[(.*), *(.*)\)")
            breaks = best_binning['bin_chr'].apply(
                lambda x: _pattern.match(x)[2])
            breaks = pd.to_numeric(breaks)
        else:
            breaks = best_binning['bin_chr']

        return breaks


# ============================================================================ #
# Rule：生产向量化实现 vs W1 参考拷贝
# ============================================================================ #

def _run_rule_pair(dtm, breaks, **rule_kwargs):
    prod = RuleOptimBin(**rule_kwargs)
    ref = RefRuleOptimBin(**rule_kwargs)
    b_prod = prod.woebin(dtm, breaks)
    b_ref = ref.woebin(dtm, breaks)
    _assert_breaks_identical(b_prod, b_ref)
    return b_prod


RULE_PARAMS_UNIT = [
    dict(lift=3, min_hit_samples=None, pvalue=0.05, direction='bad',
         initial_bins=20),
    dict(lift=3, min_hit_samples=50, pvalue=0.05, direction='bad',
         initial_bins=20),
    dict(lift=1.5, min_hit_samples=None, pvalue=0.5, direction='good',
         initial_bins=20),
    dict(lift=3, min_hit_samples=None, pvalue=0.05, direction='bad',
         initial_bins=50),
    dict(lift=2, min_hit_samples=20, pvalue=0.2, direction='good',
         initial_bins=50),
]
RULE_PARAMS_SLOW = [
    dict(lift=3, min_hit_samples=None, pvalue=0.05, direction='bad',
         initial_bins=100),
    dict(lift=1.5, min_hit_samples=30, pvalue=0.5, direction='good',
         initial_bins=100),
]


def _rule_id(p):
    return (f"lift{p['lift']}_mhs{p['min_hit_samples']}_p{p['pvalue']}"
            f"_{p['direction']}_ib{p['initial_bins']}")


@pytest.mark.parametrize('case_id', _CASE_IDS)
@pytest.mark.parametrize('params', RULE_PARAMS_UNIT,
                         ids=[_rule_id(p) for p in RULE_PARAMS_UNIT])
def test_rule_kernel_matches_reference(dtm_cases, case_id, params):
    """Rule 向量化实现与 W1 参考拷贝的差分（breaks 逐位相等）。"""
    params = dict(params)
    ib = params.pop('initial_bins')
    dtm = _drop_nan(dtm_cases[case_id])
    breaks = _initial_breaks(dtm, ib)
    if len(breaks) < 2:
        pytest.skip('初始分箱不足 2 个')
    _run_rule_pair(dtm, breaks, **params)


@pytest.mark.slow
@pytest.mark.parametrize('case_id', _CASE_IDS)
@pytest.mark.parametrize('params', RULE_PARAMS_SLOW,
                         ids=[_rule_id(p) for p in RULE_PARAMS_SLOW])
def test_rule_kernel_matches_reference_heavy(dtm_cases, case_id, params):
    """Rule 差分重用例（ib=100）。"""
    params = dict(params)
    ib = params.pop('initial_bins')
    dtm = _drop_nan(dtm_cases[case_id])
    breaks = _initial_breaks(dtm, ib)
    if len(breaks) < 2:
        pytest.skip('初始分箱不足 2 个')
    _run_rule_pair(dtm, breaks, **params)


def test_rule_monitor_bin_paths():
    """监控分箱三分段路径：左尾坏率尖峰（下降分支）与右尾（提升分支）。"""
    n = 2000
    rng = np.random.default_rng(701)
    v = np.round(rng.uniform(0, 10, n), 6)

    # 左尾尖峰：拒绝极小值分支（bad_prob 下降）
    y = rng.binomial(1, np.where(v < 1.0, 0.9, 0.05))
    dtm = _dtm(v, y)
    breaks0 = _initial_breaks(dtm, 50)
    for kwargs in (dict(lift=3, pvalue=0.05),
                   dict(lift=2, min_hit_samples=20)):
        b = _run_rule_pair(dtm, breaks0, **kwargs)
        assert len(list(b)) >= 2

    # 右尾尖峰：拒绝极大值分支（bad_prob 提升）
    y2 = rng.binomial(1, np.where(v > 9.0, 0.9, 0.05))
    dtm2 = _dtm(v, y2)
    breaks0b = _initial_breaks(dtm2, 50)
    b2 = _run_rule_pair(dtm2, breaks0b, lift=3, pvalue=0.05)
    assert len(list(b2)) >= 2

    # 全路径（含监控分箱组装）
    prod = ComposedWOEBin([QuantileInitBin(initial_bins=50),
                           RuleOptimBin(lift=3, pvalue=0.05)])
    ref = ComposedWOEBin([QuantileInitBin(initial_bins=50),
                          RefRuleOptimBin(lift=3, pvalue=0.05)])
    pd.testing.assert_frame_equal(prod(dtm.copy()), ref(dtm.copy()))
    pd.testing.assert_frame_equal(prod(dtm2.copy()), ref(dtm2.copy()))


def test_rule_no_valid_cut_returns_single_bin():
    """无合格切点（lift 门失败）→ 两侧一致返回 [-inf, inf] 单箱。"""
    rng = np.random.default_rng(702)
    n = 500
    v = np.round(rng.normal(size=n), 6)
    y = rng.binomial(1, 0.3, n)   # 纯噪声：bad_prob_all ≈ 0.3，lift ≈ 1 < 3
    dtm = _dtm(v, y)
    breaks0 = _initial_breaks(dtm, 20)
    prod_b = RuleOptimBin().woebin(dtm, breaks0)
    ref_b = RefRuleOptimBin().woebin(dtm, breaks0)
    assert list(prod_b) == [-np.inf, np.inf]
    assert list(ref_b) == [-np.inf, np.inf]


def test_rule_germancredit_categorical_matches_reference():
    """germancredit 类别变量 Rule 差分（含 category-dtype 特殊形态）。"""
    from test.conftest import GERMANCREDIT_FILE, require_data
    require_data(GERMANCREDIT_FILE)
    from syriskmodels.datasets import load_germancredit

    df = load_germancredit()
    cols = [
        'foreign.worker',
        'status.of.existing.checking.account',
        'purpose',
    ]
    param_sets = [
        dict(),
        dict(lift=1.2, direction='good', pvalue=0.5),
        dict(min_hit_samples=30, lift=1.5),
    ]
    for col in cols:
        dtm = pd.DataFrame({
            'variable': col,
            'y': df['creditability'],
            'value': df[col],
        })
        breaks0 = _initial_breaks(dtm, 20)
        for kwargs in param_sets:
            prod = RuleOptimBin(**kwargs)
            ref = RefRuleOptimBin(**kwargs)
            _assert_breaks_identical(prod.woebin(dtm, breaks0),
                                     ref.woebin(dtm, breaks0))
            prod_c = ComposedWOEBin(
                [QuantileInitBin(initial_bins=20), RuleOptimBin(**kwargs)])
            ref_c = ComposedWOEBin(
                [QuantileInitBin(initial_bins=20), RefRuleOptimBin(**kwargs)])
            pd.testing.assert_frame_equal(prod_c(dtm.copy()),
                                          ref_c(dtm.copy()))


# ============================================================================ #
# Phase 6：ComposedWOEBin 计数表缓存（免重复原始扫描 + 逐位等价）
# ============================================================================ #

def _chain_binners(chain, use_ref):
    bins = [QuantileInitBin(initial_bins=20)]
    for name in chain:
        if name == 'tree':
            cls = RefTreeOptimBin if use_ref else TreeOptimBin
            bins.append(cls(bin_num_limit=4))
        elif name == 'chi2':
            cls = RefChiMergeOptimBin if use_ref else ChiMergeOptimBin
            bins.append(cls(bin_num_limit=4))
        elif name == 'rule':
            cls = RefRuleOptimBin if use_ref else RuleOptimBin
            bins.append(cls(lift=1.5, pvalue=0.3))
        else:  # pragma: no cover
            raise AssertionError(name)
    return bins


@pytest.mark.parametrize('case_id', _CASE_IDS)
@pytest.mark.parametrize('chain', [['tree'], ['chi2'], ['rule'],
                                   ['tree', 'chi2'], ['chi2', 'tree']],
                         ids=['q+tree', 'q+chi2', 'q+rule',
                              'q+tree+chi2', 'q+chi2+tree'])
def test_composed_cache_matches_reference(dtm_cases, case_id, chain):
    """组合链（启用缓存的生产实现）vs 全参考链（无缓存）逐位一致。"""
    dtm = dtm_cases[case_id]
    special = ['-999'] if case_id == 'num_missing_sentinel' else None

    prod = ComposedWOEBin(_chain_binners(chain, use_ref=False))
    ref = ComposedWOEBin(_chain_binners(chain, use_ref=True))

    res_prod = prod(dtm.copy(), special_values=special)
    res_ref = ref(dtm.copy(), special_values=special)
    if isinstance(res_ref, str):
        assert res_prod == res_ref
        return
    pd.testing.assert_frame_equal(res_prod, res_ref)


def test_composed_cache_scans_raw_data_once(dtm_cases, monkeypatch):
    """缓存命中证明：[q,tree,chi2] 链路只允许一次原始 binning_breaks 扫描。

    tree 首级扫描一次；chi2 的首级计数表与 __call__ 的最终 binning_breaks
    都必须由段聚合缓存构造（不再调用 WOEBin.binning_breaks）。
    """
    from syriskmodels.scorecard.core import base as base_mod

    dtm = _drop_nan(dtm_cases['num_signal'])
    calls = []
    orig = base_mod.WOEBin.binning_breaks

    def spy(self, d, b):
        calls.append(type(self).__name__)
        return orig(self, d, b)

    monkeypatch.setattr(base_mod.WOEBin, 'binning_breaks', spy)
    composed = ComposedWOEBin([
        QuantileInitBin(initial_bins=20),
        TreeOptimBin(bin_num_limit=5),
        ChiMergeOptimBin(bin_num_limit=4),
    ])
    result = composed(dtm.copy())
    assert isinstance(result, pd.DataFrame)
    assert calls == ['TreeOptimBin'], (
        f'缓存未按预期命中，原始扫描调用: {calls}')


def test_composed_cache_pickle_after_call(dtm_cases):
    """调用后实例仍可 pickle（缓存 weakref 在 __getstate__ 中丢弃）。"""
    import pickle

    dtm = _drop_nan(dtm_cases['num_signal'])
    composed = ComposedWOEBin(
        [QuantileInitBin(initial_bins=20), TreeOptimBin(bin_num_limit=4)])
    first = composed(dtm.copy())
    restored = pickle.loads(pickle.dumps(composed))
    assert repr(restored) == repr(composed)
    pd.testing.assert_frame_equal(restored(dtm.copy()), first)


def test_composed_cache_identity_guard(dtm_cases):
    """缓存同一性守卫：换数据/换 breaks 对象必须走原始扫描且结果正确。"""
    dtm1 = _drop_nan(dtm_cases['num_signal'])
    dtm2 = dtm1.iloc[:500].reset_index(drop=True)

    composed = ComposedWOEBin(
        [QuantileInitBin(initial_bins=20), TreeOptimBin(bin_num_limit=4)])
    composed(dtm1.copy())          # 填充缓存

    # 不同数据：缓存必须失效，结果与全新实例一致
    fresh = ComposedWOEBin(
        [QuantileInitBin(initial_bins=20), TreeOptimBin(bin_num_limit=4)])
    pd.testing.assert_frame_equal(composed(dtm2.copy()), fresh(dtm2.copy()))

    # 外部直接调用 binning_breaks（无 woebin 上下文）行为不变
    brk = fresh.woebin(dtm2.copy())
    direct = composed.binning_breaks(dtm2.copy(), brk)
    baseline = WOEBin.binning_breaks(composed, dtm2.copy(), brk)
    pd.testing.assert_frame_equal(direct, baseline)


# ============================================================================ #
# Phase 7：Numba 后端 vs NumPy 参考实现（逐位一致）+ 后端选择策略
# ============================================================================ #

def _numba_or_skip():
    from syriskmodels.scorecard.core import kernels_numba
    if not kernels_numba.NUMBA_AVAILABLE:
        pytest.skip('numba 不可用，跳过 Numba 后端差分')
    return kernels_numba


TREE_NB_PARAMS = [
    (bnl, mono, cdl, eps)
    for bnl in (1, 3, 5, 6)
    for mono in (False, True)
    for cdl in (0.02,)
    for eps in (0.5,)
] + [(5, False, 0.0, 1e-8), (2, True, 0.05, 0.5)]


@pytest.mark.parametrize('case_id', _CASE_IDS)
def test_numba_tree_kernel_matches_numpy(dtm_cases, case_id):
    """tree 内核：Numba vs NumPy 参考 seg_bounds 逐位相等（参数扫描）。"""
    from syriskmodels.scorecard.core.kernels import tree_cut_search

    kn = _numba_or_skip()
    dtm = _drop_nan(dtm_cases[case_id])
    breaks = _initial_breaks(dtm, 20)
    if len(breaks) < 2:
        pytest.skip('初始分箱不足')
    table = TreeOptimBin().initial_count_table(dtm, breaks)
    ratios = table.count / table.count.sum()

    for bnl, mono, cdl, eps in TREE_NB_PARAMS:
        a = tree_cut_search(table.good, table.bad, ratios, eps, bnl,
                            0.05, cdl, mono)
        b = kn.tree_cut_search_numba(table.good, table.bad, ratios, eps,
                                     bnl, 0.05, cdl, mono)
        np.testing.assert_array_equal(
            a, b, err_msg=f'tree numba mismatch: {case_id} {bnl} {mono} {cdl} {eps}')


@pytest.mark.parametrize('case_id', _CASE_IDS)
def test_numba_chi2_kernel_matches_numpy(dtm_cases, case_id):
    """chi2 内核：Numba vs NumPy 参考 seg_bounds 逐位相等（含 limit>6）。"""
    from syriskmodels.scorecard.core.kernels import chi2_merge_search

    kn = _numba_or_skip()
    dtm = _drop_nan(dtm_cases[case_id])
    breaks = _initial_breaks(dtm, 20)
    if len(breaks) < 2:
        pytest.skip('初始分箱不足')
    table = ChiMergeOptimBin().initial_count_table(dtm, breaks)
    ratios = table.count / table.count.sum()

    from scipy.stats import chi2 as chi2_dist
    for p in (0.05, 0.5):
        for bnl in (1, 3, 5, 8):
            for cdl in (0.0, 0.02):
                lim = chi2_dist.isf(p, df=1)
                a = chi2_merge_search(table.good, table.bad, ratios,
                                      lim, cdl, bnl)
                b = kn.chi2_merge_search_numba(table.good, table.bad, ratios,
                                               lim, cdl, bnl)
                np.testing.assert_array_equal(
                    a, b,
                    err_msg=f'chi2 numba mismatch: {case_id} {p} {bnl} {cdl}')


@pytest.mark.slow
def test_numba_chi2_kernel_large_ib500():
    """chi2 ib=500 大表：Numba 与 NumPy 参考逐位一致（k≈500）。"""
    from syriskmodels.scorecard.core.kernels import chi2_merge_search

    kn = _numba_or_skip()
    rng = np.random.default_rng(901)
    k = 500
    good = rng.integers(0, 500, k).astype('int64')
    bad = rng.integers(0, 40, k).astype('int64')
    count = good + bad
    ratios = count / count.sum()
    from scipy.stats import chi2 as chi2_dist
    lim = chi2_dist.isf(0.05, df=1)
    a = chi2_merge_search(good, bad, ratios, lim, 0.02, 5)
    b = kn.chi2_merge_search_numba(good, bad, ratios, lim, 0.02, 5)
    np.testing.assert_array_equal(a, b)


@pytest.mark.slow
def test_numba_tree_kernel_large_ib500():
    """tree ib=500 大表：Numba 与 NumPy 参考逐位一致。"""
    from syriskmodels.scorecard.core.kernels import tree_cut_search

    kn = _numba_or_skip()
    rng = np.random.default_rng(902)
    k = 500
    good = rng.integers(0, 500, k).astype('int64')
    bad = rng.integers(0, 40, k).astype('int64')
    count = good + bad
    ratios = count / count.sum()
    for bnl in (3, 5, 6):
        for mono in (False, True):
            a = tree_cut_search(good, bad, ratios, 0.5, bnl, 0.05, 0.0, mono)
            b = kn.tree_cut_search_numba(good, bad, ratios, 0.5, bnl, 0.05,
                                         0.0, mono)
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('case_id', ['num_signal', 'num_ties_discrete',
                                     'cat_non_monotonic', 'cat_ordered_risk',
                                     'num_missing_sentinel'])
def test_numba_binner_end_to_end_matches_reference(dtm_cases, case_id):
    """binner 级：engine='numba' 全路径输出与 W1 参考拷贝逐位一致。"""
    _numba_or_skip()
    dtm = dtm_cases[case_id]
    special = ['-999'] if case_id == 'num_missing_sentinel' else None

    for cls_nb, cls_ref, kwargs in [
        (TreeOptimBin, RefTreeOptimBin,
         dict(bin_num_limit=5, count_distr_limit=0.02)),
        (TreeOptimBin, RefTreeOptimBin,
         dict(bin_num_limit=3, ensure_monotonic=True, count_distr_limit=0.0)),
        (ChiMergeOptimBin, RefChiMergeOptimBin,
         dict(bin_num_limit=4, count_distr_limit=0.02)),
        (ChiMergeOptimBin, RefChiMergeOptimBin,
         dict(bin_num_limit=8, p=0.5, count_distr_limit=0.0)),
    ]:
        prod = ComposedWOEBin(
            [QuantileInitBin(initial_bins=20), cls_nb(engine='numba', **kwargs)])
        ref = ComposedWOEBin(
            [QuantileInitBin(initial_bins=20), cls_ref(**kwargs)])
        pd.testing.assert_frame_equal(
            prod(dtm.copy(), special_values=special),
            ref(dtm.copy(), special_values=special))


def test_numba_tree_limit_guard_downgrades_to_numpy(dtm_cases):
    """bin_num_limit > 6 时 engine='numba' 自动降级 numpy，输出与参考一致。"""
    _numba_or_skip()
    from syriskmodels.scorecard.core.kernels import resolve_engine
    assert resolve_engine('numba', 100, 50000, bin_num_limit=8) == 'numpy'

    dtm = _drop_nan(dtm_cases['num_signal'])
    breaks = _initial_breaks(dtm, 100)
    prod = TreeOptimBin(bin_num_limit=8, engine='numba')
    ref = RefTreeOptimBin(bin_num_limit=8)
    _assert_breaks_identical(prod.woebin(dtm, breaks), ref.woebin(dtm, breaks))


def test_resolve_engine_policy(monkeypatch):
    """后端选择策略：auto 阈值、显式指定、不可用回退、非法值。"""
    from syriskmodels.scorecard.core import kernels_numba
    from syriskmodels.scorecard.core.kernels import (
        NUMBA_MAX_TREE_LIMIT,
        NUMBA_MIN_BINS,
        NUMBA_MIN_ROWS,
        resolve_engine,
    )

    assert resolve_engine('numpy', 10**6, 10**9) == 'numpy'
    with pytest.raises(ValueError):
        resolve_engine('cuda', 10, 10)

    # auto：小数据不启用（且不 import numba —— 见独立子进程用例）
    assert resolve_engine('auto', NUMBA_MIN_BINS - 1, 10**6) == 'numpy'
    assert resolve_engine('auto', 10**6, NUMBA_MIN_ROWS - 1) == 'numpy'
    # auto：tree 求和长度护栏
    assert resolve_engine('auto', 100, 50000,
                          bin_num_limit=NUMBA_MAX_TREE_LIMIT + 1) == 'numpy'

    if kernels_numba.NUMBA_AVAILABLE:
        assert resolve_engine('auto', 100, 50000) == 'numba'
        assert resolve_engine('auto', 100, 50000,
                              bin_num_limit=NUMBA_MAX_TREE_LIMIT) == 'numba'
        assert resolve_engine('numba', 5, 100) == 'numba'

    # numba 不可用 → 回退 numpy（显式请求也一样，不抛错）
    monkeypatch.setattr(kernels_numba, 'NUMBA_AVAILABLE', False)
    assert resolve_engine('numba', 100, 50000) == 'numpy'
    assert resolve_engine('auto', 100, 50000) == 'numpy'


def test_package_import_does_not_load_numba():
    """§7.6：import syriskmodels.scorecard 不得连带 import numba。"""
    import subprocess
    import sys

    code = ('import sys; import syriskmodels.scorecard; '
            'print("numba" in sys.modules)')
    out = subprocess.run([sys.executable, '-c', code],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == 'False', (
        f'包导入即加载了 numba：{out.stdout.strip()}')


def test_auto_engine_selects_numba_at_scale():
    """auto 策略端到端：n≥5000 且 k≥64 时走 Numba，输出与参考一致。"""
    kn = _numba_or_skip()
    rng = np.random.default_rng(903)
    n = 6000
    v = np.round(rng.normal(size=n), 6)
    y = rng.binomial(1, 1 / (1 + np.exp(-1.2 * v)))
    dtm = _dtm(v, y)
    breaks0 = _initial_breaks(dtm, 100)   # k≈100 ≥ NUMBA_MIN_BINS

    prod = TreeOptimBin(bin_num_limit=5)          # engine='auto'
    ref = RefTreeOptimBin(bin_num_limit=5)
    _assert_breaks_identical(prod.woebin(dtm, breaks0),
                             ref.woebin(dtm, breaks0))

    prod_chi2 = ChiMergeOptimBin(bin_num_limit=5)
    ref_chi2 = RefChiMergeOptimBin(bin_num_limit=5)
    _assert_breaks_identical(prod_chi2.woebin(dtm, breaks0),
                             ref_chi2.woebin(dtm, breaks0))


def test_chi2_rule_user_breaks_with_empty_bins():
    """chi2/rule：用户指定 breaks 制造空分箱（count=0 → χ²=0 短路 /
    bad_prob NaN / lift 除零路径）差分一致。"""
    rng = np.random.default_rng(801)
    n = 400
    v = np.round(rng.uniform(0, 1, n), 6)
    y = rng.binomial(1, 0.3, n)
    dtm = _dtm(v, y)
    # 数据只在 [0,1]：(-10,-5] 与 [5,10) 区间为空分箱
    breaks = [-np.inf, -5.0, 0.25, 0.5, 0.75, 5.0, np.inf]

    for kwargs in (dict(bin_num_limit=5, count_distr_limit=0.0),
                   dict(bin_num_limit=3, count_distr_limit=0.01, p=0.5)):
        _run_chi2_pair(dtm, breaks, **kwargs)

    for kwargs in (dict(lift=3, pvalue=0.05),
                   dict(lift=1.2, pvalue=0.5, direction='good')):
        _run_rule_pair(dtm, breaks, **kwargs)


def test_single_row_and_two_bin_degenerate_tables():
    """退化表边界：单分箱 / 两分箱输入下 tree/chi2/rule 差分一致。"""
    # 单值变量（quantile 只产出 1 个箱 → 内核退化路径）
    dtm1 = _dtm(np.ones(50), np.tile([0, 1], 25))
    breaks1 = _initial_breaks(dtm1, 20)
    if len(breaks1) >= 2:
        for kwargs in (dict(bin_num_limit=5),):
            prod = TreeOptimBin(**kwargs)
            ref = RefTreeOptimBin(**kwargs)
            _assert_breaks_identical(prod.woebin(dtm1, breaks1),
                                     ref.woebin(dtm1, breaks1))

    # 两个唯一值
    dtm2 = _dtm(np.array([0.0, 1.0] * 40), np.tile([0, 0, 1, 1], 20))
    breaks2 = _initial_breaks(dtm2, 20)
    for Prod, Ref, kwargs in [
        (TreeOptimBin, RefTreeOptimBin, dict(bin_num_limit=5)),
        (TreeOptimBin, RefTreeOptimBin,
         dict(bin_num_limit=1, ensure_monotonic=True)),
        (ChiMergeOptimBin, RefChiMergeOptimBin, dict(bin_num_limit=1)),
        (RuleOptimBin, RefRuleOptimBin, dict(lift=1.1, pvalue=0.9)),
    ]:
        _assert_breaks_identical(
            Prod(**kwargs).woebin(dtm2, breaks2),
            Ref(**kwargs).woebin(dtm2, breaks2))

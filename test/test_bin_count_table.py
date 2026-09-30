# -*- encoding: utf-8 -*-
"""W2 Phase 2 验收：BinCountTable 与聚合向量化。

验收口径（W2 任务书 §2.5）：count、good、bad、WOE、IV、breaks 与旧实现
一致。端到端输出由 golden 快照钉住（本阶段 golden 必须保持逐位不变）；
本文件做**结构级**差分：

1. ``WOEBin.binning`` 向量化聚合 vs 旧 ``_n0/_n1`` lambda 聚合（参考拷贝）；
2. ``BinCountTable`` 派生属性 vs 生产 ``binning_format`` 的
   count/count_distr/woe/bin_iv/total_iv；
3. ``OptimBinMixin.initial_count_table`` vs ``initial_binning``：行序与数值
   完全一致（数值型按区间升序；类别型按 badprob 降序，任务书 §2.4）。
"""
import numpy as np
import pandas as pd
import pytest

from syriskmodels.scorecard import (
    BinCountTable,
    QuantileInitBin,
    TreeOptimBin,
    WOEBin,
    woebin,
)


# --------------------------------------------------------------------------- #
# 旧实现参考拷贝（仅测试用；来源：W1 develop 的 core/base.py::WOEBin.binning）
# --------------------------------------------------------------------------- #

def _legacy_binning(dtm: pd.DataFrame, bin_chr: pd.Series) -> pd.DataFrame:
    def _n0(x):
        return np.sum(x == 0)

    def _n1(x):
        return np.sum(x == 1)

    bin_chr = bin_chr.rename(index='bin_chr')
    binning = dtm.groupby(['variable', bin_chr], observed=False)['y'].agg(
        good=_n0, bad=_n1)
    binning = binning.reset_index()
    return binning


# --------------------------------------------------------------------------- #
# 测试数据
# --------------------------------------------------------------------------- #

def _frame(n=500, seed=9):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'num': np.round(rng.normal(size=n), 6),
        'cat': rng.choice(['a', 'b', 'c', 'd', 'e'], n),
        'y': rng.binomial(1, 0.3, n),
    })


def _dtm(frame, col):
    return pd.DataFrame({
        'variable': col,
        'y': frame['y'],
        'value': frame[col],
    })


# --------------------------------------------------------------------------- #
# 1. binning() 向量化 vs legacy lambda 聚合
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('col', ['num', 'cat'])
def test_binning_vectorized_matches_legacy_lambda_agg(col):
    frame = _frame()
    dtm = _dtm(frame, col)

    if col == 'num':
        breaks = [-np.inf, -1.0, 0.0, 1.0, np.inf]
        labels = [f'[{breaks[i]},{breaks[i + 1]})'
                  for i in range(len(breaks) - 1)]
        bin_chr = pd.cut(dtm['value'], breaks, right=False, labels=labels)
    else:
        bin_chr = dtm['value']

    new = WOEBin.binning(dtm, bin_chr)
    old = _legacy_binning(dtm, bin_chr)
    pd.testing.assert_frame_equal(new, old)


def test_binning_matches_legacy_with_empty_bins_and_nan_keys():
    """边界：空分箱（observed=False 保留）与 NaN 组键（groupby 丢弃）。"""
    frame = _frame(n=200, seed=11)
    dtm = _dtm(frame, 'num')

    # 极宽区间制造空分箱
    breaks = [-np.inf, -100.0, -50.0, 0.0, 50.0, 100.0, np.inf]
    labels = [f'[{breaks[i]},{breaks[i + 1]})'
              for i in range(len(breaks) - 1)]
    bin_chr = pd.cut(dtm['value'], breaks, right=False, labels=labels)
    pd.testing.assert_frame_equal(WOEBin.binning(dtm, bin_chr),
                                  _legacy_binning(dtm, bin_chr))

    # NaN 组键（模拟未匹配到 break 的类别值）
    cat_chr = pd.Series(
        ['pos' if v > 0 else np.nan for v in dtm['value'].tolist()],
        index=dtm.index, dtype=object)
    pd.testing.assert_frame_equal(WOEBin.binning(dtm, cat_chr),
                                  _legacy_binning(dtm, cat_chr))


# --------------------------------------------------------------------------- #
# 2. BinCountTable 派生属性 vs 生产 binning_format
# --------------------------------------------------------------------------- #

def test_table_properties_match_binning_format():
    frame = _frame(n=800, seed=13)
    bins = woebin(frame, y='y', x=['num', 'cat'],
                  methods=['quantile', 'tree'],
                  initial_bins=20, bin_num_limit=5, no_cores=1)

    for var in ('num', 'cat'):
        res = bins[var]
        table = BinCountTable(
            variable=var,
            bin_chr=res['bin'].to_numpy(dtype=object),
            good=res['good'].to_numpy(),
            bad=res['bad'].to_numpy(),
            is_numeric=(var == 'num'),
            epsilon=0.5,
        )
        np.testing.assert_array_equal(table.count, res['count'].to_numpy())
        assert table.total == int(res['count'].sum())
        np.testing.assert_allclose(table.count_distr,
                                   res['count_distr'].to_numpy(),
                                   rtol=0, atol=1e-15)
        np.testing.assert_allclose(table.woe, res['woe'].to_numpy(),
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(table.bin_iv, res['bin_iv'].to_numpy(),
                                   rtol=0, atol=1e-12)
        assert table.total_iv == pytest.approx(
            float(res['total_iv'].iloc[0]), rel=0, abs=1e-12)

        # 与 TreeOptimBin.iv 的一致性（内核 IV 公式的另一个消费方）
        tree = TreeOptimBin()
        assert table.total_iv == pytest.approx(
            tree.iv(res['good'].to_numpy(), res['bad'].to_numpy()),
            rel=0, abs=1e-12)


def test_table_is_frozen_and_validated():
    good = np.array([10, 20], dtype='int64')
    bad = np.array([1, 2], dtype='int64')
    table = BinCountTable('v', np.array(['a', 'b'], dtype=object), good, bad,
                          is_numeric=False)
    with pytest.raises(Exception):
        table.good = good  # frozen dataclass
    with pytest.raises(ValueError):
        BinCountTable('v', np.array(['a'], dtype=object), good, bad,
                      is_numeric=False)
    assert table.n_bins == 2 and table.total == 33
    assert repr(table).startswith('BinCountTable(')


# --------------------------------------------------------------------------- #
# 3. initial_count_table vs initial_binning（行序 + 数值）
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('col,is_numeric', [('num', True), ('cat', False)])
def test_initial_count_table_matches_initial_binning(col, is_numeric):
    frame = _frame(n=600, seed=17)
    dtm = _dtm(frame, col)

    q = QuantileInitBin(initial_bins=20)
    breaks = q.woebin(dtm)

    tree = TreeOptimBin(bin_num_limit=5)
    df = tree.initial_binning(dtm, breaks)
    table = tree.initial_count_table(dtm, breaks)

    assert table.variable == col
    assert table.is_numeric == is_numeric
    assert table.n_bins == len(df)
    np.testing.assert_array_equal(table.bin_chr.astype(str),
                                  df['bin_chr'].astype(str).to_numpy())
    np.testing.assert_array_equal(table.good, df['good'].to_numpy())
    np.testing.assert_array_equal(table.bad, df['bad'].to_numpy())
    np.testing.assert_array_equal(table.count, df['count'].to_numpy())
    np.testing.assert_allclose(table.count_distr,
                               df['count_distr'].to_numpy(),
                               rtol=0, atol=0)


def test_initial_count_table_row_order_semantics():
    """任务书 §2.4：数值型按区间顺序；类别型按 badprob 降序。"""
    frame = _frame(n=600, seed=19)

    # 数值型：区间字符串按左端点升序
    dtm_num = _dtm(frame, 'num')
    q = QuantileInitBin(initial_bins=20)
    tree = TreeOptimBin(bin_num_limit=5)
    table_num = tree.initial_count_table(dtm_num, q.woebin(dtm_num))
    lefts = [float(s.split(',')[0].lstrip('[')) for s in table_num.bin_chr]
    assert lefts == sorted(lefts)
    assert lefts[0] == -np.inf

    # 类别型：badprob 降序
    dtm_cat = _dtm(frame, 'cat')
    table_cat = tree.initial_count_table(dtm_cat, q.woebin(dtm_cat))
    badprob = table_cat.badprob
    assert np.all(np.diff(badprob) <= 0), '类别型初始计数表必须按 badprob 降序'


def test_table_to_binning_df_roundtrip():
    frame = _frame(n=300, seed=23)
    dtm = _dtm(frame, 'num')
    q = QuantileInitBin(initial_bins=10)
    tree = TreeOptimBin(bin_num_limit=4)
    table = tree.initial_count_table(dtm, q.woebin(dtm))

    df = table.to_binning_df()
    assert list(df.columns) == ['variable', 'bin_chr', 'good', 'bad',
                                'count', 'count_distr']
    back = BinCountTable.from_binning_df(df, is_numeric=True)
    np.testing.assert_array_equal(back.good, table.good)
    np.testing.assert_array_equal(back.bad, table.bad)
    np.testing.assert_array_equal(back.bin_chr.astype(str),
                                  table.bin_chr.astype(str))
    assert back.total_iv == pytest.approx(table.total_iv, rel=0, abs=0)

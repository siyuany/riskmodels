# -*- encoding: utf-8 -*-
"""
分箱计数表模块（W2 Phase 2）

``BinCountTable`` 是粗分箱内核的**唯一数据结构**：细分箱（quantile/hist）
产出的初始分箱被压缩为 ``(bin_chr, good, bad)`` 三个等长数组，Tree /
ChiMerge / Rule 的候选搜索只依赖该结构做 NumPy 向量化运算，不再在候选
循环里操作原始 DataFrame（W1 报告 §6.2 指出的 O(n_bins²) pandas 开销
热点）。

语义约定（与 legacy ``OptimBinMixin.initial_binning`` 输出逐行一致）：

* 行序：数值型按区间升序；类别型按 badprob 降序（与
  ``DataFrame.sort_values(by='badprob', ascending=False)`` 相同的排序结果，
  由构造方负责排好序后传入）；
* ``good`` / ``bad`` 为 int64 计数；
* ``woe`` / ``total_iv`` 与 ``WOEBin.binning_format`` 及
  ``TreeOptimBin.iv`` 的 epsilon 替换公式一致（0 计数以 ``epsilon``
  替换后再归一化）。

本结构为**只读约定**：字段是普通 ndarray，内核不应原地修改；需要变更时
用 :meth:`replace` 生成新表。
"""
from dataclasses import dataclass, replace as _dc_replace
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True, eq=False)
class BinCountTable:
    """粗分箱计数表。

    参数:
        variable: 变量名
        bin_chr: 各初始分箱的分箱名（object ndarray[str]；数值型为
            ``'[a,b)'`` 区间字符串，类别型为类别名或 ``'%,%'`` 组合名）
        good: 各分箱好样本数（int64）
        bad: 各分箱坏样本数（int64）
        is_numeric: 变量是否为数值型（决定 breaks 提取与折叠语义）
        epsilon: WOE/IV 计算中替换 0 计数的值（与 ``WOEBin.epsilon`` 一致，
            默认 0.5）
    """

    variable: str
    bin_chr: np.ndarray
    good: np.ndarray
    bad: np.ndarray
    is_numeric: bool
    epsilon: float = 0.5

    def __post_init__(self):
        good = np.asarray(self.good)
        bad = np.asarray(self.bad)
        bin_chr = np.asarray(self.bin_chr, dtype=object)
        if not (good.shape == bad.shape == bin_chr.shape):
            raise ValueError(
                'BinCountTable 的 bin_chr/good/bad 长度必须一致：'
                f'{bin_chr.shape} / {good.shape} / {bad.shape}')
        # frozen dataclass 内规范化 dtype（object.__setattr__）
        object.__setattr__(self, 'good', good.astype('int64', copy=False))
        object.__setattr__(self, 'bad', bad.astype('int64', copy=False))
        object.__setattr__(self, 'bin_chr', bin_chr)

    # ------------------------------------------------------------------ #
    # 只读派生属性
    # ------------------------------------------------------------------ #

    @property
    def n_bins(self) -> int:
        """初始分箱数量。"""
        return int(self.good.shape[0])

    @property
    def count(self) -> np.ndarray:
        """各分箱样本数（good + bad），int64。"""
        return self.good + self.bad

    @property
    def total(self) -> int:
        """总样本数。"""
        return int(self.count.sum())

    @property
    def count_distr(self) -> np.ndarray:
        """各分箱样本占比（float64）。"""
        count = self.count
        return count / count.sum()

    @property
    def badprob(self) -> np.ndarray:
        """各分箱坏样本率（count=0 时为 NaN，与 legacy 一致）。"""
        return self.bad / self.count

    @property
    def woe(self) -> np.ndarray:
        """各分箱 WOE（epsilon 替换 0 计数后 log(good_dist/bad_dist)）。

        与 ``WOEBin.binning_format`` 的 woe 列公式一致。
        """
        good, bad = self._sub0()
        good_distr = good / good.sum()
        bad_distr = bad / bad.sum()
        return np.log(good_distr / bad_distr)

    @property
    def bin_iv(self) -> np.ndarray:
        """各分箱 IV 分量（(good_dist - bad_dist) * woe）。"""
        good, bad = self._sub0()
        good_distr = good / good.sum()
        bad_distr = bad / bad.sum()
        return (good_distr - bad_distr) * np.log(good_distr / bad_distr)

    @property
    def total_iv(self) -> float:
        """总 IV（bin_iv 之和）。

        与 ``TreeOptimBin.iv(good, bad)`` / ``binning_format`` 的
        ``total_iv`` 数值一致（相同行序、相同求和顺序）。
        """
        return float(self.bin_iv.sum())

    def _sub0(self):
        """0 计数以 epsilon 替换（legacy ``binning_format.sub0`` 语义）。"""
        eps = self.epsilon
        good = np.where(self.good == 0, eps, self.good).astype('float64')
        bad = np.where(self.bad == 0, eps, self.bad).astype('float64')
        return good, bad

    # ------------------------------------------------------------------ #
    # 构造 / 转换
    # ------------------------------------------------------------------ #

    @classmethod
    def from_binning_df(
        cls,
        binning: pd.DataFrame,
        is_numeric: bool,
        epsilon: float = 0.5,
    ) -> 'BinCountTable':
        """从 legacy ``initial_binning`` / ``binning`` 输出 DataFrame 构造。

        参数:
            binning: 含 ``variable`` / ``bin_chr`` / ``good`` / ``bad`` 列的
                DataFrame（行序即最终行序，本方法不排序）
            is_numeric: 变量是否数值型
            epsilon: WOE/IV 的 0 计数替换值
        """
        variable = str(binning['variable'].iloc[0]) if len(binning) else ''
        bin_chr = binning['bin_chr'].to_numpy(dtype=object).astype(str)
        return cls(
            variable=variable,
            bin_chr=np.asarray(bin_chr, dtype=object),
            good=binning['good'].to_numpy(),
            bad=binning['bad'].to_numpy(),
            is_numeric=bool(is_numeric),
            epsilon=float(epsilon),
        )

    def to_binning_df(self) -> pd.DataFrame:
        """转回 legacy ``initial_binning`` 形态的 DataFrame。

        列：``variable`` / ``bin_chr``(object) / ``good`` / ``bad`` /
        ``count`` / ``count_distr``，行序与表一致。仅供桥接与调试，
        内核热路径不应调用。
        """
        count = self.count
        return pd.DataFrame({
            'variable': self.variable,
            'bin_chr': self.bin_chr.astype(str),
            'good': self.good,
            'bad': self.bad,
            'count': count,
            'count_distr': count / count.sum(),
        })

    def replace(self, **changes) -> 'BinCountTable':
        """生成替换了部分字段的新表（frozen dataclass 的受控变更）。"""
        return _dc_replace(self, **changes)

    def __repr__(self) -> str:
        return (f'BinCountTable(variable={self.variable!r}, n_bins={self.n_bins}, '
                f'total={self.total}, is_numeric={self.is_numeric})')

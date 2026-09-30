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
from typing import Any, Optional

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
    #: 初始分箱 bin_chr 列若为 categorical dtype，这里保存其 categories
    #: **Index 对象本身**（顺序 = 进入 ``initial_binning`` 的 breaks 顺序；
    #: Index 的 ``name`` 也是可观测行为 —— 它经 legacy groupby-agg 的
    #: category dtype 保留、再经 ``set_categories`` 传播到最终 breaks 列的
    #: categories.names，多级组合链上必须原样携带）。legacy 语义中它决定
    #: "无合并段"时类别型 breaks Series 的 dtype（category）与下游
    #: ``set_categories`` 采用的顺序（W2 Phase 3/6 差分发现）。
    categories: Optional[Any] = None

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
        bin_chr_col = binning['bin_chr']
        categories = None
        if isinstance(bin_chr_col.dtype, pd.CategoricalDtype):
            # 保存 Index 对象本身（含 name —— 见字段注释）
            categories = bin_chr_col.cat.categories
        bin_chr = bin_chr_col.to_numpy(dtype=object).astype(str)
        return cls(
            variable=variable,
            bin_chr=np.asarray(bin_chr, dtype=object),
            good=binning['good'].to_numpy(),
            bad=binning['bad'].to_numpy(),
            is_numeric=bool(is_numeric),
            epsilon=float(epsilon),
            categories=categories,
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

    def segment_sums(self, seg_bounds) -> tuple:
        """按段边界聚合 good/bad（int64 前缀和差分，精确）。

        返回 ``(seg_good, seg_bad)``，段 i = 行 ``[b_i, b_{i+1})``。
        """
        bounds = np.asarray(seg_bounds, dtype='int64')
        n = int(self.good.shape[0])
        pre_g = np.zeros(n + 1, dtype='int64')
        pre_b = np.zeros(n + 1, dtype='int64')
        if n > 0:
            np.cumsum(self.good, out=pre_g[1:])
            np.cumsum(self.bad, out=pre_b[1:])
        seg_g = pre_g[bounds[1:]] - pre_g[bounds[:-1]]
        seg_b = pre_b[bounds[1:]] - pre_b[bounds[:-1]]
        return seg_g, seg_b

    def __repr__(self) -> str:
        return (f'BinCountTable(variable={self.variable!r}, n_bins={self.n_bins}, '
                f'total={self.total}, is_numeric={self.is_numeric})')


def binning_from_segments(
    table: BinCountTable,
    seg_bounds,
    breaks,
    is_numeric: bool,
) -> Optional[pd.DataFrame]:
    """由父级计数表 + 段边界聚合出与全新扫描**逐位等价**的 binning 表。

    W2 Phase 6（ComposedWOEBin 缓存）核心：上一级粗分箱的输出 ``breaks``
    与段边界 ``seg_bounds`` 描述的是父级计数表行的一个**连续区间划分**，
    对 ``dtm`` 重新执行 ``binning_breaks``（pd.cut/merge + groupby 全量
    扫描）得到的分组计数与直接对父表做段聚合在数学上恒等；本函数进一步
    保证输出 DataFrame 的**行序 / dtype / categories** 与全新扫描一致：

    * 数值型：labels 用与 ``WOEBin.binning_breaks`` 完全相同的构造式
      （``'[{},{})'.format``，break_list = ``-inf`` + 升序去重有限边界 +
      ``inf``）生成，bin_chr 为 ``Categorical(labels, ordered=True)``
      （= ``pd.cut(labels=...)`` 的 dtype），行序 = 区间升序；
    * 类别型：段内 bin_chr 以 ``'%,%'`` 拼接；行序 = ``set_categories``
      后的 categories 顺序（breaks 为 categorical Series 时取其
      ``cat.categories`` —— legacy 的 pandas 语义；否则取 breaks 值序）；
      bin_chr 为 ``Categorical(ordered=True)``，与
      ``astype('category').cat.set_categories(breaks, ordered=True)`` +
      ``sort_values`` 的输出一致；
    * good/bad：int64 前缀和差分（精确，等于 groupby.sum 的整数结果）。

    校验失败（breaks 与段不一一对应、解析异常等）时返回 ``None``，
    调用方**必须回退**到原始扫描路径 —— 缓存只是性能提示，绝不改变语义。

    参数:
        table: 父级计数表
        seg_bounds: 段边界（相对父表行号）
        breaks: 与 seg_bounds 对应的 breaks 输出（上一级内核产物）
        is_numeric: 变量是否数值型

    返回:
        与 ``WOEBin.binning_breaks(dtm, breaks)`` 输出一致的 DataFrame
        （列：variable / bin_chr / good / bad），或 None（回退信号）
    """
    try:
        seg_bounds = np.asarray(seg_bounds, dtype='int64')
        n_seg = int(seg_bounds.shape[0] - 1)
        if n_seg < 1:
            return None
        seg_g, seg_b = table.segment_sums(seg_bounds)
        bin_chr_arr = table.bin_chr

        if is_numeric:
            # 复刻 binning_breaks 的 break_list / labels 构造
            values = [float(v) for v in list(breaks)]
            finite = sorted({v for v in values if np.isfinite(v)})
            break_list = [-np.inf] + finite + [np.inf]
            labels = ['[{},{})'.format(break_list[i], break_list[i + 1])
                      for i in range(len(break_list) - 1)]
            if len(labels) != n_seg:
                return None
            # 校验：labels 与段一一对应（右边界逐一相等）
            for i in range(n_seg):
                last = str(bin_chr_arr[int(seg_bounds[i + 1]) - 1])
                right = last[last.rindex(',') + 1:-1]
                lab_right = labels[i][labels[i].rindex(',') + 1:-1]
                if float(right) != float(lab_right):
                    return None
            bin_chr = pd.Categorical(labels, categories=labels, ordered=True)
            return pd.DataFrame({
                'variable': [table.variable] * n_seg,
                'bin_chr': bin_chr,
                'good': seg_g,
                'bad': seg_b,
            })

        # 类别型：段内拼接
        joined = ['%,%'.join([str(x) for x in bin_chr_arr[a:b]])
                  for a, b in zip(seg_bounds[:-1], seg_bounds[1:])]

        # categories 顺序 = 全新扫描下 set_categories(breaks) 的结果。
        # 保真细节：全新路径的 categories 索引会携带 breaks Series 的
        # name（'bin_chr'），并传播到最终输出 breaks 列的 categories.names
        # —— 属可观测行为（assert_frame_equal 会比对）。这里以相同方式
        # 构造：categorical breaks 直接取其 cat.categories 对象；
        # Series 用 Index(breaks)（保留 name）；其余用值列表。
        if isinstance(breaks, pd.Series) and isinstance(
                breaks.dtype, pd.CategoricalDtype):
            cats_index = breaks.cat.categories
        elif isinstance(breaks, pd.Series):
            cats_index = pd.Index(breaks)
        else:
            cats_index = pd.Index([str(v) for v in list(breaks)])
        cats = [str(c) for c in cats_index]
        if len(cats) != n_seg or set(cats) != set(joined):
            return None
        # 行序 = categories 顺序（sort_values 于唯一类别位序上是全序，
        # 与全新扫描的排序结果一致且稳定）
        pos = {name: i for i, name in enumerate(joined)}
        order = [pos[c] for c in cats]
        bin_chr = pd.Categorical(cats, categories=cats_index, ordered=True)
        return pd.DataFrame({
            'variable': [table.variable] * n_seg,
            'bin_chr': bin_chr,
            'good': seg_g[order],
            'bad': seg_b[order],
        })
    except (TypeError, ValueError, IndexError, KeyError):
        return None

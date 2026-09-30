# -*- encoding: utf-8 -*-
"""NumPy 精确等价分箱内核（W2 Phase 3–5）。

本模块承载 Tree / ChiMerge / Rule 粗分箱的搜索内核：**只接受定长数组与
标量参数，输出段边界索引**，不接触 pandas/object/变长字符串（字符串仅在
边界组装层处理）。设计目标（W2 任务书硬性约束）：

1. **精确等价**：与 W1 develop 的 pandas 实现逐位一致 —— 差分测试见
   ``test/test_binning_equivalence.py``（旧实现参考拷贝为 oracle）。
   为保证逐位一致，内核显式复刻了 legacy 的浮点行为：

   * int 计数段和：前缀和差分（int64 精确，与 groupby.sum 一致，已验证）；
   * float ``count_distr`` 段和：pandas ``groupby.sum(float64)`` 使用
     **Kahan 补偿求和**（经 20 组对抗性随机试验逐位验证），内核以相同
     顺序、相同补偿算术复刻（:func:`kahan_prefix_sums` /
     :func:`kahan_suffix_sums_batched`）；
   * IV：复刻 ``TreeOptimBin.iv`` 的算式与运算次序 ——
     ``np.where(x==0, eps, x)`` → ``np.sum``（numpy pairwise，
     ``axis=1`` 批量与逐行标量逐位一致，已验证）→ 除法 → ``np.log``；
   * bad_prob 单调约束：复刻 ``utils.monotonic`` 对 NaN（任何 NaN →
     非单调）与等值（视为单调）的语义；
   * tie-breaking：IV 相同取**最小候选索引**（legacy 为稳定排序 +
     升序插入，等价于 ``np.argmax`` 的首个最大值）。

2. **性能**：候选评估全部批量向量化；每轮复杂度 O(k·C) 的 numpy 元素
   运算 + O(k) 的 Kahan 扫描步，替代 legacy 每候选一次 groupby 全量重建
   （O(k²) 量级的 pandas 开销）。

3. **Numba 可移植**（Phase 7）：内核不做 pandas/object 操作，Numba 版本
   必须以相同算式与求和顺序实现，输出与本模块逐位一致；``fastmath``
   必须为 False。

段边界约定：``seg_bounds = [b0=0, b1, ..., bm=k]``，段 i 覆盖初始计数表
行 ``[b_i, b_{i+1})``（行序 = legacy ``initial_binning`` 输出行序：数值型
区间升序、类别型 badprob 降序）。
"""
from typing import List

import numpy as np
import pandas as pd

__all__ = [
    'kahan_prefix_sums',
    'kahan_suffix_sums_batched',
    'tree_cut_search',
    'chi2_pair_stats',
    'chi2_merge_search',
    'segments_to_breaks',
]


# --------------------------------------------------------------------------- #
# Kahan 补偿求和（与 pandas groupby.sum(float64) 逐位一致）
# --------------------------------------------------------------------------- #

def kahan_prefix_sums(values: np.ndarray, start: int, end: int) -> np.ndarray:
    """``values[start..j]``（前向、含 j）的 Kahan 前缀和，j = start..end-1。

    返回数组 ``out``，``out[j - start]`` = Kahan(values[start..j])。
    每个前缀的累加路径与 pandas ``groupby.sum`` 对该区间的求和**逐位一致**
    （相同的补偿算术与顺序；返回 sum 不做末尾补偿修正，与 pandas 相同）。
    """
    n = end - start
    out = np.empty(max(n, 0), dtype='float64')
    s = 0.0
    c = 0.0
    for i, v in enumerate(values[start:end].tolist()):
        y = v - c
        t = s + y
        c = (t - s) - y
        s = t
        out[i] = s
    return out


def kahan_suffix_sums_batched(
    values: np.ndarray,
    start: int,
    end: int,
    cand: np.ndarray,
) -> np.ndarray:
    """对每个候选 c（start ≤ c ≤ end-2）计算 ``values[c+1 .. end-1]`` 的
    **前向** Kahan 和。

    候选批向量化：位置 j 只有已进入自身区间（c < j）的候选参与累加；
    ``cand`` 必须升序（活跃集是其前缀切片）。每个候选的累加路径与
    pandas ``groupby.sum`` 一致（前向、逐元素补偿）。
    """
    cand = np.asarray(cand)
    n_c = cand.shape[0]
    s = np.zeros(n_c, dtype='float64')
    comp = np.zeros(n_c, dtype='float64')
    if n_c == 0:
        return s
    vals = values[start:end].tolist()
    for j in range(start + 1, end):
        # 活跃候选数：#{c < j}（cand 升序 → 前缀切片）
        a = int(np.searchsorted(cand, j, side='left'))
        if a == 0:
            continue
        v = vals[j - start]
        sv = s[:a]
        cv = comp[:a]
        y = v - cv
        t = sv + y
        comp[:a] = (t - sv) - y
        s[:a] = t
    return s


# --------------------------------------------------------------------------- #
# Tree 内核
# --------------------------------------------------------------------------- #

def tree_cut_search(
    good: np.ndarray,
    bad: np.ndarray,
    ratios: np.ndarray,
    epsilon: float,
    bin_num_limit: int,
    min_iv_inc: float,
    count_distr_limit: float,
    ensure_monotonic: bool,
) -> np.ndarray:
    """TreeOptimBin 贪心切分搜索的 NumPy 精确等价实现。

    复刻 legacy 语义（逐条对应 W1 develop ``TreeOptimBin.woebin``）：

    * cp 标记：初始仅最后一行 cp=True；每轮候选 = ~cp 的行（升序）；
      被接受的切点置 cp=True（段边界行天然不可再切）；
    * 每轮对**所有**候选计算切分后分区的 total_iv；
    * 约束：分区所有段 count_distr > count_distr_limit（严格大于；
      Kahan 段和逐位复刻）；ensure_monotonic=True 时 bad_prob 序列
      单调（含等值；任何 NaN → 非单调 → 拒绝）；
    * 接受条件：``(curr_iv - last_iv + 1e-8) / (last_iv + 1e-8) >
      min_iv_inc``（运算次序一致；last_iv 初始为 0）；
    * 择优：IV 最大者；并列取最小索引（legacy 稳定排序等价）；
    * 终止：本轮无合格候选，或段数 > bin_num_limit（循环条件在候选
      生成**前**检查 → 最终段数上限为 bin_num_limit + 1，保持 legacy
      的 off-by-one 行为）。

    参数:
        good/bad: 初始计数表的好/坏样本数（int64，行序即段内顺序）
        ratios: 初始 count_distr（float64，= count / count.sum()，与
            legacy ``initial_binning`` 的输出列逐位一致）
        epsilon: IV 计算的 0 计数替换值（legacy ``self.epsilon``）
        bin_num_limit/min_iv_inc/count_distr_limit/ensure_monotonic:
            与 ``TreeOptimBin`` 同名参数一致

    返回:
        seg_bounds: int64 数组 [0, b1, ..., k]
    """
    good = np.asarray(good, dtype='int64')
    bad = np.asarray(bad, dtype='int64')
    ratios = np.asarray(ratios, dtype='float64')
    k = int(good.shape[0])

    pre_g = np.zeros(k + 1, dtype='int64')
    pre_b = np.zeros(k + 1, dtype='int64')
    if k > 0:
        np.cumsum(good, out=pre_g[1:])
        np.cumsum(bad, out=pre_b[1:])

    bounds: List[int] = [0, k]
    cp = np.zeros(k, dtype=bool)
    if k > 0:
        cp[k - 1] = True
    last_iv = 0.0  # legacy 初始为 int 0；与 float 运算逐位等价

    while (len(bounds) - 1) <= bin_num_limit:
        cand = np.flatnonzero(~cp)
        if cand.size == 0:
            break

        bounds_arr = np.asarray(bounds, dtype='int64')
        n_seg = bounds_arr.shape[0] - 1
        seg_of = np.searchsorted(bounds_arr, cand, side='right') - 1
        s_of = bounds_arr[seg_of]

        # ---- count_distr 约束（仅新产生的两个子段需检查：未变段在既往
        # 轮次已通过，且 Kahan 段和是确定性函数，结果不变）----
        distr_ok = np.ones(cand.size, dtype=bool)
        for si in range(n_seg):
            m = seg_of == si
            if not m.any():
                continue
            s, e = bounds[si], bounds[si + 1]
            c_loc = cand[m]
            # 左子段 [s..c]：定起点前缀 Kahan 扫描
            pref = kahan_prefix_sums(ratios, s, e - 1)
            left = pref[c_loc - s] > count_distr_limit
            # 右子段 [c+1..e)：候选批向量化前向 Kahan
            right = kahan_suffix_sums_batched(
                ratios, s, e, c_loc) > count_distr_limit
            distr_ok[m] = left & right

        # ---- 候选分区 IV / bad_prob（矩阵批量；算式与 legacy 逐位一致）----
        seg_g = pre_g[bounds_arr[1:]] - pre_g[bounds_arr[:-1]]
        seg_b = pre_b[bounds_arr[1:]] - pre_b[bounds_arr[:-1]]
        gsub_seg = np.where(seg_g == 0, epsilon, seg_g).astype('float64')
        bsub_seg = np.where(seg_b == 0, epsilon, seg_b).astype('float64')

        gL = pre_g[cand + 1] - pre_g[s_of]
        bL = pre_b[cand + 1] - pre_b[s_of]
        gR = seg_g[seg_of] - gL
        bR = seg_b[seg_of] - bL

        C = cand.size
        T = n_seg + 1
        rows = np.arange(C)
        t_idx = np.arange(T)
        # 列 t：t <= j → 原段 t；t >= j+1 → 原段 t-1；j/j+1 列随后覆写为左右子段
        src = np.where(t_idx[None, :] <= seg_of[:, None],
                       t_idx[None, :], t_idx[None, :] - 1)
        src = np.clip(src, 0, n_seg - 1)

        Gsub = gsub_seg[src]
        Bsub = bsub_seg[src]
        Gsub[rows, seg_of] = np.where(gL == 0, epsilon, gL).astype('float64')
        Gsub[rows, seg_of + 1] = np.where(gR == 0, epsilon, gR).astype('float64')
        Bsub[rows, seg_of] = np.where(bL == 0, epsilon, bL).astype('float64')
        Bsub[rows, seg_of + 1] = np.where(bR == 0, epsilon, bR).astype('float64')

        G = Gsub.sum(axis=1)
        B = Bsub.sum(axis=1)
        gd = Gsub / G[:, None]
        bd = Bsub / B[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            iv_c = ((gd - bd) * np.log(gd / bd)).sum(axis=1)

        # ---- 接受条件（算式与运算次序同 legacy）----
        rel_inc = ((iv_c - last_iv) + 1e-8) / (last_iv + 1e-8)
        eligible = distr_ok & (rel_inc > min_iv_inc)

        if ensure_monotonic and eligible.any():
            seg_cnt = seg_g + seg_b
            CntM = seg_cnt[src].astype('float64')
            CntM[rows, seg_of] = (gL + bL).astype('float64')
            CntM[rows, seg_of + 1] = (gR + bR).astype('float64')
            BadM = seg_b[src].astype('float64')
            BadM[rows, seg_of] = bL.astype('float64')
            BadM[rows, seg_of + 1] = bR.astype('float64')
            with np.errstate(divide='ignore', invalid='ignore'):
                bp = BadM / CntM
            d = np.diff(bp, axis=1)
            # utils.monotonic 语义：任何 NaN → 非单调；等值 → 单调
            mono_ok = np.all(d >= 0, axis=1) | np.all(d <= 0, axis=1)
            eligible = eligible & mono_ok

        if not eligible.any():
            break

        masked = np.where(eligible, iv_c, -np.inf)
        best = int(np.argmax(masked))   # 并列取最小候选索引（cand 升序）
        best_idx = int(cand[best])
        last_iv = float(iv_c[best])

        pos = int(np.searchsorted(bounds_arr, best_idx, side='right'))
        bounds.insert(pos, best_idx + 1)
        cp[best_idx] = True

    return np.asarray(bounds, dtype='int64')


# --------------------------------------------------------------------------- #
# ChiMerge 内核
# --------------------------------------------------------------------------- #

def chi2_pair_stats(good: np.ndarray, bad: np.ndarray) -> np.ndarray:
    """相邻分箱对 (i-1, i) 的 Yates 修正 χ² 数组（位置 0 为 NaN）。

    与 legacy ``ChiMergeOptimBin.chi2_stat``（scipy
    ``chi2_contingency(correction=True)``）逐位一致：

    * 2×2 表为 ``[[good_i, bad_i], [good_{i-1}, bad_{i-1}]]``；
    * 首行（无 lag，NaN）→ NaN；
    * 行/列边际含 0 → 0.0（legacy 在调用 scipy 前显式短路）；
    * 其余 = 闭式 Yates：``Σ max(0, |O-E|-0.5)² / E``，E 由边际外积/总和
      得到；求和按 scipy 的 2×2 flatten（行主序）左结合次序
      ``((t00+t01)+t10)+t11``。已用多组边界表（含过修正区 D<n/2 →
      scipy 精确返回 0）与 scipy 输出逐位比对验证。计数为整数 →
      边际与总和的浮点转换精确，结合序无关。
    """
    good = np.asarray(good)
    bad = np.asarray(bad)
    n = good.shape[0]
    chi2 = np.full(n, np.nan, dtype='float64')
    if n < 2:
        return chi2

    a = good[1:].astype('float64')    # good_i
    b = bad[1:].astype('float64')     # bad_i
    c = good[:-1].astype('float64')   # good_{i-1}
    d = bad[:-1].astype('float64')    # bad_{i-1}

    r0 = a + b
    r1 = c + d
    c0 = a + c
    c1 = b + d
    tot = r0 + r1

    with np.errstate(divide='ignore', invalid='ignore'):
        e00 = r0 * c0 / tot
        e01 = r0 * c1 / tot
        e10 = r1 * c0 / tot
        e11 = r1 * c1 / tot
        t00 = np.maximum(np.abs(a - e00) - 0.5, 0.0)
        t01 = np.maximum(np.abs(b - e01) - 0.5, 0.0)
        t10 = np.maximum(np.abs(c - e10) - 0.5, 0.0)
        t11 = np.maximum(np.abs(d - e11) - 0.5, 0.0)
        val = ((t00 * t00 / e00 + t01 * t01 / e01)
               + t10 * t10 / e10) + t11 * t11 / e11

    zero_marginal = (r0 == 0) | (r1 == 0) | (c0 == 0) | (c1 == 0)
    chi2[1:] = np.where(zero_marginal, 0.0, val)
    return chi2


def chi2_merge_search(
    good: np.ndarray,
    bad: np.ndarray,
    ratios: np.ndarray,
    chi2_limit: float,
    count_distr_limit: float,
    bin_num_limit: int,
) -> np.ndarray:
    """ChiMergeOptimBin 合并循环的 NumPy 精确等价实现。

    复刻 legacy 决策语义（逐条对应 W1 develop
    ``ChiMergeOptimBin.woebin``）：

    * 每轮重算全部相邻对 χ²（首行 NaN；``min`` 为 pandas skipna 语义，
      即对 chi2[1:] 取最小；单分箱时 min=NaN → 各分支条件为 False）；
    * 分支优先级：``min_chi2 < chi2_limit`` → 取 χ² 等于最小值的**首个**
      行；否则 ``min_count_distr < count_distr_limit`` → 取 count_distr
      最小值首行，且 ``idx == 0`` 或（``idx < n-1`` 且
      ``chi2[idx] > chi2[idx+1]``）时 ``idx += 1``；否则
      ``n_bins > bin_num_limit`` → 同分支一取最小 χ² 首行；否则终止；
    * 合并：idx 并入 idx-1 —— good/bad 整数加法（精确）、count_distr
      为**标量浮点左加**（复刻 legacy 增量维护的舍入路径，区别于
      tree 的 Kahan 段和）、bin_chr 以 ``'%,%'`` 拼接（在组装层完成）；
    * 数值型 bin_chr 的逐轮正则折叠只影响中间字符串，最终 breaks 恒为
      各段末区间右边界（见 :func:`segments_to_breaks` 的等价性说明）。

    参数:
        good/bad: 初始计数表（int64）
        ratios: 初始 count_distr（float64）
        chi2_limit: ``chi2.isf(p, df=1)``
        count_distr_limit/bin_num_limit: 同 legacy

    返回:
        seg_bounds: int64 数组 [0, b1, ..., k]
    """
    good = np.array(good, dtype='int64')
    bad = np.array(bad, dtype='int64')
    distr = np.array(ratios, dtype='float64')
    k = int(good.shape[0])
    bounds = list(range(k + 1))

    chi2 = chi2_pair_stats(good, bad)

    while True:
        n = int(good.shape[0])
        min_chi2 = float(np.min(chi2[1:])) if n >= 2 else np.nan
        min_distr = float(np.min(distr))

        if min_chi2 < chi2_limit:
            # 分箱坏占比差异不显著（NaN < x 恒为 False，与 pandas 一致）
            idx = 1 + int(np.argmin(chi2[1:]))
        elif min_distr < count_distr_limit:
            # 分箱占比过少
            idx = int(np.argmin(distr))
            if idx == 0 or (idx < n - 1 and chi2[idx] > chi2[idx + 1]):
                idx = idx + 1
        elif n > bin_num_limit:
            # 分箱数太多
            idx = 1 + int(np.argmin(chi2[1:]))
        else:
            break

        # 合并 idx 到 idx-1（增量维护，与 legacy 标量运算逐位一致）
        good[idx - 1] = good[idx - 1] + good[idx]
        bad[idx - 1] = bad[idx - 1] + bad[idx]
        distr[idx - 1] = distr[idx - 1] + distr[idx]
        good = np.delete(good, idx)
        bad = np.delete(bad, idx)
        distr = np.delete(distr, idx)
        del bounds[idx]

        chi2 = chi2_pair_stats(good, bad)

    return np.asarray(bounds, dtype='int64')


# --------------------------------------------------------------------------- #
# 段边界 → breaks 组装（legacy 语义）
# --------------------------------------------------------------------------- #

def segments_to_breaks(
    bin_chr: np.ndarray,
    is_numeric: bool,
    seg_bounds: np.ndarray,
    categories: tuple = None,
) -> pd.Series:
    """把段边界组装为 legacy 语义的 breaks Series。

    数值型：breaks = 各段**最后一个**初始区间标签的右边界字符串经
    ``pd.to_numeric`` 转 float64。与 legacy（正则折叠
    ``,[.\\d]+\\)%,%\\[[.\\d]+,`` 后取 ``^\\[(.*), *(.*)\\)`` 的 group(2)）
    恒等：折叠成功时 group(2) 即末区间右边界；内边界为负数/科学计数导致
    折叠正则失配时，贪婪匹配的 group(2) 仍是末区间右边界（已逐例验证，
    差分测试覆盖负数边界场景）。

    类别型：段内分箱名按行序以 ``'%,%'`` 拼接。**dtype 复刻 legacy 的
    groupby-agg 行为**（W2 Phase 3 差分发现的 pandas 语义）：

    * 若所有段拼接结果都属于初始 categories（即每段都是"单一初始类别"，
      无实际合并）→ legacy 的 ``agg(bin_chr=join)`` 保持 **category
      dtype**（categories/ordered 与输入一致，values 为段序）。下游
      ``binning_breaks`` 的 ``cat.set_categories(breaks)`` 对 categorical
      入参采用其 **categories**（= 初始 breaks 顺序）而非 values ——
      最终分箱行序为初始顺序；
    * 任一段发生真实合并（拼接串不在 categories 中）→ legacy agg 退化为
      str dtype，``set_categories`` 采用 values（段序）。

    返回 Series 的 ``name`` 固定为 ``'bin_chr'``（legacy 返回
    ``best_binning['bin_chr']`` 同名列）。
    """
    bin_chr = np.asarray(bin_chr, dtype=object)
    segs = [(int(a), int(b)) for a, b in
            zip(np.asarray(seg_bounds)[:-1], np.asarray(seg_bounds)[1:])]

    if is_numeric:
        labels = []
        for a, b in segs:
            last = str(bin_chr[b - 1])
            labels.append(last[last.rindex(',') + 1:-1])
        return pd.to_numeric(
            pd.Series(labels, dtype=object, name='bin_chr'))

    joined = ['%,%'.join([str(x) for x in bin_chr[a:b]]) for a, b in segs]
    if categories is not None:
        cat_list = [str(c) for c in categories]
        cat_set = set(cat_list)
        if all(j in cat_set for j in joined):
            return pd.Series(
                pd.Categorical(joined, categories=cat_list, ordered=True),
                name='bin_chr')
    return pd.Series(joined, name='bin_chr')

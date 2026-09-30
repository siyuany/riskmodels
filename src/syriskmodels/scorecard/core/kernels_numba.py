# -*- encoding: utf-8 -*-
"""Numba 编译分箱内核（W2 Phase 7）。

与 :mod:`syriskmodels.scorecard.core.kernels` 的 NumPy 参考实现**逐位一致**
（差分测试见 ``test/test_binning_equivalence.py``），约束（任务书 §7.2）：

* ``@njit(cache=True, nogil=True, fastmath=False)`` —— fastmath 必须为
  False（任何重结合/倒数近似都会破坏与 legacy 的逐位一致性）；
* 内核只接受**定长连续数组 + 标量**，输出写入预分配的段边界数组；
  不在内核内处理 pandas/object/变长字符串（bin_chr 组装在 Python 边界层）；
* 类别变量天然已编码：内核消费的是 BinCountTable 的 good/bad/ratios
  数值数组，字符串只存在于边界层；
* 浮点逐位一致的关键复刻点：

  - ``_nb_pairwise_sum``：numpy ``np.sum`` 对 float64 的 pairwise 归约在
    **n ≤ 8** 时为纯标量路径（n<8 顺序累加；n==8 八车道 +
    ``((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))`` 结合，尾部并入车道 0），
    跨平台可复刻；**n ≥ 9 时 numpy 走 SIMD 相关路径，不可移植复刻** ——
    因此 tree 内核仅在 ``bin_num_limit ≤ 6``（分区段数 T = 段数+1 ≤ 8）
    时启用 Numba（``kernels.resolve_engine`` 强制该护栏，超出自动降级
    NumPy 后端并告警）。n ≤ 8 的逐位一致性经 40 万+ 对抗样本探针与
    差分 fuzz 验证；
  - ``np.log``：numba 降级为 libm 标量调用，与 numpy 元素级 ``np.log``
    在本项目支持的平台上逐位一致（arm64 实测 0 失配 / 40 万样本；
    CI x86_64 由差分 fuzz 把关）；
  - Kahan 补偿求和、χ² 四元素左结合求和、标量增量浮点加法均与
    NumPy 参考实现同序同算式。

* Rule 内核不在此模块：其热路径依赖 ``scipy.stats.fisher_exact``
  （无法进入 njit），且 NumPy 向量化版已达标（任务书 §7.5 的 Numba
  性能目标只含 tree/chi2）—— RuleOptimBin 的 engine 恒解析为 'numpy'。

预热与缓存（任务书 §7.6）
------------------------
* 本模块**不在任何生产模块导入时被加载**：仅当 ``resolve_engine`` 判定
  需要 Numba 后端时才 import（numba 包自身 import 约 0.3–1s，一次性）；
* ``cache=True``：首次调用触发 JIT 编译（每内核约 0.5–2s），编译产物
  缓存于本文件旁 ``__pycache__/*.nbi``；同环境的后续**新进程**直接加载
  缓存（约 50–100ms）。缓存失效（numba/numpy/源码版本变化）时自动重编译；
* auto 策略下小数据（初始分箱 < 64 或样本 < 5000）不走 Numba，
  JIT/加载固定成本不会拖累小任务；benchmark 的 warmup 轮会吸收首进程
  编译成本；
* 不与 multiprocessing 混用：``woebin(no_cores>1)`` 的进程并行与本模块
  正交（各 worker 进程独立加载缓存），内核自身为单线程（``nogil=True``
  已释放 GIL，未来可用线程/``prange`` 扩展，不引入 mp）。
"""
import numpy as np

try:  # pragma: no cover - 取决于环境
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    njit = None
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:

    @njit(cache=True, nogil=True, fastmath=False)
    def _nb_pairwise_sum(a, n):
        """numpy ``np.sum``（float64, n ≤ 8）的逐位复刻。

        n < 8：``res = 0.0`` 起顺序累加；n == 8：八车道累加 +
        ``((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))``。调用方（tree 内核）
        保证 n ≤ 8（resolve_engine 的 bin_num_limit ≤ 6 护栏）。
        """
        if n < 8:
            res = 0.0
            for i in range(n):
                res += a[i]
            return res
        r0 = a[0]
        r1 = a[1]
        r2 = a[2]
        r3 = a[3]
        r4 = a[4]
        r5 = a[5]
        r6 = a[6]
        r7 = a[7]
        i = 8
        while i < n - (n % 8):
            r0 += a[i]
            r1 += a[i + 1]
            r2 += a[i + 2]
            r3 += a[i + 3]
            r4 += a[i + 4]
            r5 += a[i + 5]
            r6 += a[i + 6]
            r7 += a[i + 7]
            i += 8
        while i < n:
            r0 += a[i]
            i += 1
        return ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))

    @njit(cache=True, nogil=True, fastmath=False)
    def _tree_cut_search_nb(good, bad, ratios, epsilon, bin_num_limit,
                            min_iv_inc, count_distr_limit, ensure_monotonic,
                            bounds_out):
        """tree_cut_search 的 Numba 版（逐位一致；bin_num_limit ≤ 6）。"""
        k = good.shape[0]
        pre_g = np.zeros(k + 1, np.int64)
        pre_b = np.zeros(k + 1, np.int64)
        for i in range(k):
            pre_g[i + 1] = pre_g[i] + good[i]
            pre_b[i + 1] = pre_b[i] + bad[i]

        bounds_out[0] = 0
        bounds_out[1] = k
        n_bounds = 2

        cp = np.zeros(k, np.bool_)
        if k > 0:
            cp[k - 1] = True
        last_iv = 0.0

        cand = np.empty(max(k, 1), np.int64)
        seg_of = np.empty(max(k, 1), np.int64)
        distr_ok = np.empty(max(k, 1), np.bool_)
        pref = np.empty(max(k, 1), np.float64)

        max_t = bin_num_limit + 3
        gsub = np.empty(max_t, np.float64)
        bsub = np.empty(max_t, np.float64)
        cntw = np.empty(max_t, np.float64)
        badw = np.empty(max_t, np.float64)
        terms = np.empty(max_t, np.float64)

        while (n_bounds - 1) <= bin_num_limit:
            n_seg = n_bounds - 1
            n_cand = 0
            for i in range(k):
                if not cp[i]:
                    cand[n_cand] = i
                    n_cand += 1
            if n_cand == 0:
                break

            for ci in range(n_cand):
                idx = cand[ci]
                j = 0
                while bounds_out[j + 1] <= idx:
                    j += 1
                seg_of[ci] = j
                distr_ok[ci] = True

            # ---- count_distr 约束（Kahan 段和，与 pandas groupby.sum 一致）----
            for j in range(n_seg):
                s = bounds_out[j]
                e = bounds_out[j + 1]
                # 左子段 [s..c]：定起点前缀 Kahan 扫描
                S = 0.0
                C = 0.0
                for jj in range(s, e - 1):
                    v = ratios[jj]
                    y = v - C
                    t = S + y
                    C = (t - S) - y
                    S = t
                    pref[jj - s] = S
                for ci in range(n_cand):
                    if seg_of[ci] != j:
                        continue
                    idx = cand[ci]
                    left = pref[idx - s]
                    # 右子段 [idx+1..e)：前向 Kahan
                    S2 = 0.0
                    C2 = 0.0
                    for jj in range(idx + 1, e):
                        v = ratios[jj]
                        y = v - C2
                        t = S2 + y
                        C2 = (t - S2) - y
                        S2 = t
                    distr_ok[ci] = (left > count_distr_limit) and \
                        (S2 > count_distr_limit)

            # ---- 候选评估 ----
            best_val = -np.inf
            best_ci = -1
            for ci in range(n_cand):
                if not distr_ok[ci]:
                    continue
                idx = cand[ci]
                j = seg_of[ci]
                s = bounds_out[j]
                e = bounds_out[j + 1]

                gL = pre_g[idx + 1] - pre_g[s]
                bL = pre_b[idx + 1] - pre_b[s]
                gR = (pre_g[e] - pre_g[s]) - gL
                bR = (pre_b[e] - pre_b[s]) - bL

                T = n_seg + 1
                for t in range(T):
                    if t < j:
                        gs = pre_g[bounds_out[t + 1]] - pre_g[bounds_out[t]]
                        bs = pre_b[bounds_out[t + 1]] - pre_b[bounds_out[t]]
                    elif t == j:
                        gs = gL
                        bs = bL
                    elif t == j + 1:
                        gs = gR
                        bs = bR
                    else:
                        gs = pre_g[bounds_out[t]] - pre_g[bounds_out[t - 1]]
                        bs = pre_b[bounds_out[t]] - pre_b[bounds_out[t - 1]]
                    if gs == 0:
                        gsub[t] = epsilon
                    else:
                        gsub[t] = float(gs)
                    if bs == 0:
                        bsub[t] = epsilon
                    else:
                        bsub[t] = float(bs)
                    cntw[t] = float(gs + bs)   # count = good + bad
                    badw[t] = float(bs)

                G = _nb_pairwise_sum(gsub, T)
                B = _nb_pairwise_sum(bsub, T)
                for t in range(T):
                    gd = gsub[t] / G
                    bd = bsub[t] / B
                    terms[t] = (gd - bd) * np.log(gd / bd)
                iv = _nb_pairwise_sum(terms, T)

                rel = ((iv - last_iv) + 1e-8) / (last_iv + 1e-8)
                if rel <= min_iv_inc:
                    continue

                if ensure_monotonic:
                    mono_ok = True
                    inc = True
                    dec = True
                    prev = badw[0] / cntw[0] if cntw[0] != 0.0 else np.nan
                    if np.isnan(prev):
                        mono_ok = False
                    for t in range(1, T):
                        if not mono_ok:
                            break
                        cur = badw[t] / cntw[t] if cntw[t] != 0.0 else np.nan
                        if np.isnan(cur):
                            mono_ok = False
                            break
                        d = cur - prev
                        if not (d >= 0.0):
                            inc = False
                        if not (d <= 0.0):
                            dec = False
                        prev = cur
                    if mono_ok and not (inc or dec):
                        mono_ok = False
                    if not mono_ok:
                        continue

                if iv > best_val:
                    best_val = iv
                    best_ci = ci

            if best_ci < 0:
                break

            best_idx = cand[best_ci]
            last_iv = best_val
            j = seg_of[best_ci]
            for t in range(n_bounds, j + 1, -1):
                bounds_out[t] = bounds_out[t - 1]
            bounds_out[j + 1] = best_idx + 1
            n_bounds += 1
            cp[best_idx] = True

        return n_bounds

    @njit(cache=True, nogil=True, fastmath=False)
    def _chi2_merge_search_nb(good0, bad0, ratios0, chi2_limit,
                              count_distr_limit, bin_num_limit, bounds_out):
        """chi2_merge_search 的 Numba 版（逐位一致；无 bin_num_limit 限制）。"""
        k = good0.shape[0]
        good = good0.copy()
        bad = bad0.copy()
        distr = ratios0.copy()
        for i in range(k + 1):
            bounds_out[i] = i
        n = k
        chi2 = np.empty(max(k, 1), np.float64)

        while True:
            for i in range(1, n):
                a = float(good[i])
                b = float(bad[i])
                c = float(good[i - 1])
                d = float(bad[i - 1])
                r0 = a + b
                r1 = c + d
                c0 = a + c
                c1 = b + d
                if r0 == 0.0 or r1 == 0.0 or c0 == 0.0 or c1 == 0.0:
                    chi2[i] = 0.0
                    continue
                tot = r0 + r1
                e00 = r0 * c0 / tot
                e01 = r0 * c1 / tot
                e10 = r1 * c0 / tot
                e11 = r1 * c1 / tot
                t00 = abs(a - e00) - 0.5
                if t00 < 0.0:
                    t00 = 0.0
                t01 = abs(b - e01) - 0.5
                if t01 < 0.0:
                    t01 = 0.0
                t10 = abs(c - e10) - 0.5
                if t10 < 0.0:
                    t10 = 0.0
                t11 = abs(d - e11) - 0.5
                if t11 < 0.0:
                    t11 = 0.0
                chi2[i] = ((t00 * t00 / e00 + t01 * t01 / e01)
                           + t10 * t10 / e10) + t11 * t11 / e11

            min_chi2 = np.nan
            arg_chi2 = -1
            for i in range(1, n):
                if arg_chi2 < 0 or chi2[i] < min_chi2:
                    min_chi2 = chi2[i]
                    arg_chi2 = i
            min_distr = distr[0] if n > 0 else np.nan
            arg_distr = 0
            for i in range(1, n):
                if distr[i] < min_distr:
                    min_distr = distr[i]
                    arg_distr = i

            if n >= 2 and min_chi2 < chi2_limit:
                idx = arg_chi2
            elif n >= 1 and min_distr < count_distr_limit:
                idx = arg_distr
                if idx == 0 or (idx < n - 1 and chi2[idx] > chi2[idx + 1]):
                    idx = idx + 1
            elif n > bin_num_limit:
                if arg_chi2 < 0:
                    raise ValueError(
                        'chi2 merge: no pair available (legacy IndexError '
                        'equivalent)')
                idx = arg_chi2
            else:
                break

            good[idx - 1] = good[idx - 1] + good[idx]
            bad[idx - 1] = bad[idx - 1] + bad[idx]
            distr[idx - 1] = distr[idx - 1] + distr[idx]
            for t in range(idx, n - 1):
                good[t] = good[t + 1]
                bad[t] = bad[t + 1]
                distr[t] = distr[t + 1]
            n -= 1
            # 删除 bounds[idx]：t ∈ [idx, n]（n 已自减；活跃边界为 0..n）
            for t in range(idx, n + 1):
                bounds_out[t] = bounds_out[t + 1]

        return n + 1


# --------------------------------------------------------------------------- #
# Python 边界层（定长连续数组进出；不做任何数值运算）
# --------------------------------------------------------------------------- #

def tree_cut_search_numba(good, bad, ratios, epsilon, bin_num_limit,
                          min_iv_inc, count_distr_limit, ensure_monotonic):
    """``kernels.tree_cut_search`` 的 Numba 后端（签名与返回值一致）。

    调用前提（由 ``kernels.resolve_engine`` 保证）：numba 可用且
    ``bin_num_limit ≤ 6``（分区求和长度 ≤ 8 的逐位一致护栏）。
    """
    good = np.ascontiguousarray(good, dtype='int64')
    bad = np.ascontiguousarray(bad, dtype='int64')
    ratios = np.ascontiguousarray(ratios, dtype='float64')
    k = good.shape[0]
    bounds_out = np.empty(k + 2, dtype='int64')
    m = _tree_cut_search_nb(
        good, bad, ratios, float(epsilon), int(bin_num_limit),
        float(min_iv_inc), float(count_distr_limit),
        bool(ensure_monotonic), bounds_out)
    return bounds_out[:m].copy()


def chi2_merge_search_numba(good, bad, ratios, chi2_limit,
                            count_distr_limit, bin_num_limit):
    """``kernels.chi2_merge_search`` 的 Numba 后端（签名与返回值一致）。"""
    good = np.ascontiguousarray(good, dtype='int64')
    bad = np.ascontiguousarray(bad, dtype='int64')
    ratios = np.ascontiguousarray(ratios, dtype='float64')
    k = good.shape[0]
    bounds_out = np.empty(k + 1, dtype='int64')
    m = _chi2_merge_search_nb(
        good, bad, ratios, float(chi2_limit), float(count_distr_limit),
        int(bin_num_limit), bounds_out)
    return bounds_out[:m].copy()

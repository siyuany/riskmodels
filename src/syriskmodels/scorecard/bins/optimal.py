# -*- encoding: utf-8 -*-
"""
粗分箱模块

提供 ChiMerge、决策树、规则等最优分箱方法
"""
import re
from typing import Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import chi2, chi2_contingency, fisher_exact

from syriskmodels.scorecard.core.base import WOEBin, OptimBinMixin
from syriskmodels.scorecard.core.factory import WOEBinFactory
from syriskmodels.utils import monotonic


@WOEBinFactory.register(['chi2', 'chimerge'])
class ChiMergeOptimBin(WOEBin, OptimBinMixin):
    """ChiMerge 最优分箱

    对相邻分箱进行 Chi2 列联表独立性检验，基于检验的统计量进行分箱合并。

    参数:
        bin_num_limit: 分箱数上限，默认 5
        p: 独立性检验显著性，默认 0.05
        count_distr_limit: 最小分箱样本占比，默认 0.02
        ensure_monotonic: 是否要求单调，默认 False（暂不支持）
    """

    def __init__(self,
                 bin_num_limit: int = 5,
                 p: float = 0.05,
                 count_distr_limit: float = 0.02,
                 ensure_monotonic: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.bin_num_limit = bin_num_limit
        self.p = p
        self.count_distr_limit = count_distr_limit
        self.ensure_monotonic = ensure_monotonic
        self.chi2_limit = chi2.isf(p, df=1)

    @staticmethod
    def chi2_stat(binning):
        """计算两分箱之间的 Chi2 统计量，使用 Yate's 连续性修正"""
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
        """执行 ChiMerge 分箱。

        W2 Phase 4：合并循环改为 NumPy 精确等价实现
        （:func:`syriskmodels.scorecard.core.kernels.chi2_merge_search` +
        :func:`~syriskmodels.scorecard.core.kernels.chi2_pair_stats`），
        与 legacy pandas 实现（保留在 ``chi2_stat`` 及
        test/test_binning_equivalence.py 的参考拷贝中）在 χ² 公式
        （scipy Yates 修正闭式复刻）、三分支决策优先级、idx 修正规则、
        tie-breaking（取最小索引）、count_distr 标量增量维护、
        最终 breaks 提取等全部语义上逐位一致。
        """
        assert breaks is not None, \
            f"使用{self.__class__.__name__}类进行分箱，需要传入初始分箱（细分箱）结果"
        from syriskmodels.scorecard.core.kernels import (
            chi2_merge_search,
            segments_to_breaks,
        )

        table = self.initial_count_table(dtm, breaks)
        count = table.count
        # 与 legacy initial_binning 的 count_distr 列逐位一致
        ratios = count / count.sum()

        seg_bounds = chi2_merge_search(
            table.good,
            table.bad,
            ratios,
            chi2_limit=self.chi2_limit,
            count_distr_limit=self.count_distr_limit,
            bin_num_limit=self.bin_num_limit,
        )

        # legacy ChiMerge 在入口即 bin_chr.astype('str')，最终 breaks 恒为
        # str/float dtype（无 tree 的 category-dtype 语义），categories 传 None
        return segments_to_breaks(
            table.bin_chr, table.is_numeric, seg_bounds)


@WOEBinFactory.register('tree')
class TreeOptimBin(WOEBin, OptimBinMixin):
    """树分箱方法，从细分箱生成的切分点中挑选最优切分点，自顶向下逐步生成分箱树，完成分箱。

    算法：使用 node_id 和 cp (cut point) 标记机制，通过贪心搜索寻找使 IV 增量最大的
    切分点，直到满足停止条件。

    参数:
        bin_num_limit: 分箱数上限，默认 5
        min_iv_inc: 增加切分点后 IV 相对增幅最小值，默认 0.05
        count_distr_limit: 最小分箱样本占比，默认 0.02
        ensure_monotonic: 是否要求严格单调，默认 False
    """

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
        """执行树分箱。

        W2 Phase 3：搜索内核改为 NumPy 精确等价实现
        （:func:`syriskmodels.scorecard.core.kernels.tree_cut_search`），
        与 legacy pandas 实现（保留在 ``merge_binning`` / ``node_split`` /
        ``iv`` 及 test/test_binning_equivalence.py 的参考拷贝中）在
        cp 标记、Kahan count_distr 段和、IV 算式、单调约束 NaN 语义、
        接受条件、tie-breaking（并列取最小索引）、段数上限 off-by-one、
        breaks 提取等全部语义上逐位一致。
        """
        assert breaks is not None, \
            f"使用{self.__class__.__name__}类进行分箱，需要传入初始分箱（细分箱）结果"
        from syriskmodels.scorecard.core.kernels import (
            segments_to_breaks,
            tree_cut_search,
        )

        table = self.initial_count_table(dtm, breaks)
        count = table.count
        # 与 legacy initial_binning 的 count_distr 列逐位一致（同算式同顺序）
        ratios = count / count.sum()

        seg_bounds = tree_cut_search(
            table.good,
            table.bad,
            ratios,
            epsilon=self.epsilon,
            bin_num_limit=self.bin_num_limit,
            min_iv_inc=self.min_iv_inc,
            count_distr_limit=self.count_distr_limit,
            ensure_monotonic=self.ensure_monotonic,
        )

        return segments_to_breaks(
            table.bin_chr, table.is_numeric, seg_bounds,
            categories=table.categories)

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


@WOEBinFactory.register('rule')
class RuleOptimBin(WOEBin, OptimBinMixin):
    """规则优化分箱算法，用于生成单变量规则。该分箱方式会生成三个分箱（不包含特殊值分箱），
    分别为拒绝分箱、监控分箱、通过分箱。其中拒绝分箱的坏率提升度需大于 `min_lift`，通过
    分箱为紧邻拒绝分箱、占比>5%的样本，其余为通过分箱。建议上游细分箱方法采用
    `QuantileInitBin`，且`initial_bins > 20`。

    * 假设检验：拒绝分箱坏率显著高于整体坏率（alpha=0.05），不满足时无法分箱
    * 最小命中样本数：拒绝分箱最少样本数，默认不限制，建议设置为50以上，不满足时无法分箱

    参数:
        lift: 风险阈值，默认为 3
        min_hit_samples: 最小命中样本数，默认为 None 代表不限制命中样本数
        direction: 规则挖掘方向, good - 挖掘好客户, bad - 挖掘坏客户
    """

    def __init__(self,
                 lift: float = 3,
                 min_hit_samples: Optional[int] = None,
                 pvalue: float = 0.05,
                 direction: str = 'bad',
                 eps: float = 1e-8,
                 **kwargs):
        # B-7：接收并透传 **kwargs（与 TreeOptimBin/ChiMergeOptimBin 一致），
        # 使 WOEBinFactory 的 kwargs 分发机制可用。
        # 语义决策（W2 报告 §B-7）：本类的 ``eps`` 保持历史含义 ——
        # lift 计算的平滑项（(bad_prob + eps) / bad_prob_all），默认 1e-8，
        # **不**转发给基类；基类 ``epsilon``（WOE 零计数替换值）保持默认
        # 0.5。若把 eps 转发给基类会把 rule 分箱 WOE/IV 的零替换值从 0.5
        # 变为 1e-8，改变默认分箱输出，违反 W2「默认不改变分箱输出」约束。
        super().__init__(**kwargs)
        self._min_lift = lift
        self._min_hit_samples = min_hit_samples or 0
        self._p = pvalue
        self._eps = eps
        assert direction in ['good', 'bad'], '挖掘方向为good/bad两者之一'
        self._direction = direction

    def cut_binning(self, binning, idx):
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

        if is_numeric_dtype(dtm['value']):
            best_binning['bin_chr'] = best_binning['bin_chr'].apply(
                lambda x: re.sub(r',[.\d]+\)%,%\[[.\d]+,', ',', x))
            _pattern = re.compile(r"^\[(.*), *(.*)\)")
            breaks = best_binning['bin_chr'].apply(
                lambda x: _pattern.match(x)[2])
            breaks = pd.to_numeric(breaks)
        else:
            breaks = best_binning['bin_chr']

        return breaks

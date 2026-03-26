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
        """执行 ChiMerge 分箱"""
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

            if is_numeric_dtype(dtm['value']):
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
        if is_numeric_dtype(dtm['value']):
            _pattern = re.compile(r"^\[(.*), *(.*)\)")
            breaks = binning_chi2['bin_chr'].apply(
                lambda x: _pattern.match(x)[2])
            breaks = pd.to_numeric(breaks)
        else:
            breaks = binning_chi2['bin_chr']

        return breaks


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
                 eps: float = 1e-8):
        super().__init__()
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

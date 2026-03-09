# -*- encoding: utf-8 -*-
"""
粗分箱模块

提供 ChiMerge、决策树、规则等最优分箱方法
"""
import re
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import chi2, chi2_contingency

from syriskmodels.scorecard.core.base import WOEBin, OptimBinMixin
from syriskmodels.scorecard.core.factory import WOEBinFactory
from scipy.stats import chi2, chi2_contingency


# 扩展 OptimBinMixin，添加 initial_binning 方法
class _OptimBinMixinExt(OptimBinMixin):
    def initial_binning(self, dtm, breaks):
        binning = self.binning_breaks(dtm, breaks)
        binning['count'] = binning['good'] + binning['bad']
        binning['count_distr'] = binning['count'] / binning['count'].sum()
        if not is_numeric_dtype(dtm['value']):
            binning['badprob'] = binning['bad'] / binning['count']
            binning = binning.sort_values(by='badprob', ascending=False).reset_index(drop=True)
        return binning


@WOEBinFactory.register(['chi2', 'chimerge'])
class ChiMergeOptimBin(WOEBin, _OptimBinMixinExt):
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
    def chi2_stat(binning: pd.DataFrame) -> pd.DataFrame:
        """计算两分箱之间的 Chi2 统计量"""
        binning = binning.copy()
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
    
    def woebin(self, dtm: pd.DataFrame, breaks: list = None) -> list:
        """执行 ChiMerge 分箱
        
        参数:
            dtm: 输入数据
            breaks: 细分箱切分点
        
        返回:
            合并后的切分点
        """
        assert breaks is not None, f"使用{self.__class__.__name__}类进行分箱，需要传入初始分箱（细分箱）结果"
        
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
                idx = binning_chi2[binning_chi2['count_distr'] == min_count_distr].index[0]
                if idx == 0 or (idx < len(binning_chi2) - 1 and
                               (binning_chi2['chi2'][idx] > binning_chi2['chi2'][idx + 1])):
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
                binning_chi2.loc[idx, 'count']
            )
            binning_chi2.loc[idx - 1, 'count_distr'] = (
                binning_chi2.loc[idx - 1, 'count_distr'] +
                binning_chi2.loc[idx, 'count_distr']
            )
            binning_chi2.loc[idx - 1, 'good'] = (
                binning_chi2.loc[idx - 1, 'good'] +
                binning_chi2.loc[idx, 'good']
            )
            binning_chi2.loc[idx - 1, 'bad'] = (
                binning_chi2.loc[idx - 1, 'bad'] +
                binning_chi2.loc[idx, 'bad']
            )
            
            if is_numeric_dtype(dtm['value']):
                # 数值类型分箱合并
                binning_chi2['bin_chr'] = binning_chi2['bin_chr'].apply(
                    lambda x: re.sub(r',[.\d]+\)%,%\[[.\d]+,', ',', x)
                )
            
            index = binning_chi2.index.tolist()
            index.remove(idx)
            binning_chi2 = binning_chi2.iloc[index].reset_index(drop=True)
            binning_chi2 = self.chi2_stat(binning_chi2)
        # End of loop
        
        # 切分点提取
        if is_numeric_dtype(dtm['value']):
            _pattern = re.compile(r"^\[(.*), *(.*)\)")
            breaks = binning_chi2['bin_chr'].apply(lambda x: _pattern.match(x)[2])
            breaks = pd.to_numeric(breaks)
        else:
            breaks = binning_chi2['bin_chr']
        
        return breaks.tolist() if isinstance(breaks, (pd.Series, np.ndarray)) else breaks


@WOEBinFactory.register('tree')
class TreeOptimBin(WOEBin, _OptimBinMixinExt):
    """决策树最优分箱
    
    从细分箱生成的切分点中挑选最优切分点，自顶向下逐步生成分箱树。
    
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
    
    def woebin(self, dtm: pd.DataFrame, breaks: list = None) -> list:
        """执行决策树分箱
        
        参数:
            dtm: 输入数据
            breaks: 细分箱切分点
        
        返回:
            最优切分点
        """
        assert breaks is not None, f"使用{self.__class__.__name__}类进行分箱，需要传入初始分箱（细分箱）结果"
        
        binning = self.initial_binning(dtm, breaks)
        
        # 计算初始 IV
        total_iv = self._calculate_iv(binning)
        
        # 自顶向下分箱
        nodes = [binning]
        final_bins = []
        
        while nodes and len(final_bins) < self.bin_num_limit:
            # 选择 IV 最大的节点进行分裂
            best_node = max(nodes, key=lambda x: self._calculate_iv(x))
            nodes.remove(best_node)
            
            # 寻找最佳切分点
            best_split = self._find_best_split(best_node, total_iv)
            
            if best_split is not None:
                # 执行分裂
                left_node, right_node = self._split_node(best_node, best_split)
                nodes.extend([left_node, right_node])
            else:
                # 无法分裂，加入最终分箱
                final_bins.append(best_node)
        
        # 合并最终分箱
        if not final_bins:
            final_bins = nodes
        
        result = pd.concat(final_bins, ignore_index=True)
        
        # 提取切分点
        if is_numeric_dtype(dtm['value']):
            _pattern = re.compile(r"^\[(.*), *(.*)\)")
            breaks = result['bin_chr'].apply(lambda x: _pattern.match(x)[2] if _pattern.match(x) else x)
            breaks = pd.to_numeric(breaks)
        else:
            breaks = result['bin_chr']
        
        return breaks.tolist() if isinstance(breaks, (pd.Series, np.ndarray)) else breaks
    
    def _calculate_iv(self, binning: pd.DataFrame) -> float:
        """计算 IV 值"""
        total_good = binning['good'].sum()
        total_bad = binning['bad'].sum()
        
        def sub0(x):
            return np.where(x == 0, self.epsilon, x)
        
        good_distr = sub0(binning['good']) / total_good
        bad_distr = sub0(binning['bad']) / total_bad
        
        woe = np.log(good_distr / bad_distr)
        iv = ((good_distr - bad_distr) * woe).sum()
        
        return iv
    
    def _find_best_split(self, node: pd.DataFrame, total_iv: float) -> tuple:
        """寻找最佳切分点"""
        if len(node) <= 1:
            return None
        
        # 检查样本占比
        if node['count_distr'].min() < self.count_distr_limit:
            return None
        
        # 计算当前 IV
        current_iv = self._calculate_iv(node)
        
        # 尝试所有可能的切分点
        best_iv = 0
        best_idx = None
        
        for i in range(1, len(node)):
            left = node.iloc[:i]
            right = node.iloc[i:]
            
            iv_left = self._calculate_iv(left)
            iv_right = self._calculate_iv(right)
            iv_split = iv_left + iv_right
            
            # 检查 IV 提升
            iv_inc = (iv_split - current_iv) / current_iv if current_iv > 0 else 0
            
            if iv_inc >= self.min_iv_inc and iv_split > best_iv:
                best_iv = iv_split
                best_idx = i
        
        return best_idx
    
    def _split_node(self, node: pd.DataFrame, idx: int) -> tuple:
        """分裂节点"""
        left = node.iloc[:idx].copy()
        right = node.iloc[idx:].copy()
        
        return left, right


@WOEBinFactory.register('rule')
class RuleOptimBin(WOEBin, _OptimBinMixinExt):
    """规则最优分箱
    
    基于业务规则进行分箱合并。
    
    参数:
        bin_num_limit: 分箱数上限，默认 5
        min_iv_inc: 最小 IV 增益，默认 0.05
        count_distr_limit: 最小分箱样本占比，默认 0.02
    """
    
    def __init__(self,
                 bin_num_limit: int = 5,
                 min_iv_inc: float = 0.05,
                 count_distr_limit: float = 0.02,
                 **kwargs):
        super().__init__(**kwargs)
        self.bin_num_limit = bin_num_limit
        self.min_iv_inc = min_iv_inc
        self.count_distr_limit = count_distr_limit
    
    def woebin(self, dtm: pd.DataFrame, breaks: list = None) -> list:
        """执行规则分箱"""
        assert breaks is not None, f"使用{self.__class__.__name__}类进行分箱，需要传入初始分箱（细分箱）结果"
        
        binning = self.initial_binning(dtm, breaks)
        
        # 简化的规则分箱：基于坏样本率单调性合并
        while len(binning) > self.bin_num_limit:
            # 计算坏样本率
            binning['badprob'] = binning['bad'] / binning['count']
            
            # 找到坏样本率最接近的相邻分箱
            badprob_diff = binning['badprob'].diff().abs()
            min_diff_idx = badprob_diff[1:].idxmin()
            
            # 合并
            binning.loc[min_diff_idx - 1, 'bin_chr'] = '%,%'.join([
                binning.loc[min_diff_idx - 1, 'bin_chr'],
                binning.loc[min_diff_idx, 'bin_chr']
            ])
            binning.loc[min_diff_idx - 1, 'good'] += binning.loc[min_diff_idx, 'good']
            binning.loc[min_diff_idx - 1, 'bad'] += binning.loc[min_diff_idx, 'bad']
            binning.loc[min_diff_idx - 1, 'count'] += binning.loc[min_diff_idx, 'count']
            
            binning = binning.drop(min_diff_idx).reset_index(drop=True)
        
        # 提取切分点
        if is_numeric_dtype(dtm['value']):
            _pattern = re.compile(r"^\[(.*), *(.*)\)")
            breaks = binning['bin_chr'].apply(lambda x: _pattern.match(x)[2] if _pattern.match(x) else x)
            breaks = pd.to_numeric(breaks)
        else:
            breaks = binning['bin_chr']
        
        return breaks.tolist() if isinstance(breaks, (pd.Series, np.ndarray)) else breaks

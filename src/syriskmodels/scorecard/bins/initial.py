# -*- encoding: utf-8 -*-
"""
细分箱模块

提供等频和等宽细分箱方法
"""
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from syriskmodels.scorecard.core.base import InitBin
from syriskmodels.scorecard.core.factory import WOEBinFactory
from syriskmodels.utils import round_


@WOEBinFactory.register('quantile')
class QuantileInitBin(InitBin):
    """等频细分箱
    
    对数值型变量，通过分位数寻找切分点。
    对类别型变量，直接返回所有类别值。
    
    参数:
        initial_bins: 等频分箱的箱数，默认 20
        sig_figs: 切分点有效数字位数，默认 4
    """
    
    def __init__(self, initial_bins: int = 20, sig_figs: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.n_bins = initial_bins
        self.sig_figs = sig_figs
    
    def woebin(self, dtm: pd.DataFrame, breaks=None) -> list:
        """执行等频分箱
        
        参数:
            dtm: 输入数据 (variable, y, value 三列)
            breaks: 不使用该参数
        
        返回:
            切分点列表
        """
        if is_numeric_dtype(dtm['value']):
            # 数值型变量
            xvalue = dtm['value'].astype(float)
            breaks = np.quantile(xvalue, np.linspace(0, 1, self.n_bins + 1))
            breaks = round_(np.unique(breaks), self.sig_figs)
            breaks[0] = -np.inf
            breaks[-1] = np.inf
            breaks = np.unique(breaks)
            breaks = self.check_empty_bins(dtm, breaks)
        else:
            # 类别型变量
            breaks = np.unique(dtm['value'])
        
        return breaks.tolist() if isinstance(breaks, np.ndarray) else breaks


@WOEBinFactory.register('hist')
class HistogramInitBin(InitBin):
    """等宽细分箱
    
    对类别型变量，直接返回所有类别值。
    对数值型变量，首先排除 outlier 样本，对剩余样本的 range 等分成 n_bins 等分。
    
    参数:
        initial_bins: 等宽分箱的箱数，默认 20
    """
    
    def __init__(self, initial_bins: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.n_bins = initial_bins
    
    @staticmethod
    def _pretty(low: float, high: float, n: int) -> np.ndarray:
        """生成美观的切分点
        
        参数:
            low: 最小值
            high: 最大值
            n: 分箱数
        
        返回:
            切分点数组
        """
        def nice_number(x):
            exp = np.floor(np.log10(abs(x)))
            f = abs(x) / 10**exp
            if f < 1.5:
                nf = 1.
            elif f < 3.:
                nf = 2.
            elif f < 7.:
                nf = 5.
            else:
                nf = 10.
            return np.sign(x) * nf * 10.**exp
        
        d = abs(nice_number((high - low) / (n - 1)))
        min_x = np.floor(low / d) * d
        max_x = np.ceil(high / d) * d
        return np.arange(min_x, max_x + 0.5 * d, d)
    
    def woebin(self, dtm: pd.DataFrame, breaks=None) -> list:
        """执行等宽分箱
        
        参数:
            dtm: 输入数据
            breaks: 不使用该参数
        
        返回:
            切分点列表
        """
        if is_numeric_dtype(dtm['value']):
            # 数值型变量
            xvalue = dtm['value'].astype(float)
            
            # outlier 处理
            iq = xvalue.quantile([0.01, 0.25, 0.75, 0.99])
            iqr = iq[0.75] - iq[0.25]
            if iqr == 0:
                prob_down = 0.01
                prob_up = 0.99
            else:
                prob_down = 0.25
                prob_up = 0.75
            
            xvalue_rm_outlier = xvalue[
                (xvalue >= iq[prob_down] - 3 * iqr) &
                (xvalue <= iq[prob_up] + 3 * iqr)
            ]
            
            n_bins = self.n_bins
            len_uniq_x = len(np.unique(xvalue_rm_outlier))
            if len_uniq_x < n_bins:
                n_bins = len_uniq_x
            
            # initial breaks
            if len_uniq_x == n_bins:
                breaks = np.unique(xvalue_rm_outlier)
            else:
                breaks = self._pretty(
                    low=min(xvalue_rm_outlier),
                    high=max(xvalue_rm_outlier),
                    n=self.n_bins
                )
            
            breaks = list(filter(
                lambda x: np.nanmin(xvalue) < x <= np.nanmax(xvalue),
                breaks
            ))
            breaks = [float('-inf')] + sorted(breaks) + [float('inf')]
            breaks = self.check_empty_bins(dtm, breaks)
        else:
            # 类别型变量
            breaks = np.unique(dtm['value'])
        
        return breaks.tolist() if isinstance(breaks, np.ndarray) else breaks

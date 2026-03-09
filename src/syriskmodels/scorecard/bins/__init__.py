# -*- encoding: utf-8 -*-
"""
分箱算法模块
"""
from syriskmodels.scorecard.bins.initial import (
    QuantileInitBin,
    HistogramInitBin
)
from syriskmodels.scorecard.bins.optimal import (
    ChiMergeOptimBin,
    TreeOptimBin,
    RuleOptimBin
)

__all__ = [
    'QuantileInitBin',
    'HistogramInitBin',
    'ChiMergeOptimBin',
    'TreeOptimBin',
    'RuleOptimBin',
]

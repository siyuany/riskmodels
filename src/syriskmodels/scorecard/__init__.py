# -*- encoding: utf-8 -*-
"""
scorecard 模块

提供 WOE 分箱、评分卡生成等功能
"""

# 常量
from syriskmodels.scorecard.constants import VariableStatus

# 异常
from syriskmodels.scorecard.exceptions import (
    WOEBinError,
    ConstantVariableError,
    TooManyCategoriesError,
    InvalidBreaksError,
    DataValidationError,
    BinningAlgorithmError,
    WOEComputationError
)

# 核心类
from syriskmodels.scorecard.core import (
    WOEBin,
    InitBin,
    OptimBinMixin,
    ComposedWOEBin,
    WOEBinFactory
)

# 分箱算法
from syriskmodels.scorecard.bins import (
    QuantileInitBin,
    HistogramInitBin,
    ChiMergeOptimBin,
    TreeOptimBin,
    RuleOptimBin
)

# 辅助函数
from syriskmodels.scorecard.utils.binning_helpers import (
    extract_numeric_breaks,
    format_numeric_bin_names,
    extract_breaks_from_binning,
    compute_woe,
    compute_iv,
    merge_adjacent_bins
)

# 验证函数
from syriskmodels.scorecard.utils.validation import (
    check_uniques,
    replace_blank_string,
    check_y,
    x_variable,
    check_breaks_list,
    check_special_values
)

__all__ = [
    # 常量
    'VariableStatus',
    # 异常
    'WOEBinError',
    'ConstantVariableError',
    'TooManyCategoriesError',
    'InvalidBreaksError',
    'DataValidationError',
    'BinningAlgorithmError',
    'WOEComputationError',
    # 核心类
    'WOEBin',
    'InitBin',
    'OptimBinMixin',
    'ComposedWOEBin',
    'WOEBinFactory',
    # 分箱算法
    'QuantileInitBin',
    'HistogramInitBin',
    'ChiMergeOptimBin',
    'TreeOptimBin',
    'RuleOptimBin',
    # 辅助函数
    'extract_numeric_breaks',
    'format_numeric_bin_names',
    'extract_breaks_from_binning',
    'compute_woe',
    'compute_iv',
    'merge_adjacent_bins',
    # 验证函数
    'check_uniques',
    'replace_blank_string',
    'check_y',
    'x_variable',
    'check_breaks_list',
    'check_special_values',
]

# -*- encoding: utf-8 -*-
"""
scorecard 模块

提供 WOE 分箱、评分卡生成等功能
"""

from syriskmodels.scorecard.constants import VariableStatus
from syriskmodels.scorecard.exceptions import (
    WOEBinError,
    ConstantVariableError,
    TooManyCategoriesError,
    InvalidBreaksError,
    DataValidationError,
    BinningAlgorithmError,
    WOEComputationError
)
from syriskmodels.scorecard.utils.binning_helpers import (
    extract_numeric_breaks,
    format_numeric_bin_names,
    extract_breaks_from_binning,
    compute_woe,
    compute_iv,
    merge_adjacent_bins
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
    # 辅助函数
    'extract_numeric_breaks',
    'format_numeric_bin_names',
    'extract_breaks_from_binning',
    'compute_woe',
    'compute_iv',
    'merge_adjacent_bins',
]

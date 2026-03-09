# -*- encoding: utf-8 -*-
"""
scorecard 模块常量定义

根据评审反馈，仅保留必要的枚举类，避免过度设计
"""
from enum import IntEnum


class VariableStatus(IntEnum):
    """变量状态枚举
    
    用于表示变量在分箱前的状态检查结果
    
    使用示例:
        >>> from syriskmodels.scorecard.constants import VariableStatus
        >>> status = check_uniques(series)
        >>> if status == VariableStatus.CONSTANT:
        ...     return 'CONST'
    """
    
    OK = 0
    """变量正常，可以分箱
    
    触发条件：唯一值数 > 1 且（数值型或类别数≤max_cate_num）
    """
    
    CONSTANT = 10
    """常量变量
    
    触发条件：唯一值数 ≤ 1
    处理方式：返回 'CONST'，跳过该变量
    """
    
    TOO_MANY_CATEGORIES = 20
    """类别过多
    
    触发条件：非数值型且唯一值数 > max_cate_num
    处理方式：返回 'TOO_MANY_VALUES'，跳过该变量
    
    注意：max_cate_num 是用户可配置参数，默认值为 50
    """

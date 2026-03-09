# -*- encoding: utf-8 -*-
"""
scorecard 模块异常定义

定义分箱和评分卡相关的异常层次结构
"""


class WOEBinError(Exception):
    """分箱相关异常基类
    
    所有分箱相关的自定义异常都继承自此类
    
    使用示例:
        >>> try:
        ...     result = woebin(dt, y)
        ... except WOEBinError as e:
        ...     logging.error(f"分箱失败：{e}")
    """
    pass


class ConstantVariableError(WOEBinError):
    """常量变量异常
    
    当变量为常量（唯一值≤1）时抛出
    
    参数:
        variable: 常量变量名
    
    异常消息格式:
        "变量 '{variable}' 为常量变量，不适合分箱"
    """
    
    def __init__(self, variable: str):
        self.variable = variable
        super().__init__(f"变量 '{variable}' 为常量变量，不适合分箱")


class TooManyCategoriesError(WOEBinError):
    """类别过多异常
    
    当类别变量类别数超过限制时抛出
    
    参数:
        variable: 变量名
        n_categories: 实际类别数
        max_allowed: 最大允许类别数
    
    异常消息格式:
        "变量 '{variable}' 类别数 ({n_categories}) 超过最大允许值 ({max_allowed})"
    """
    
    def __init__(self, variable: str, n_categories: int, max_allowed: int):
        self.variable = variable
        self.n_categories = n_categories
        self.max_allowed = max_allowed
        super().__init__(
            f"变量 '{variable}' 类别数 ({n_categories}) "
            f"超过最大允许值 ({max_allowed})"
        )


class InvalidBreaksError(WOEBinError):
    """无效切分点异常
    
    当切分点无效时抛出
    
    参数:
        message: 错误详情
    
    常见错误消息:
        - "切分点必须为数值型"
        - "切分点数量与分箱数不匹配"
        - "切分点包含重复值"
        - "切分点未排序"
        - "非连续区间无法合并"
    """
    
    def __init__(self, message: str):
        super().__init__(message)


class DataValidationError(WOEBinError):
    """数据验证异常
    
    当输入数据验证失败时抛出
    
    参数:
        message: 错误详情
    
    常见错误消息:
        - "目标变量 '{y}' 不在数据集中"
        - "目标变量包含空值"
        - "解释变量 '{x}' 不在数据集中"
        - "数据集为空"
    """
    
    def __init__(self, message: str):
        super().__init__(message)


class BinningAlgorithmError(WOEBinError):
    """分箱算法异常
    
    当分箱算法执行失败时抛出
    
    参数:
        message: 错误详情
        algorithm: 算法名称（可选）
    
    常见错误消息:
        - "ChiMerge 算法执行失败：卡方检验异常"
        - "Tree 分箱失败：无法找到满足条件的切分点"
        - "等频分箱失败：数据包含大量重复值"
    """
    
    def __init__(self, message: str, algorithm: str = None):
        self.algorithm = algorithm
        super().__init__(message)


class WOEComputationError(WOEBinError):
    """WOE 计算异常
    
    当 WOE 计算失败时抛出（如好样本和坏样本均为 0）
    
    参数:
        message: 错误详情
    
    异常消息格式:
        "无法计算 WOE：好样本和坏样本均为 0"
    
    使用示例:
        >>> def compute_woe(good, bad, epsilon):
        ...     if np.all(good == 0) and np.all(bad == 0):
        ...         raise WOEComputationError("无法计算 WOE：好样本和坏样本均为 0")
    """
    
    def __init__(self, message: str):
        super().__init__(message)

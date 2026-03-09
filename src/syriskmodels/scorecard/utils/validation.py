# -*- encoding: utf-8 -*-
"""
数据验证模块

提供数据检查和验证函数
"""
from typing import List, Union, Dict, Optional
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import syriskmodels.logging as logging
from syriskmodels.scorecard.constants import VariableStatus
from syriskmodels.scorecard.exceptions import (
    DataValidationError,
    ConstantVariableError,
    TooManyCategoriesError
)


def check_uniques(
    s: pd.Series, 
    max_cate_num: int = 50
) -> VariableStatus:
    """检查变量唯一值数量，判断是否适合分箱
    
    检查规则:
    1. 唯一值数 ≤ 1 → CONSTANT (常量变量)
    2. 非数值型且唯一值数 > max_cate_num → TOO_MANY_CATEGORIES
    3. 其他情况 → OK
    
    参数:
        s: 待检查的数据序列
        max_cate_num: 类别变量最大允许类别数，默认 50
    
    返回:
        VariableStatus 枚举值
    
    示例:
        >>> s = pd.Series([1, 2, 3, 4, 5])
        >>> check_uniques(s)
        <VariableStatus.OK: 0>
        
        >>> s = pd.Series([999, 999, 999])
        >>> check_uniques(s)
        <VariableStatus.CONSTANT: 10>
        
        >>> s = pd.Series([f'cat_{i}' for i in range(100)])
        >>> check_uniques(s, max_cate_num=50)
        <VariableStatus.TOO_MANY_CATEGORIES: 20>
    """
    n_uniques = len(np.unique(s[~s.isna()]))
    
    if n_uniques <= 1:
        return VariableStatus.CONSTANT
    elif (not is_numeric_dtype(s)) and n_uniques > max_cate_num:
        return VariableStatus.TOO_MANY_CATEGORIES
    else:
        return VariableStatus.OK


def replace_blank_string(s: pd.Series) -> pd.Series:
    """将空字符串替换为 NaN
    
    参数:
        s: 输入序列
    
    返回:
        替换后的序列
    
    示例:
        >>> s = pd.Series(['a', '', 'b', ''])
        >>> replace_blank_string(s)
        0       a
        1     NaN
        2       b
        3     NaN
        dtype: object
    """
    return s.replace('', np.nan)


def check_y(
    dat: pd.DataFrame, 
    y: str, 
    *, 
    positive: Union[int, float] = 1
) -> pd.DataFrame:
    """检查并转换目标变量
    
    检查项:
    1. y 列是否存在
    2. y 列是否为数值型
    3. y 列是否包含空值
    4. y 列是否为二值变量 (0, 1)
    
    检查通过后，将等于 positive 的值转换为 1，其余转换为 0
    
    参数:
        dat: 输入 DataFrame
        y: 目标变量名
        positive: 正样本标识值，默认 1
    
    返回:
        处理后的 DataFrame（y 列已转换为 0/1）
    
    异常:
        KeyError: y 列不存在
        TypeError: y 列不是数值型
        ValueError: y 列不是二值变量或 positive 值不存在
    
    示例:
        >>> df = pd.DataFrame({'y': ['good', 'bad', 'good'], 'x': [1, 2, 3]})
        >>> result = check_y(df, 'y', positive='bad')
        >>> result['y'].tolist()
        [0, 1, 0]
    """
    dat = dat.copy()
    
    # 检查列是否存在
    if y not in dat.columns:
        raise KeyError(f"目标变量 '{y}' 不在数据集中")
    
    try:
        # 检查是否为数值型
        if not is_numeric_dtype(dat[y]):
            msg = f'目标变量 {y} 不是数值型，dtype={dat[y].dtypes}'
            logging.error(msg)
            raise TypeError(msg)
        
        # 检查空值
        if dat[y].isna().any():
            logging.warn(f'{y} 列包含空值，已删除对应记录')
            dat = dat.dropna(subset=[y])
        
        # 检查是否为二值变量
        if dat[y].nunique() != 2:
            logging.error(f'目标变量 {y} 不是二值变量')
            raise ValueError(f'目标变量 {y} 不是二值变量')
        
        # 检查 positive 值是否存在
        if np.all(dat[y] != positive):
            logging.error(f'positive 值 ({positive}) 不在目标变量中')
            raise ValueError(f'positive 值 ({positive}) 不在目标变量中')
        
        # 转换为 0/1
        dat[y] = np.where(dat[y] == positive, 1, 0)
        
    except KeyError as err:
        logging.error(f"目标变量 '{y}' 不存在")
        raise KeyError(y) from err
    
    return dat


def x_variable(
    dat: pd.DataFrame, 
    y: Union[str, List[str]], 
    x: Optional[Union[str, List[str]]] = None, 
    var_skip: Optional[Union[str, List[str]]] = None
) -> List[str]:
    """确定解释变量列表
    
    参数:
        dat: 输入 DataFrame
        y: 目标变量名或列表
        x: 解释变量名或列表，为 None 时使用除 y 外的所有列
        var_skip: 需要跳过的变量名或列表
    
    返回:
        解释变量名列表
    
    示例:
        >>> df = pd.DataFrame({'y': [0, 1], 'x1': [1, 2], 'x2': [3, 4], 'skip': [5, 6]})
        >>> x_variable(df, 'y')
        ['x1', 'x2', 'skip']
        >>> x_variable(df, 'y', var_skip='skip')
        ['x1', 'x2']
    """
    from syriskmodels.utils import str_to_list
    
    y = str_to_list(y)
    if var_skip is not None:
        y = y + str_to_list(var_skip)
    
    x_all = list(set(dat.columns) - set(y))
    
    if x is None:
        x = x_all
    else:
        x = str_to_list(x)
        
        if any([i in list(x_all) for i in x]) is False:
            x = x_all
        else:
            x_not_in_x_all = set(x).difference(x_all)
            if len(x_not_in_x_all) > 0:
                logging.warn(f"解释变量 {len(x_not_in_x_all)} 不存在，已移除：{', '.join(x_not_in_x_all)}")
                x = set(x).intersection(x_all)
    
    return list(x)


def check_breaks_list(breaks_list) -> Dict:
    """检查并转换 breaks_list 参数
    
    参数:
        breaks_list: 用户提供的切分点字典或字符串
    
    返回:
        切分点字典
    
    异常:
        Exception: breaks_list 不是字典
    
    示例:
        >>> check_breaks_list(None)
        {}
        >>> check_breaks_list({'age': [20, 30, 40]})
        {'age': [20, 30, 40]}
    """
    if breaks_list is not None:
        # 是字符串则 eval 转换
        if isinstance(breaks_list, str):
            breaks_list = eval(breaks_list)
        # 不是字典则抛出异常
        if not isinstance(breaks_list, dict):
            raise Exception("breaks_list 必须是字典类型")
    else:
        breaks_list = {}
    return breaks_list


def check_special_values(special_values, xs: List[str]) -> Dict:
    """检查并转换 special_values 参数
    
    参数:
        special_values: 用户提供的特殊值列表或字典
        xs: 变量名列表
    
    返回:
        特殊值字典
    
    异常:
        Exception: special_values 不是列表或字典
    
    示例:
        >>> check_special_values(None, ['age', 'income'])
        {}
        >>> check_special_values(['missing', -999], ['age'])
        {'age': ['missing', -999]}
        >>> check_special_values({'age': ['missing']}, ['age', 'income'])
        {'age': ['missing']}
    """
    if special_values is not None:
        if isinstance(special_values, list):
            logging.warn("special_values 应该是字典类型。如果传入列表，所有变量将使用相同的特殊值")
            sv_dict = {}
            for i in xs:
                sv_dict[i] = special_values
            special_values = sv_dict
        elif not isinstance(special_values, dict):
            raise Exception("special_values 必须是列表或字典类型")
    else:
        special_values = {}
    return special_values

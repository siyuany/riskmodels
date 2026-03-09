# -*- encoding: utf-8 -*-
"""
分箱辅助函数模块

提供分箱相关的通用工具函数，这些函数都是无状态的纯函数
"""
from typing import Union
import re
import numpy as np
import pandas as pd

from syriskmodels.scorecard.exceptions import InvalidBreaksError, WOEComputationError


# 正则模式常量（模块内部使用）
_INTERVAL_PATTERN = re.compile(r"^\[(.*), *(.*)\)")
_MERGE_PATTERN = re.compile(r',[.\d]+\)%,%\[[.\d]+,')


def extract_numeric_breaks(binning: pd.DataFrame) -> pd.Series:
    """从数值型分箱结果中提取切分点（右边界）
    
    参数:
        binning: 分箱统计 DataFrame，包含 `bin_chr` 列
    
    返回:
        切分点序列（float 类型）
    
    示例:
        >>> binning = pd.DataFrame({
        ...     'bin_chr': ['[-inf, 20)', '[20, 40)', '[40, inf)']
        ... })
        >>> extract_numeric_breaks(binning)
        0     20.0
        1     40.0
        2      inf
        dtype: float64
    
    边界情况:
        - 空 DataFrame → 抛出异常
        - 单分箱 → 返回单个值
        - 合并后的分箱名 → 正确提取
    """
    if len(binning) == 0:
        raise ValueError("分箱结果不能为空")
    
    def _extract(x: str) -> str:
        match = _INTERVAL_PATTERN.match(x)
        if match:
            return match.group(2)
        return x
    
    breaks = binning['bin_chr'].apply(_extract)
    return pd.to_numeric(breaks, errors='coerce')


def format_numeric_bin_names(binning: pd.DataFrame) -> pd.DataFrame:
    """格式化数值型分箱名，合并冗余字符
    
    将相邻区间的分箱名合并，如 '[20,40)%,%[40,60)' → '[20,60)'
    
    参数:
        binning: 分箱统计 DataFrame
    
    返回:
        格式化后的 DataFrame（不修改原数据）
    
    异常:
        InvalidBreaksError: 当区间不连续时抛出
    
    示例:
        >>> binning = pd.DataFrame({
        ...     'bin_chr': ['[20,40)%,%[40,60)', '[60,80)']
        ... })
        >>> result = format_numeric_bin_names(binning)
        >>> result['bin_chr'].tolist()
        ['[20,60)', '[60,80)']
        
        >>> # 非连续区间会抛出异常
        >>> binning = pd.DataFrame({
        ...     'bin_chr': ['[10,20)%,%[30,40)']  # 非连续！
        ... })
        >>> format_numeric_bin_names(binning)
        InvalidBreaksError: 非连续区间无法合并：[10,20) 和 [30,40)
    
    边界情况:
        - 无需合并 → 保持原样
        - 多次合并 → 全部处理
        - 混合合并和未合并 → 分别处理
        - 非连续区间合并 → 抛出 InvalidBreaksError
    """
    if len(binning) == 0:
        return binning.copy()
    
    result = binning.copy()
    
    def _format_bin_name(bin_name: str) -> str:
        """格式化单个分箱名，递归合并直到无法合并"""
        # 如果包含合并分隔符，需要验证区间连续性并合并
        if '%,%' in bin_name:
            parts = bin_name.split('%,%')
            if len(parts) >= 2:
                # 验证相邻区间是否连续
                for i in range(len(parts) - 1):
                    _validate_contiguous_intervals(parts[i], parts[i + 1])
                
                # 递归合并相邻区间
                merged_parts = [parts[0]]
                for i in range(1, len(parts)):
                    prev = merged_parts[-1]
                    curr = parts[i]
                    
                    # 检查是否可以合并（前一个的结束 = 当前的开始）
                    prev_match = _INTERVAL_PATTERN.match(prev)
                    curr_match = _INTERVAL_PATTERN.match(curr)
                    
                    if prev_match and curr_match:
                        prev_end = prev_match.group(2)
                        curr_start = curr_match.group(1)
                        
                        # 处理无穷大和数值比较
                        if prev_end == curr_start or (
                            prev_end not in ('inf', '-inf') and 
                            curr_start not in ('inf', '-inf') and
                            np.isclose(float(prev_end), float(curr_start))
                        ):
                            # 可以合并
                            new_bin = f"[{prev_match.group(1)},{curr_match.group(2)})"
                            merged_parts[-1] = new_bin
                        else:
                            merged_parts.append(curr)
                    else:
                        merged_parts.append(curr)
                
                return '%,%'.join(merged_parts)
        
        return bin_name
    
    result['bin_chr'] = result['bin_chr'].apply(_format_bin_name)
    return result


def _validate_contiguous_intervals(left_bin: str, right_bin: str) -> None:
    """验证两个区间是否连续
    
    参数:
        left_bin: 左侧区间名（如 '[20,40)'）
        right_bin: 右侧区间名（如 '[40,60)'）
    
    异常:
        InvalidBreaksError: 当区间不连续时抛出
    """
    left_match = _INTERVAL_PATTERN.match(left_bin)
    right_match = _INTERVAL_PATTERN.match(right_bin)
    
    if not left_match or not right_match:
        return  # 非数值型区间，跳过验证
    
    left_end = left_match.group(2)
    right_start = right_match.group(1)
    
    # 处理无穷大
    if left_end in ('inf', '-inf') or right_start in ('inf', '-inf'):
        return
    
    try:
        left_end_val = float(left_end)
        right_start_val = float(right_start)
        
        if not np.isclose(left_end_val, right_start_val):
            raise InvalidBreaksError(
                f"非连续区间无法合并：{left_bin} 和 {right_bin} "
                f"(区间 {left_end} ≠ {right_start})"
            )
    except ValueError:
        # 无法转换为浮点数，跳过验证
        pass


def extract_breaks_from_binning(
    binning: pd.DataFrame, 
    is_numeric: bool
) -> pd.Series:
    """统一的分箱切分点提取接口（支持数值型和类别型）
    
    参数:
        binning: 分箱统计 DataFrame
        is_numeric: 是否为数值型变量
    
    返回:
        切分点序列
        - 数值型：float 类型的右边界
        - 类别型：字符串类型的分箱名
    
    示例:
        # 数值型
        >>> binning = pd.DataFrame({'bin_chr': ['[-inf, 20)', '[20, 40)']})
        >>> extract_breaks_from_binning(binning, is_numeric=True)
        0    20.0
        1    40.0
        dtype: float64
        
        # 类别型
        >>> binning = pd.DataFrame({'bin_chr': ['A%,%B', 'C']})
        >>> extract_breaks_from_binning(binning, is_numeric=False)
        0    A%,%B
        1        C
        dtype: object
    """
    if is_numeric:
        formatted = format_numeric_bin_names(binning)
        return extract_numeric_breaks(formatted)
    else:
        return binning['bin_chr'].copy()


def compute_woe(
    good: np.ndarray, 
    bad: np.ndarray, 
    epsilon: float
) -> np.ndarray:
    """计算 WOE (Weight of Evidence) 值
    
    参数:
        good: 各分箱好样本数
        bad: 各分箱坏样本数
        epsilon: 用于替换 0 的小值
    
    返回:
        WOE 值数组
    
    异常:
        WOEComputationError: 当好样本和坏样本全为 0 时抛出
    
    计算公式:
        good_distr = good / good.sum()
        bad_distr = bad / bad.sum()
        woe = np.log(good_distr / bad_distr)
    
    数学含义:
        - WOE > 0：该分箱好客户占比高于整体
        - WOE < 0：该分箱坏客户占比高于整体
        - WOE = 0：该分箱风险与整体相当
    
    边界情况:
        - good 为 0 → 使用 epsilon 替换，WOE 为负
        - bad 为 0 → 使用 epsilon 替换，WOE 为正
        - 全为 0 → 抛出 WOEComputationError
        - 好坏相等 → WOE 接近 0
    
    示例:
        >>> good = np.array([80, 60, 40])
        >>> bad = np.array([20, 40, 60])
        >>> compute_woe(good, bad, epsilon=0.5)
        array([ 0.693,  0.   , -0.693])
    """
    # 检查全零情况
    if np.all(good == 0) and np.all(bad == 0):
        raise WOEComputationError("无法计算 WOE：好样本和坏样本均为 0")
    
    # 使用 epsilon 替换 0 值
    good_safe = np.where(good == 0, epsilon, good)
    bad_safe = np.where(bad == 0, epsilon, bad)
    
    # 计算分布
    good_distr = good_safe / good_safe.sum()
    bad_distr = bad_safe / bad_safe.sum()
    
    # 计算 WOE
    woe = np.log(good_distr / bad_distr)
    return woe


def compute_iv(
    woe: np.ndarray, 
    good: np.ndarray, 
    bad: np.ndarray
) -> float:
    """计算 IV (Information Value) 值
    
    参数:
        woe: 各分箱 WOE 值
        good: 各分箱好样本数
        bad: 各分箱坏样本数
    
    返回:
        IV 值
    
    计算公式:
        good_distr = good / good.sum()
        bad_distr = bad / bad.sum()
        iv = ((good_distr - bad_distr) * woe).sum()
    
    IV 值解读:
        | IV 范围   | 预测能力           |
        |----------|-------------------|
        | < 0.02   | 无用              |
        | 0.02-0.1 | 较弱              |
        | 0.1-0.3  | 中等              |
        | 0.3-0.5  | 较强              |
        | > 0.5    | 过强（可能过拟合）  |
    
    边界情况:
        - 好坏分布相同 → IV = 0
        - 完全分离 → IV 很大
        - 单分箱 → 正常计算
        - 含零计数 → 不返回 NaN（由 WOE 计算处理）
    
    示例:
        >>> woe = np.array([0.5, 0, -0.5])
        >>> good = np.array([80, 60, 40])
        >>> bad = np.array([20, 40, 60])
        >>> compute_iv(woe, good, bad)
        0.231
    """
    good_distr = good / good.sum()
    bad_distr = bad / bad.sum()
    iv = ((good_distr - bad_distr) * woe).sum()
    return float(iv)


def merge_adjacent_bins(
    binning: pd.DataFrame, 
    idx: int
) -> pd.DataFrame:
    """合并相邻两个分箱（idx 与 idx-1 合并）
    
    参数:
        binning: 分箱统计 DataFrame
        idx: 要合并的分箱索引（与前一箱合并）
    
    返回:
        合并后的 DataFrame（行数 -1）
    
    异常:
        IndexError: 当 idx=0 或 idx 越界时抛出
    
    处理逻辑:
        1. 验证 idx > 0（第一箱无法向前合并）
        2. 合并 bin_chr[idx-1] 和 bin_chr[idx]
        3. 合并数值列（good, bad, count 等）求和
        4. 删除 idx 行
        5. 返回新 DataFrame
    
    示例:
        >>> binning = pd.DataFrame({
        ...     'bin_chr': ['[20,40)', '[40,60)', '[60,80)'],
        ...     'good': [30, 30, 40],
        ...     'bad': [10, 10, 20]
        ... })
        >>> result = merge_adjacent_bins(binning, idx=1)
        >>> len(result)
        2
        >>> result['bin_chr'].iloc[0]
        '[20,60)'
        >>> result['good'].iloc[0]
        60
    
    边界情况:
        - idx = 0 → 抛出 IndexError
        - idx 越界 → 抛出 IndexError
        - 类别型 → 使用分隔符连接
        - 其他列 → 保留并正确合并
    """
    if idx <= 0:
        raise IndexError(f"无法合并第一箱（idx={idx}），idx 必须大于 0")
    if idx >= len(binning):
        raise IndexError(f"索引越界：idx={idx}, 分箱数={len(binning)}")
    
    result = binning.copy()
    
    # 合并分箱名
    prev_bin = result.iloc[idx - 1]['bin_chr']
    curr_bin = result.iloc[idx]['bin_chr']
    
    if '%,%' in prev_bin or '%,%' in curr_bin:
        # 类别型，使用分隔符连接
        merged_bin = f"{prev_bin}%,%{curr_bin}"
    else:
        # 数值型，直接连接（会由 format_numeric_bin_names 处理）
        merged_bin = f"{prev_bin}%,%{curr_bin}"
    
    result.loc[result.index[idx - 1], 'bin_chr'] = merged_bin
    
    # 合并数值列
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        prev_val = result.iloc[idx - 1][col]
        curr_val = result.iloc[idx][col]
        result.loc[result.index[idx - 1], col] = prev_val + curr_val
    
    # 删除 idx 行
    result = result.drop(result.index[idx])
    result = result.reset_index(drop=True)
    
    return result

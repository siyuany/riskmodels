# -*- encoding: utf-8 -*-
"""
数据集加载模块

提供类似 sklearn.datasets 的便捷数据加载接口，用于快速访问内置演示数据集。

数据文件路径优先级：
1. 环境变量 SYRISKMODELS_DATA_DIR
2. 当前工作目录下的 data/ 目录
3. 仓库根目录下的 data/ 目录（开发模式）
"""
import os
from pathlib import Path

import pandas as pd


def get_data_dir() -> Path:
    """返回内置数据集目录路径

    数据目录查找优先级：
    1. 环境变量 SYRISKMODELS_DATA_DIR
    2. 当前工作目录下的 data/ 目录
    3. 仓库根目录下的 data/ 目录（开发模式）

    Returns:
        Path: 数据目录的绝对路径。

    Example:
        >>> from syriskmodels.datasets import get_data_dir
        >>> data_dir = get_data_dir()
        >>> list(data_dir.glob('*.csv.gz'))
        [PosixPath('.../data/creditcard.csv.gz'), ...]
    """
    env_dir = os.environ.get('SYRISKMODELS_DATA_DIR')
    if env_dir:
        return Path(env_dir).resolve()
    
    cwd_data = Path.cwd() / 'data'
    if cwd_data.exists():
        return cwd_data
    
    dev_data = Path(__file__).resolve().parent.parent.parent / 'data'
    if dev_data.exists():
        return dev_data
    
    return cwd_data


def load_germancredit() -> pd.DataFrame:
    """加载德国信用数据集 (germancredit)

    该数据集包含 1000 条个人信用记录，包含 20 个特征变量和 1 个目标变量
    (creditability)，其中 1 表示坏样本（原始值 ``'bad'``），
    0 表示好样本（原始值 ``'good'``）。加载时自动将字符串映射为 0/1。

    Returns:
        pd.DataFrame: 德国信用数据集，包含 1000 行和 21 列。

    Raises:
        FileNotFoundError: 当数据文件 ``data/germancredit.csv.gz`` 不存在时。

    Example:
        >>> from syriskmodels.datasets import load_germancredit
        >>> df = load_germancredit()
        >>> df.shape
        (1000, 21)
        >>> df['creditability'].value_counts()
        0    700
        1    300
        Name: creditability, dtype: int64
    """
    path = get_data_dir() / 'germancredit.csv.gz'
    if not path.exists():
        raise FileNotFoundError(
            f'数据文件不存在: {path}。请确认 data/ 目录下包含 germancredit.csv.gz')
    df = pd.read_csv(path)
    # 原始 creditability 列为字符串 ('good'/'bad')，映射为 0/1
    if pd.api.types.is_string_dtype(df['creditability']):
        df['creditability'] = df['creditability'].map({'good': 0, 'bad': 1})
    return df


def load_creditcard() -> pd.DataFrame:
    """加载信用卡欺诈数据集 (creditcard)

    该数据集包含信用卡交易记录，包含 28 个 PCA 特征变量 (V1-V28)、
    交易时间 (Time)、交易金额 (Amount) 和目标变量 (Class)，
    其中 1 表示欺诈交易，0 表示正常交易。

    Returns:
        pd.DataFrame: 信用卡欺诈数据集。

    Raises:
        FileNotFoundError: 当数据文件 ``data/creditcard.csv.gz`` 不存在时。

    Example:
        >>> from syriskmodels.datasets import load_creditcard
        >>> df = load_creditcard()
        >>> df.columns[:5].tolist()
        ['Time', 'V1', 'V2', 'V3', 'V4']
    """
    path = get_data_dir() / 'creditcard.csv.gz'
    if not path.exists():
        raise FileNotFoundError(
            f'数据文件不存在: {path}。请确认 data/ 目录下包含 creditcard.csv.gz')
    return pd.read_csv(path)
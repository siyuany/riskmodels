# -*- encoding: utf-8 -*-
from functools import wraps

import numpy as np
import pandas as pd


def sample_stats(df, target):
    """
    统计样本中好坏样本数量及坏样本占比，使用示例如下：

    >>> sample_df = pd.DataFrame({
    ...     'id': np.arange(10),
    ...     'prod': ['a', 'a', 'a', 'b', 'b', 'b', 'b', 'c', 'c', 'c'],
    ...     'y': [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1]})
    >>> sample_df
       id prod  y
    0   0    a  0
    1   1    a  0
    2   2    a  1
    3   3    b  1
    4   4    b  0
    5   5    b  1
    6   6    b  0
    7   7    c  1
    8   8    c  0
    9   9    c  0
    >>> sample_stats(sample_df, target='y')
    TotalCnt     10
    GoodCnt       6
    BadCnt        4
    BadRate     0.4
    dtype: object
    >>> sample_df.groupby('prod').apply(sample_stats, target='y')
          TotalCnt  GoodCnt  BadCnt   BadRate
    prod
    a            3        2       1  0.333333
    b            4        2       2  0.500000
    c            3        2       1  0.333333

    Args:
        df: pd.DataFrame, 样本集
        target: 目标变量名

    Returns:
        pd.Series，各个统计量组成的数列

    """
    y = df[target]
    ttl_cnt = len(y)
    good_cnt = np.sum(y == 0)
    bad_cnt = np.sum(y == 1)
    bad_rate = bad_cnt / (good_cnt + bad_cnt)
    return pd.Series([ttl_cnt, good_cnt, bad_cnt, bad_rate],
                     index=['TotalCnt', 'GoodCnt', 'BadCnt', 'BadRate'],
                     dtype='object')


def monotonic(series: pd.Series) -> str:
    """
    判断一组序列(pd.Series)的单调类型，包括以下五种类型：

    * 'increasing' - 单调递增
    * 'decreasing' - 单调递减
    * 'up_u_shape' - 先单调递减，后单调递增
    * 'down_u_shape' - 先单调递增，后单调递增

    Args:
        series: pd.Series, 数值型数组序列

    Returns:
        str, 单调类型

    """
    if series.is_monotonic_increasing:
        return 'increasing'
    elif series.is_monotonic_decreasing:
        return 'decreasing'
    else:
        for i in range(2, len(series)):
            if series[:i].is_monotonic_increasing and \
                    series[i:].is_monotonic_decreasing:
                return 'down_u_shape'
            elif series[:i].is_monotonic_decreasing and \
                    series[i:].is_monotonic_increasing:
                return 'up_u_shape'
    return 'non_monotonic'


def round_(a, significant_figures=1):
    """
    按照有效数字个数进行四舍五入

    Args:
        a:
        significant_figures:

    Returns:

    """
    e = np.ceil(np.log10(np.where(a == 0, 1, np.abs(a))))
    sig_figs = a / 10**e
    sig_figs = np.round(sig_figs, significant_figures)
    return sig_figs * 10**e


def interactive_mode():
    # noinspection PyPackageRequirements
    import __main__ as main
    if hasattr(main, '__file__'):
        return False
    else:
        return True


def resolve_no_cores(no_cores):
    """解析并行度 —— 默认串行（B-11）。

    ``None`` / ``<1`` 一律视为 1（串行）；只有用户**显式**传入
    ``no_cores>1`` 才启用并行。历史实现在 ``no_cores=None`` 时按
    ``ceil(len(xs)/5)`` 自动开启 ``mp.Pool``：小数据下更慢（W1 实测 +45%），
    且 spawn 语义（macOS/Windows）下 Jupyter/REPL/stdin 环境中 worker
    无法导入 ``__main__`` 会**无限挂起**，故 W2 移除自动并行。
    """
    if no_cores is None or no_cores < 1:
        return 1
    return int(no_cores)


def parallel_starmap(func, tasks, no_cores, timeout=None):
    """安全的多进程 starmap（B-11）。

    仅在显式 ``no_cores>1`` 时由 ``woebin`` / ``woebin_ply`` 调用：

    * 使用平台默认进程上下文（Linux=fork，macOS/Windows=spawn）；
      spawn + 交互式环境（Jupyter/REPL/管道 stdin，``__main__`` 不可导入）
      下 worker 会永久挂起 —— 此时自动回退串行并发出 ``UserWarning``；
    * ``starmap_async().get(timeout)``：worker 内异常会在主进程重新抛出
      （错误传播）；超时则 ``terminate()`` 进程池并抛 ``TimeoutError``；
    * ``with`` 上下文保证进程池无论成败都被清理（不再有泄漏的 Pool）。

    Args:
        func: 可调用对象（须可 pickle：模块级函数/实例）
        tasks: 参数元组可迭代对象
        no_cores: 进程数（>1）
        timeout: 整体超时秒数；None 表示不限时

    Returns:
        list，结果顺序与 tasks 一致
    """
    import itertools
    import multiprocessing as mp
    import warnings

    tasks = list(tasks)
    ctx = mp.get_context()
    if ctx.get_start_method() == 'spawn' and interactive_mode():
        warnings.warn(
            '交互式环境（Jupyter/REPL/stdin）下 spawn 模式的 worker 无法导入 '
            '__main__，将永久挂起；已回退串行执行（no_cores=1）。'
            '如需并行，请以脚本方式运行并显式传入 no_cores>1。',
            UserWarning,
            stacklevel=3)
        return list(itertools.starmap(func, tasks))

    with ctx.Pool(processes=no_cores) as pool:
        async_result = pool.starmap_async(func, tasks)
        try:
            return async_result.get(timeout=timeout)
        except mp.TimeoutError as err:
            pool.terminate()
            raise TimeoutError(
                f'多进程任务未在 {timeout}s 内完成，已终止进程池；'
                '可减少变量数、增大 timeout 或改用 no_cores=1 串行执行'
            ) from err


def str_to_list(x):
    if x is not None and isinstance(x, str):
        x = [x]
    return x


def exception_trace(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            raise RuntimeError(f'Error in running {func.__name__} with '
                               f'args={args}'
                               f'kwargs={kwargs}')

    return wrapper

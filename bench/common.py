# -*- encoding: utf-8 -*-
"""bench 公共工具：环境信息、计时器、JSON 输出。

设计要点
--------
* **只依赖公共 API**：bench 脚本不 import 任何私有实现，不修改 src/。
* **一律 ``no_cores=1``**：避免 multiprocessing spawn 开销与进程不确定性
  （见 W1 报告 B-11），使基线可复现。
* **1 次 warmup + N 次计时取中位数**：抵消 import / 缓存 / JIT 抖动。
* 输出机器与版本信息，便于跨机器对比时判断基线是否可比。
"""
import json
import multiprocessing as mp
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / 'results'


# --------------------------------------------------------------------------- #
# 环境信息
# --------------------------------------------------------------------------- #

def environment_info() -> Dict[str, Any]:
    """采集机器 / 解释器 / 关键依赖版本（用于判断基线可比性）。"""
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn
    import statsmodels

    return {
        'python': sys.version.split()[0],
        'python_implementation': platform.python_implementation(),
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor() or 'unknown',
        'cpu_count': mp.cpu_count(),
        'mp_start_method': mp.get_start_method(),
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'scikit_learn': sklearn.__version__,
        'statsmodels': statsmodels.__version__,
        'pyarrow': _safe_version('pyarrow'),
        'pythonhashseed': os.environ.get('PYTHONHASHSEED', '<unset>'),
    }


def _safe_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except ImportError:
        return '<not installed>'
    return getattr(module, '__version__', '<unknown>')


# --------------------------------------------------------------------------- #
# 计时
# --------------------------------------------------------------------------- #

def time_case(
    fn: Callable[[], Any],
    *,
    repeats: int = 3,
    warmup: int = 1,
    on_result: Optional[Callable[[Any], None]] = None,
) -> Dict[str, Any]:
    """执行 ``warmup`` 次预热 + ``repeats`` 次计时，返回统计量。

    参数:
        fn: 无参可调用对象（单次被测操作）
        repeats: 计时次数
        warmup: 预热次数（不计入统计）
        on_result: 每次执行后接收返回值的回调（用于提取结果规模等元数据）

    返回:
        ``{'seconds': [...], 'median_seconds': float, 'min_seconds': float,
        'max_seconds': float, 'stdev_seconds': float, 'repeats': int,
        'warmup': int}``
    """
    if repeats < 1:
        raise ValueError('repeats 必须 >= 1')
    if warmup < 0:
        raise ValueError('warmup 必须 >= 0')

    for _ in range(warmup):
        result = fn()
        if on_result is not None:
            on_result(result)

    seconds: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        seconds.append(time.perf_counter() - start)
        if on_result is not None:
            on_result(result)

    return {
        'seconds': [round(s, 6) for s in seconds],
        'median_seconds': round(statistics.median(seconds), 6),
        'min_seconds': round(min(seconds), 6),
        'max_seconds': round(max(seconds), 6),
        'stdev_seconds': round(statistics.stdev(seconds), 6) if len(seconds) > 1 else 0.0,
        'repeats': repeats,
        'warmup': warmup,
    }


def per_variable_seconds(median_seconds: float, n_variables: int) -> float:
    """每变量平均耗时（用于横向对比不同变量规模的用例）。"""
    if n_variables <= 0:
        return float('nan')
    return round(median_seconds / n_variables, 6)


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #

def load_dataset(name: str, nrows: Optional[int] = None):
    """加载内置数据集。

    参数:
        name: ``'germancredit'`` 或 ``'creditcard'``
        nrows: 只取前 n 行（None 表示全量）

    返回:
        ``(df, meta)``；meta 含 ``n_rows`` / ``n_cols`` / ``target`` / ``source``
    """
    from syriskmodels.datasets import load_creditcard, load_germancredit

    if name == 'germancredit':
        df = load_germancredit()
        target = 'creditability'
    elif name == 'creditcard':
        df = load_creditcard()
        target = 'Class'
    else:
        raise ValueError(f"未知数据集 {name}（可选 germancredit / creditcard）")

    total_rows = int(df.shape[0])
    if nrows is not None and nrows < total_rows:
        # 确定性抽样：取前 nrows 行，保持原有顺序（不做随机抽样）
        df = df.head(nrows)
    df = df.reset_index(drop=True)

    return df, {
        'name': name,
        'n_rows': int(df.shape[0]),
        'n_cols': int(df.shape[1]),
        'total_rows_available': total_rows,
        'target': target,
        'sampling': 'head' if df.shape[0] < total_rows else 'full',
    }


def default_numeric_variables(df, target: str, limit: Optional[int] = None) -> List[str]:
    """按列顺序取数值型变量（确定性顺序，不依赖 set 遍历）。"""
    import pandas as pd

    variables = [
        col for col in df.columns
        if col != target and pd.api.types.is_numeric_dtype(df[col])
    ]
    return variables[:limit] if limit is not None else variables


def default_variables(df, target: str) -> List[str]:
    """按列顺序取全部解释变量（不含 target）。"""
    return [col for col in df.columns if col != target]


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #

def build_payload(
    *,
    suite: str,
    cases: List[Dict[str, Any]],
    command: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """组装最终写入 JSON 的 payload。"""
    payload: Dict[str, Any] = {
        'suite': suite,
        'generated_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'command': command or ' '.join([Path(sys.argv[0]).name] + sys.argv[1:]),
        'environment': environment_info(),
        'no_cores': 1,
        'cases': cases,
    }
    if extra:
        payload.update(extra)
    return payload


def write_payload(payload: Dict[str, Any], output: Optional[str]) -> Path:
    """写出 JSON；``output`` 为空时写入 ``bench/results/<suite>_<date>.json``。"""
    if output:
        path = Path(output)
        if not path.is_absolute():
            path = REPO_ROOT / path
    else:
        date = datetime.now().strftime('%Y%m%d')
        path = DEFAULT_RESULTS_DIR / f'{payload["suite"]}_{date}.json'

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=False)
        fh.write('\n')
    return path


def print_case_summary(case: Dict[str, Any]) -> None:
    """控制台打印单用例摘要。"""
    timing = case['timing']
    print(
        f"  {case['case_id']:<52s} "
        f"median={timing['median_seconds']:8.3f}s "
        f"min={timing['min_seconds']:8.3f}s "
        f"per_var={case.get('per_variable_seconds', float('nan')):7.4f}s "
        f"vars={case.get('n_variables', '-')}"
    )

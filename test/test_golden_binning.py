# -*- encoding: utf-8 -*-
"""Golden 快照回归测试（分箱结果基线）。

目的
----
为后续性能重构 / 算法改造提供**行为护栏**：在任何重构前后，只要分箱结果
发生变化，这里就会失败并给出最小 diff。

数据源
------
1. 合成数据 ``_synthetic_frame()``：固定种子 1000 行，覆盖数值、类别、缺失、
   特殊值（-999 / -1）、常量列、超多类别列等边界。
2. ``load_germancredit()``：真实数据集全量（1000 行 × 20 变量），
   **不依赖 creditcard 全量**以避免 CI 过慢。

覆盖用例与快照列见 ``test/golden/README.md``。

对比方式
--------
* 快照存 ``test/golden/*.json``，浮点统一 ``round(value, 8)`` 后**严格相等**比较。
* 变量、分箱、列顺序全部按确定性顺序输出（变量按名排序），不依赖哈希顺序。
* 需重新生成时：``UPDATE_GOLDEN=1 PYTHONHASHSEED=0 python -m pytest test/test_golden_binning.py``。

已知在 pandas 3 下无法通过的路径不在本文件内修复，统一记录在
``test/test_known_bugs.py`` 与 ``docs/plans/w1-baseline-report.md``。
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from syriskmodels.scorecard import woebin

pytestmark = pytest.mark.golden

GOLDEN_DIR = Path(__file__).resolve().parent / 'golden'
#: 快照格式版本；结构变更时递增，测试会提示需要 UPDATE_GOLDEN=1
GOLDEN_VERSION = 1
#: 浮点比较精度
FLOAT_DIGITS = 8
#: 参与快照的列（按确定性顺序）
SNAPSHOT_COLUMNS = [
    'variable',
    'bin',
    'count',
    'good',
    'bad',
    'badprob',
    'woe',
    'bin_iv',
    'total_iv',
    'breaks',
    'is_special_values',
]

UPDATE_GOLDEN = os.environ.get('UPDATE_GOLDEN') == '1'


# --------------------------------------------------------------------------- #
# 数据源
# --------------------------------------------------------------------------- #

def _synthetic_frame() -> pd.DataFrame:
    """合成数据：数值 + 类别 + 缺失 + 特殊值 + 常量 + 超多类别。

    可移植性要求（重要）
    --------------------
    本函数**只用整数算术与精确可表示的浮点运算**，不调用 ``np.random.*`` 的
    RNG 内核，也不使用 ``exp``/``log``/``normal`` 等超越函数。原因：

    相同 seed 下 ``Generator.standard_normal`` / ``binomial`` / ``lognormal``
    在不同 CPU 架构（macOS arm64 vs Linux x86_64）会走不同 SIMD 内核并产生
    **不同位序列**。实测在 CI（ubuntu x86_64）上，RNG 生成的 ``num_signal``
    与本地 arm64 逐位不同，导致 tree/chi2 分箱结果分歧，
    ``synthetic_quantile_chi2_ib20_limit5`` 在 3 个 CI job 上全数失败。

    因此这里的数据由行号推导（``%``/``//`` 与 1/2^k 缩放，均为 IEEE-754 精确
    运算），在任意平台上逐位一致；快照里的 ``data_fingerprint`` 会前置校验
    这一点（见 :func:`_data_fingerprint`）。

    索引为 ``range(1000)``，列顺序固定。
    """
    n = 1000
    idx = np.arange(n, dtype=np.int64)

    # 三个数值变量：全部由行号整数推导，1/32 与 1/64 均为二进制精确小数
    base = (idx * 7 % 101) * 3
    df = pd.DataFrame({
        'num_normal': (base - 150) / 32.0,                     # 范围约 [-4.6, 4.6]
        'num_skewed': 625.0 + (idx * 13 % 619),                # 整数值，范围 [625, 1243]
        'num_discrete': (idx * 5 % 6).astype(np.int64),        # 离散 0..5
    })

    # 类别变量：整数 → 类别标签，8 个类别（其中 cat5 用于特殊值演示）
    category = np.array([f'cat{c}' for c in (idx * 3 % 8).tolist()], dtype=object)
    df['cat_ok'] = category

    # 缺失：数值列与类别列各有空值（隐式 'missing' 特殊值路径）
    df.loc[:59, 'num_normal'] = np.nan
    df.loc[60:99, 'cat_ok'] = np.nan

    # 特殊值：-999（数值哨兵）、-1（整型哨兵）、'unknown'（类别哨兵）
    df.loc[100:159, 'num_skewed'] = -999.0
    df.loc[160:199, 'num_discrete'] = -1
    df.loc[200:239, 'cat_ok'] = 'unknown'

    # 常量列（应被识别为 CONST 并跳过）
    df['const_int'] = 7

    # 超多类别列（默认 max_cate_num=50 → TOO_MANY_VALUES）
    df['many_cats'] = np.array([f'c{i:04d}' for i in range(n)], dtype=object)

    # 信号变量：与 target 相关，供 tree/chi2 有切分空间。
    # 用独立于 num_normal 的推导方式，避免两列完全共线。
    signal = (idx * 11 % 97) * 2 + (idx % 7)
    df['num_signal'] = signal / 32.0

    # target：确定性生成（不使用 RNG），并留出不参与任何特征的抖动分量，
    # 保证好坏样本不会在某个分箱里被完美分开（否则 WOE 全部落在 epsilon 上）。
    jitter = (idx * 29 % 1024) / 1024.0
    threshold = ((signal - 90.0) / 32.0) + jitter - 0.5
    df['target'] = (threshold > 0).astype(np.int64)

    return df[['num_normal', 'num_skewed', 'num_discrete', 'cat_ok',
               'const_int', 'many_cats', 'num_signal', 'target']]


#: 合成数据列顺序（指纹与快照共用，避免两处漂移）
FINGERPRINT_COLUMNS = [
    'num_normal', 'num_skewed', 'num_discrete', 'cat_ok',
    'const_int', 'many_cats', 'num_signal', 'target',
]


def _column_checksum(series: pd.Series) -> str:
    """单列内容指纹：CRC32 over 每行 ``repr`` 的规范化拼接。

    用 CRC32 而非 sha256/md5：这里是**确定性自检**（catch 数据漂移），
    不是安全用途；短摘要便于人工核对快照内容。

    逐行 ``repr``：``repr(float)`` 输出最短往返表示，跨平台一致；
    ``NaN`` 统一写成 ``nan``，不依赖 pandas/numpy 的内部表示。
    """
    import zlib

    parts = []
    for value in series.tolist():
        if value is None:
            parts.append('none')
        elif isinstance(value, float):
            parts.append('nan' if value != value else repr(value))
        else:
            parts.append(repr(value))
    payload = '\x1f'.join(parts).encode('utf-8')
    return format(zlib.crc32(payload) & 0xFFFFFFFF, '08x')


def _data_fingerprint(frame: pd.DataFrame) -> Dict[str, Any]:
    """整表指纹：列顺序 + 行数 + 每列 CRC32。

    作为快照的**前置校验**：若生成器在别的平台上产出了不同数据，
    校验会立刻失败并直接指出是哪一列，而不是抛出一大坨分箱 diff。
    """
    return {
        'n_rows': int(frame.shape[0]),
        'columns': list(frame.columns),
        'column_checksums': {
            column: _column_checksum(frame[column]) for column in frame.columns
        },
    }


def _germancredit_frame() -> pd.DataFrame:
    """germancredit 全量；数据缺失时 skip。"""
    from test.conftest import GERMANCREDIT_FILE, require_data
    require_data(GERMANCREDIT_FILE)
    from syriskmodels.datasets import load_germancredit
    return load_germancredit()


# --------------------------------------------------------------------------- #
# 用例定义
# --------------------------------------------------------------------------- #

_CASES: List[Dict[str, Any]] = [
    {
        'name': 'synthetic_quantile_ib20',
        'dataset': 'synthetic',
        'x': ['num_normal', 'num_skewed', 'num_discrete', 'cat_ok', 'num_signal'],
        'target': 'target',
        'methods': ['quantile'],
        'kwargs': {'initial_bins': 20},
    },
    {
        'name': 'synthetic_quantile_tree_ib20_limit5',
        'dataset': 'synthetic',
        'x': ['num_normal', 'num_skewed', 'num_discrete', 'cat_ok', 'num_signal'],
        'target': 'target',
        'methods': ['quantile', 'tree'],
        'kwargs': {'initial_bins': 20, 'bin_num_limit': 5},
    },
    {
        'name': 'synthetic_quantile_chi2_ib20_limit5',
        'dataset': 'synthetic',
        'x': ['num_normal', 'num_skewed', 'num_discrete', 'cat_ok', 'num_signal'],
        'target': 'target',
        'methods': ['quantile', 'chi2'],
        'kwargs': {'initial_bins': 20, 'bin_num_limit': 5},
    },
    {
        # 类别型特殊值 + 缺失值（当前 pandas 3 下可正常完成）
        'name': 'synthetic_categorical_special_values',
        'dataset': 'synthetic',
        'x': ['num_normal', 'cat_ok', 'num_signal'],
        'target': 'target',
        'methods': ['quantile', 'tree'],
        'kwargs': {'initial_bins': 20, 'bin_num_limit': 5},
        'special_values': {
            'cat_ok': ['unknown', 'missing'],
            'num_normal': ['missing'],
        },
    },
    {
        # 无显式 special_values，但数据本身含缺失（走 add_missing_spl_val 隐式路径）
        'name': 'synthetic_implicit_missing',
        'dataset': 'synthetic',
        'x': ['num_normal', 'cat_ok'],
        'target': 'target',
        'methods': ['quantile', 'tree'],
        'kwargs': {'initial_bins': 20, 'bin_num_limit': 5},
    },
    {
        'name': 'synthetic_constant_and_many_cats',
        'dataset': 'synthetic',
        'x': ['const_int', 'many_cats', 'num_signal'],
        'target': 'target',
        'methods': ['quantile', 'tree'],
        'kwargs': {'initial_bins': 20, 'bin_num_limit': 5},
    },
    {
        # B-4 修复护栏：float 列 + 显式数值型特殊值（数据无 NaN，'missing'
        # 条目不产生分箱；-999.0 单独成箱）。修复前该路径在 pandas 3 下抛
        # ValueError（merge 键 object↔float64 冲突）。
        'name': 'synthetic_numeric_special_values',
        'dataset': 'synthetic',
        'x': ['num_skewed', 'num_signal'],
        'target': 'target',
        'methods': ['quantile', 'tree'],
        'kwargs': {'initial_bins': 20, 'bin_num_limit': 5},
        'special_values': {'num_skewed': ['-999', 'missing']},
    },
    {
        # B-5 修复护栏：int64 列 + 'missing' 特殊值（sv 值集合含 NaN →
        # 保持 float 语义，'-1' 的 bin 标签为 '-1.0'）。修复前该路径抛
        # ValueError: cannot convert float NaN to integer。
        'name': 'synthetic_integer_missing_special_value',
        'dataset': 'synthetic',
        'x': ['num_discrete'],
        'target': 'target',
        'methods': ['quantile', 'tree'],
        'kwargs': {'initial_bins': 20, 'bin_num_limit': 5},
        'special_values': {'num_discrete': ['-1', 'missing']},
    },
    {
        'name': 'germancredit_quantile_ib20',
        'dataset': 'germancredit',
        'x': None,  # 全变量（除 target）
        'target': 'creditability',
        'methods': ['quantile'],
        'kwargs': {'initial_bins': 20},
    },
    {
        'name': 'germancredit_quantile_tree_ib20_limit5',
        'dataset': 'germancredit',
        'x': None,
        'target': 'creditability',
        'methods': ['quantile', 'tree'],
        'kwargs': {'initial_bins': 20, 'bin_num_limit': 5},
    },
    {
        'name': 'germancredit_quantile_chi2_ib20_limit5',
        'dataset': 'germancredit',
        'x': None,
        'target': 'creditability',
        'methods': ['quantile', 'chi2'],
        'kwargs': {'initial_bins': 20, 'bin_num_limit': 5},
    },
    {
        # 类别型特殊值：显式 'others' + 隐式 missing（数值列缺失见 test_known_bugs.py）
        'name': 'germancredit_special_values',
        'dataset': 'germancredit',
        'x': [
            'status.of.existing.checking.account',
            'purpose',
            'other.installment.plans',
            'credit.amount',
        ],
        'target': 'creditability',
        'methods': ['quantile', 'tree'],
        'kwargs': {'initial_bins': 20, 'bin_num_limit': 5},
        'special_values': {
            'status.of.existing.checking.account': ['missing'],
            'purpose': ['missing', 'others'],
            'other.installment.plans': ['none'],
        },
    },
]


# --------------------------------------------------------------------------- #
# 归一化：把分箱结果转成确定性、可 JSON 化的结构
# --------------------------------------------------------------------------- #

def _jsonable(value: Any) -> Any:
    """numpy / pandas 标量 → JSON 原生类型；浮点统一 round 到 FLOAT_DIGITS。"""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if np.isnan(value):
            return None
        if np.isinf(value):
            return 'inf' if value > 0 else '-inf'
        return round(value, FLOAT_DIGITS)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _normalize_bin_result(result: Any) -> Any:
    """把单个变量的分箱结果归一化。

    - ``str``（``'CONST'`` / ``'TOO_MANY_VALUES'``）原样保留状态字符串
    - ``DataFrame`` 按 :data:`SNAPSHOT_COLUMNS` 顺序逐行归一化
    """
    if isinstance(result, str):
        return result
    if not isinstance(result, pd.DataFrame):
        return f'<unexpected type: {type(result).__name__}>'

    missing = [c for c in SNAPSHOT_COLUMNS if c not in result.columns]
    assert not missing, f'分箱结果缺少快照列 {missing}；实际列: {result.columns.tolist()}'

    records = []
    for row in result[SNAPSHOT_COLUMNS].itertuples(index=False, name=None):
        record = {}
        for column, raw in zip(SNAPSHOT_COLUMNS, row):
            if column == 'breaks':
                record[column] = [] if raw is None else [_jsonable(v) for v in list(raw)]
            else:
                record[column] = _jsonable(raw)
        records.append(record)
    return records


def _run_case(case: Dict[str, Any], frame: pd.DataFrame) -> Dict[str, Any]:
    """执行单个用例，返回归一化后的快照内容。

    用例抛异常时**不向上抛出**，而是把异常类型与消息记入快照（``status='error'``）。
    这样做的目的：

    * 一个已知失败路径不会把整个 golden 套件变成 error，其余用例仍可比对；
    * 快照会“钉住”当前失败行为，若将来修复，比对会失败并提示重新生成，
      不会出现“悄悄变绿”的假通过。

    已知失败路径本身由 ``test_known_bugs.py`` 用显式断言与 xfail 覆盖。
    """
    target = case['target']
    x = case['x']
    if x is None:
        x = sorted(c for c in frame.columns if c != target)

    try:
        bins = woebin(
            frame,
            y=target,
            x=x,
            methods=case['methods'],
            special_values=case.get('special_values'),
            no_cores=1,
            **case['kwargs'],
        )
    except Exception as err:  # noqa: BLE001 —— 失败行为需要被快照记录
        return {
            'dataset': case['dataset'],
            'target': target,
            'methods': list(case['methods']),
            'kwargs': {k: _jsonable(v) for k, v in case['kwargs'].items()},
            'special_values': _jsonable(case.get('special_values')),
            'n_rows': int(frame.shape[0]),
            'status': 'error',
            'error_type': type(err).__name__,
            'error_message': str(err)[:300],
        }

    assert set(bins.keys()) == set(x), (
        f'用例 {case["name"]}：woebin 返回变量与请求变量不一致，'
        f'缺失={sorted(set(x) - set(bins.keys()))}，'
        f'多余={sorted(set(bins.keys()) - set(x))}'
    )

    variables = {}
    for variable in sorted(bins.keys()):
        variables[variable] = _normalize_bin_result(bins[variable])

    return {
        'dataset': case['dataset'],
        'target': target,
        'methods': list(case['methods']),
        'kwargs': {k: _jsonable(v) for k, v in case['kwargs'].items()},
        'special_values': _jsonable(case.get('special_values')),
        'n_rows': int(frame.shape[0]),
        'status': 'ok',
        'variables': variables,
    }


def _snapshot_path(dataset: str) -> Path:
    return GOLDEN_DIR / f'{dataset}_binning.json'


def _load_snapshot(dataset: str) -> Dict[str, Any]:
    path = _snapshot_path(dataset)
    if not path.exists():
        pytest.fail(
            f'缺少 golden 快照 {path}。首次生成或需要更新时执行：'
            f'UPDATE_GOLDEN=1 PYTHONHASHSEED=0 python -m pytest test/test_golden_binning.py'
        )
    with path.open(encoding='utf-8') as fh:
        return json.load(fh)


def _dump_snapshot(dataset: str, payload: Dict[str, Any]) -> None:
    path = _snapshot_path(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write('\n')


# --------------------------------------------------------------------------- #
# fixture：按数据集分别生成结果
#
# 关键点：synthetic 与 germancredit **分成两个 fixture**。若合成在一个 fixture 里
# 顺手加载 germancredit，则数据缺失时 `pytest.skip` 会把 12 个 golden 用例
# （含 6 个纯合成用例）全部跳过，等于在干净克隆 / CI 中丢掉全部护栏。
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GoldenRun:
    """一份数据集的用例执行结果与快照 payload。"""

    dataset: str
    results: Dict[str, Dict[str, Any]]
    payload: Dict[str, Any]

    def cases(self) -> Dict[str, Dict[str, Any]]:
        return self.payload['cases']


def _run_dataset(dataset: str, frame: pd.DataFrame) -> GoldenRun:
    results = {
        case['name']: _run_case(case, frame)
        for case in _CASES if case['dataset'] == dataset
    }
    payload = {
        'golden_version': GOLDEN_VERSION,
        'dataset': dataset,
        'float_digits': FLOAT_DIGITS,
        'snapshot_columns': SNAPSHOT_COLUMNS,
        'data_fingerprint': _data_fingerprint(frame),
        'cases': results,
    }
    return GoldenRun(dataset=dataset, results=results, payload=payload)


_SYNTHETIC_CASES = [c for c in _CASES if c['dataset'] == 'synthetic']
_GERMANCREDIT_CASES = [c for c in _CASES if c['dataset'] == 'germancredit']


@pytest.fixture(scope='module')
def synthetic_golden() -> GoldenRun:
    """合成数据用例结果（不依赖任何外部数据集）。"""
    return _run_dataset('synthetic', _synthetic_frame())


@pytest.fixture(scope='module')
def germancredit_golden() -> GoldenRun:
    """germancredit 用例结果（数据缺失时 skip，不影响合成用例）。"""
    return _run_dataset('germancredit', _germancredit_frame())


# --------------------------------------------------------------------------- #
# 测试
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    'case', _SYNTHETIC_CASES, ids=[c['name'] for c in _SYNTHETIC_CASES])
def test_golden_binning_case_synthetic(case, synthetic_golden):
    """合成数据用例与快照严格比对（无需外部数据集）。"""
    _check_golden_case(case, synthetic_golden)


@pytest.mark.parametrize(
    'case', _GERMANCREDIT_CASES, ids=[c['name'] for c in _GERMANCREDIT_CASES])
def test_golden_binning_case_germancredit(case, germancredit_golden):
    """germancredit 用例与快照严格比对（数据缺失时 skip）。"""
    _check_golden_case(case, germancredit_golden)


def _check_golden_case(case: Dict[str, Any], run: GoldenRun) -> None:
    """单用例比对：先校验数据指纹，再确认用例跑通，最后严格相等比较。"""
    assert run.dataset == case['dataset']
    actual = run.results[case['name']]

    # 前置校验：数据指纹 + 用例成功状态。
    # （已知失败路径不在本文件覆盖，见 test_known_bugs.py）
    _assert_fingerprint_matches(run)
    _assert_case_ok(case['name'], actual)

    if UPDATE_GOLDEN:
        _dump_snapshot(run.dataset, run.payload)
        pytest.skip(f'UPDATE_GOLDEN=1：已重写 {_snapshot_path(run.dataset)}')

    snapshot = _load_snapshot(run.dataset)
    assert snapshot.get('golden_version') == GOLDEN_VERSION, (
        f'快照版本不匹配（文件={snapshot.get("golden_version")}，'
        f'期望={GOLDEN_VERSION}）；请 UPDATE_GOLDEN=1 重新生成'
    )
    assert case['name'] in snapshot['cases'], (
        f'快照 {_snapshot_path(run.dataset)} 中没有用例 {case["name"]}；'
        f'请 UPDATE_GOLDEN=1 重新生成'
    )

    expected = snapshot['cases'][case['name']]
    assert actual == expected, _diff_message(case['name'], expected, actual)


def _assert_fingerprint_matches(run: GoldenRun) -> None:
    """校验快照里记录的数据指纹与当前生成/加载的数据一致。

    这一步专门用来把"数据生成不可移植"和"分箱行为变化"两类失败区分开：
    前者会在这里立刻失败并指出是哪一列，后者才会进入分箱 diff。
    历史教训：第一版合成数据用 RNG 内核生成，在 Linux x86_64 上与 macOS arm64
    逐位不同，导致 CI 上只有分箱 diff 可看，根因被埋在一大坨 JSON 里。
    """
    if UPDATE_GOLDEN:
        return
    try:
        snapshot = _load_snapshot(run.dataset)
    except Exception:
        return  # 快照缺失时由后续 _load_snapshot 给出更明确的提示
    expected = snapshot.get('data_fingerprint')
    if expected is None:
        pytest.fail(
            f'快照 {_snapshot_path(run.dataset)} 缺少 data_fingerprint 字段；'
            f'请 UPDATE_GOLDEN=1 重新生成'
        )
    actual = run.payload['data_fingerprint']
    if actual == expected:
        return

    detail = []
    if actual['n_rows'] != expected['n_rows']:
        detail.append(f"行数 {expected['n_rows']} -> {actual['n_rows']}")
    if actual['columns'] != expected['columns']:
        detail.append(f"列  {expected['columns']} -> {actual['columns']}")
    for column in expected['column_checksums']:
        exp = expected['column_checksums'][column]
        act = actual['column_checksums'].get(column, '<missing>')
        if exp != act:
            detail.append(f'列 {column}: checksum {exp} -> {act}')
    pytest.fail(
        f'{run.dataset} 数据指纹不匹配（数据生成不可移植或数据已变更）：\n  '
        + '\n  '.join(detail)
        + f'\n若是生成器变更，执行 UPDATE_GOLDEN=1 重新生成快照；'
          f'若是平台差异，请把生成器改为不依赖 RNG 内核/超越函数的确定性实现。'
    )


def _assert_case_ok(case_name: str, result: Dict[str, Any]) -> None:
    """确认 golden 用例成功执行（status == 'ok'）。"""
    if result.get('status') != 'ok':
        pytest.fail(
            f'golden 用例 {case_name} 未能成功执行：'
            f'{result.get("error_type")}: {result.get("error_message")}\n'
            f'若是新发现的 pandas 3 兼容问题，请先在 test_known_bugs.py 中'
            f'以 xfail 用例记录，再决定是否从 golden 用例集中移出。'
        )


@pytest.mark.parametrize('dataset', ['synthetic', 'germancredit'])
def test_golden_snapshot_matches_all_cases(dataset, request):
    """整份快照比对（失败时给出逐用例最小 diff，便于定位）。"""
    run = request.getfixturevalue(f'{dataset}_golden')
    assert run.dataset == dataset

    _assert_fingerprint_matches(run)
    for name in run.results:
        _assert_case_ok(name, run.results[name])

    if UPDATE_GOLDEN:
        _dump_snapshot(dataset, run.payload)
        pytest.skip(f'UPDATE_GOLDEN=1：已重写 {_snapshot_path(dataset)}')

    snapshot = _load_snapshot(dataset)
    expected_cases = snapshot['cases']
    actual_cases = run.results

    assert set(actual_cases) == set(expected_cases), (
        f'用例集合与快照不一致：'
        f'快照有 {sorted(set(expected_cases) - set(actual_cases))}，'
        f'代码有 {sorted(set(actual_cases) - set(expected_cases))}'
    )
    for name in sorted(expected_cases):
        assert actual_cases[name] == expected_cases[name], _diff_message(
            name, expected_cases[name], actual_cases[name])


def _diff_message(case_name: str, expected: Any, actual: Any) -> str:
    """生成最小可读 diff（定位到变量 / 分箱 / 字段）。"""
    lines = [f'golden 快照不一致：case={case_name}']
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        lines.append(f'  期望: {expected!r}')
        lines.append(f'  实际: {actual!r}')
        return '\n'.join(lines)

    for key in sorted(set(expected) | set(actual)):
        if key == 'variables':
            continue
        if expected.get(key) != actual.get(key):
            lines.append(f'  参数 {key}: 期望 {expected.get(key)!r} != 实际 {actual.get(key)!r}')

    exp_vars = expected.get('variables', {})
    act_vars = actual.get('variables', {})
    if set(exp_vars) != set(act_vars):
        lines.append(f'  变量集合: 快照={sorted(exp_vars)} 实际={sorted(act_vars)}')

    for variable in sorted(set(exp_vars) & set(act_vars)):
        exp_bins, act_bins = exp_vars[variable], act_vars[variable]
        if exp_bins == act_bins:
            continue
        if isinstance(exp_bins, str) or isinstance(act_bins, str):
            lines.append(f'  变量 {variable}: 期望 {exp_bins!r} != 实际 {act_bins!r}')
            continue
        lines.append(f'  变量 {variable}: 分箱数 期望 {len(exp_bins)} != 实际 {len(act_bins)}')
        for i, (exp_row, act_row) in enumerate(zip(exp_bins, act_bins)):
            for field in SNAPSHOT_COLUMNS:
                if exp_row.get(field) != act_row.get(field):
                    lines.append(
                        f'    bin[{i}].{field}: 期望 {exp_row.get(field)!r} '
                        f'!= 实际 {act_row.get(field)!r}'
                    )
    return '\n'.join(lines)

# -*- encoding: utf-8 -*-
"""test/ 根级 pytest 配置。

集中处理两件事：

1. **数据集可用性**：本仓库的 ``data/*.csv.gz``（germancredit 17KB、
   creditcard ~65MB）**已随 Git 跟踪入库**，干净克隆即可运行全部集成/golden
   用例（CI 的 ``integration`` job 会真实执行 creditcard 全量）。
   ``require_data`` 的 skip 逻辑仅作为**兜底护栏**保留：当数据被人为移除、
   浅克隆缺文件或 ``SYRISKMODELS_DATA_DIR`` 指向空目录时，依赖真实数据集的
   用例应当**明确跳过并给出原因**，而不是以 ``FileNotFoundError`` 报错
   （那会被误读为代码缺陷）。
2. **确定性环境**：分箱结果只依赖输入数据，但为避免哈希随机化影响集合/字典
   遍历顺序，这里要求解释器以 ``PYTHONHASHSEED=0`` 启动（CI 已显式设置）。
   哈希种子在解释器启动时固化，无法在 conftest 中修改，因此这里**只校验不
   修改**：未设置 ``PYTHONHASHSEED=0`` 时 ``golden`` 用例会被跳过并说明原因，
   而不是静默地产生不可复现的快照。

测试内所有分箱/转换调用一律传 ``no_cores=1``，不使用 multiprocessing。
"""
import os
from pathlib import Path

import pytest

# 数据目录候选：优先环境变量，其次仓库根目录（本文件位于 <repo>/test/）
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = Path(os.environ.get('SYRISKMODELS_DATA_DIR') or (_REPO_ROOT / 'data'))

GERMANCREDIT_FILE = 'germancredit.csv.gz'
CREDITCARD_FILE = 'creditcard.csv.gz'

#: golden 快照要求解释器以固定哈希种子启动
HASH_SEED_OK = os.environ.get('PYTHONHASHSEED') == '0'


def data_file_path(name: str) -> Path:
    """返回 ``data/`` 下数据集文件的路径（不校验存在性）。"""
    return _DATA_DIR / name


def has_data(name: str) -> bool:
    """数据集文件是否存在且非空。"""
    path = data_file_path(name)
    return path.is_file() and path.stat().st_size > 0


def require_data(name: str) -> Path:
    """返回数据集路径；不存在时 skip 当前用例并说明原因与获取方式。

    注意：``data/*.csv.gz`` 正常随仓库跟踪，干净克隆下本函数不应触发 skip；
    触发即说明本地数据被移除或 ``SYRISKMODELS_DATA_DIR`` 配置有误。
    """
    if not has_data(name):
        pytest.skip(
            f'缺少数据集 {name}（查找路径 {data_file_path(name)}）。'
            f'该文件正常随 Git 入库（data/ 目录）；请检查是否被本地删除，'
            f'或 SYRISKMODELS_DATA_DIR 是否指向了错误目录。'
        )
    return data_file_path(name)


@pytest.fixture(scope='session')
def germancredit_file() -> Path:
    """germancredit 数据集路径（缺失则跳过）。"""
    return require_data(GERMANCREDIT_FILE)


@pytest.fixture(scope='session')
def creditcard_file() -> Path:
    """creditcard 数据集路径（缺失则跳过）。"""
    return require_data(CREDITCARD_FILE)


@pytest.fixture(scope='session')
def germancredit_df():
    """germancredit DataFrame（缺失则跳过）。"""
    require_data(GERMANCREDIT_FILE)
    from syriskmodels.datasets import load_germancredit
    return load_germancredit()


def pytest_report_header(config) -> str:
    """在 pytest 头部打印数据集可用性与哈希种子，便于 CI 日志溯源。"""
    status = ', '.join(
        f'{name}={"present" if has_data(name) else "MISSING"}'
        for name in (GERMANCREDIT_FILE, CREDITCARD_FILE)
    )
    return (
        f'riskmodels datasets: {status} (dir={_DATA_DIR}); '
        f'PYTHONHASHSEED={os.environ.get("PYTHONHASHSEED", "<unset>")}'
    )


def pytest_collection_modifyitems(config, items) -> None:
    """哈希种子未固定时跳过 golden 快照用例（避免产生不可复现的基线）。"""
    if HASH_SEED_OK:
        return
    skip_marker = pytest.mark.skip(
        reason='golden 快照要求在启动时固定哈希种子：'
               '请以 PYTHONHASHSEED=0 python -m pytest ... 运行'
    )
    for item in items:
        if item.get_closest_marker('golden') is not None:
            item.add_marker(skip_marker)

# -*- encoding: utf-8 -*-
"""随机流指纹（W1 报告 §1.1 的低成本建议，W2 落地）。

目的
----
``build_scorecard`` 用 ``np.random.default_rng(random_state).random(n)``
做训练/测试集划分。一旦 numpy 版本升级改变该随机流，02_test 划分会漂移，
集成测试结果随之变化，且**症状会出现在离根因很远的地方**（W1 报告 B-14
同类隐患）。本文件把若干 ``Generator`` 采样值的校验和钉死：流变化时这里
立刻失败并给出明确提示，而不是让下游用例"偶尔结果不同"。

跨平台说明
----------
只钉**与架构无关**的流：``Generator.random()``（xoshiro256** 原始 64bit
输出 → double 的精确转换）与 ``Generator.integers()``（纯整数域）。
**不**钉 ``standard_normal`` / ``binomial`` / ``lognormal`` —— 这些分布在
不同 CPU 架构走不同 SIMD 内核、逐位不同（W1 报告 B-14 的教训，golden
合成数据因此已改为纯整数算术生成）。
"""
import hashlib

import numpy as np

#: default_rng(0).random(64) 的前 4 个值（repr 精度）
_EXPECTED_RANDOM_FIRST4 = [
    0.6369616873214543,
    0.2697867137638703,
    0.04097352393619469,
    0.016527635528529094,
]
#: default_rng(0).random(64) 原始字节的 sha256 前 32 位
_EXPECTED_RANDOM_SHA256_32 = '97a942a4e10b14332c04a8f35a5fbd61'
#: default_rng(42).integers(0, 10**9, 8)
_EXPECTED_INTEGERS = [
    89250953, 773956048, 654571518, 438878439,
    433015235, 858597919, 201469535, 94177347,
]


def test_default_rng_random_stream_fingerprint():
    """``default_rng(seed).random(n)`` 流指纹（build_scorecard 划分依赖）。"""
    values = np.random.default_rng(0).random(64)
    assert values[:4].tolist() == _EXPECTED_RANDOM_FIRST4, (
        'numpy Generator.random() 随机流已变化！build_scorecard 的 '
        '02_test 划分与依赖它的集成测试结果将漂移。请核对 numpy 版本，'
        '并在确认后更新本指纹与相关基线（见 W1 报告 §1.1 / B-14）。'
    )
    digest = hashlib.sha256(values.tobytes()).hexdigest()[:32]
    assert digest == _EXPECTED_RANDOM_SHA256_32, (
        f'random(64) 校验和变化：{digest} != {_EXPECTED_RANDOM_SHA256_32}')


def test_default_rng_integers_stream_fingerprint():
    """``default_rng(seed).integers(...)`` 流指纹（测试数据生成依赖）。"""
    values = np.random.default_rng(42).integers(0, 10**9, 8)
    assert values.tolist() == _EXPECTED_INTEGERS, (
        'numpy Generator.integers() 随机流已变化！依赖固定种子整数数据的'
        '测试（known_bugs/equivalence fuzz 等）输入将漂移。请核对 numpy '
        '版本后更新本指纹。'
    )

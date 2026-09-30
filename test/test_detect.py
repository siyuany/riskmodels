# -*- encoding: utf-8 -*-
"""detector.detect 单元测试。

确定性：用固定种子的 ``np.random.default_rng(0)`` 生成数据，不依赖全局随机
状态（``np.random.rand`` 需显式 seed，容易引入不可复现结果）。

并行：显式传 ``n_cores=1``，避免 ProcessPoolExecutor 在 pytest 下的
spawn/pickle 问题（见 W1 报告 B-11）。
"""
import numpy as np
import pandas as pd
import pytest

from syriskmodels.detector import detect


@pytest.fixture
def detect_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.random((50, 100)),
        columns=['V' + str(i) for i in range(100)],
    )


class TestDetect:

    def test_detect_shape(self, detect_df):
        res1 = detect(detect_df, n_cores=1)
        res2 = detect(detect_df, n_cores=1)
        assert res1.shape[0] == 100
        assert res2.shape[0] == 100

    def test_detect_deterministic(self, detect_df):
        """同一输入两次运行结果一致（确定性护栏）。"""
        res1 = detect(detect_df, n_cores=1)
        res2 = detect(detect_df, n_cores=1)
        pd.testing.assert_frame_equal(res1, res2)

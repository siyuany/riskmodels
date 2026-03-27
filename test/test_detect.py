# -*- encoding: utf-8 -*-
import numpy as np
import pandas as pd
import unittest

from syriskmodels.detector import detect


class TestDetect(unittest.TestCase):

    def test_detect(self):
        arr = np.random.rand(50, 100)
        df = pd.DataFrame(arr, columns=['V' + str(i) for i in range(100)])
        # 使用 n_cores=1 避免 pytest 下 ProcessPoolExecutor 的 pickle 问题
        # （子进程与主进程的 concurrent.futures 模块引用可能不一致）
        res1 = detect(df, n_cores=1)
        res2 = detect(df, n_cores=1)
        self.assertEqual(res1.shape[0], 100)
        self.assertEqual(res2.shape[0], 100)


if __name__ == '__main__':
    unittest.main()
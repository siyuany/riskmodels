# -*- encoding: utf-8 -*-
"""
从 test/dataset.h5 中导出 germancredit、creditcard 为 CSV，供集成测试使用。
在项目根目录执行: python test/export_dataset_csv.py
"""
import os
import sys

# 保证可找到 dataset.h5（与 test 同目录）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))

import pandas as pd

H5_PATH = os.path.join(_SCRIPT_DIR, 'dataset.h5')
OUT_GERMAN = os.path.join(_SCRIPT_DIR, 'germancredit.csv')
OUT_CREDIT = os.path.join(_SCRIPT_DIR, 'creditcard.csv')


def main():
    if not os.path.isfile(H5_PATH):
        print(f'未找到 {H5_PATH}，请先准备 dataset.h5')
        sys.exit(1)
    df1 = pd.read_hdf(H5_PATH, key='germancredit')
    df2 = pd.read_hdf(H5_PATH, key='creditcard')
    df1.to_csv(OUT_GERMAN, index=False)
    df2.to_csv(OUT_CREDIT, index=False)
    print(f'已导出: {OUT_GERMAN}, {OUT_CREDIT}')


if __name__ == '__main__':
    main()

# -*- encoding: utf-8 -*-

import os
import sys


def addpath():
    cur_path = os.path.dirname(os.path.abspath(__file__))
    lib_path = os.path.join(os.path.dirname(cur_path), 'src')
    sys.path.insert(0, lib_path)
    return sys.path


if __name__ == '__main__':
    import syriskmodels

    print(syriskmodels.__path__)

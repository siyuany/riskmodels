# -*- encoding: utf-8 -*-
"""scorecard 真实数据集集成测试。

数据来源
--------
统一走 :mod:`syriskmodels.datasets` 的 ``load_germancredit()`` /
``load_creditcard()``，不再依赖 ``test/germancredit.csv``、
``test/creditcard.csv`` 这类**硬编码且不入版本库**的路径
（仓库只提供被 .gitignore 忽略的 ``test/*.csv.gz`` 软链接，干净克隆后
``test/*.csv`` 从不曾存在，历史实现因此必然失败）。

历史实现中的 ``_load_csvs_in_subprocess``（子进程读 CSV → pickle 回传）是为了
规避“pytest 下主进程读 CSV 产生 ``_NoValueType``”的旧问题，该问题在当前
pandas 3 下已无法复现（见 ``docs/plans/w1-baseline-report.md`` 的 bug 清单），
因此改为普通 ``pandas`` 读取，可读性优先。数据集在模块级缓存，只加载一次
（历史实现每个测试的 ``setUp`` 都重新读一遍 creditcard，全量 gz 解压约 1s，
7 个用例重复 7 次）。

确定性与并行
------------
* 所有分箱 / 转换调用显式传 ``no_cores=1``，避免 multiprocessing spawn。
* ``creditcard`` 相关用例标记 ``@pytest.mark.slow``：数据文件约 65MB（解压后
  约 1.3GB 量级），属于大数据量用例。

断言口径与历史实现保持一致，仅做必要的形式调整（``unittest.TestCase`` 的
``setUp`` 改为 pytest 的 autouse fixture —— 项目其余测试已统一为 pytest 风格）。
"""
import numpy as np
import pandas as pd
import pytest

from syriskmodels.scorecard import (
    WOEBinFactory,
    WOEBin,
    woebin,
    RuleOptimBin,
    QuantileInitBin,
    sc_bins_to_df,
)
from syriskmodels.contrib.build_scorecard import build_scorecard

# --------------------------------------------------------------------------- #
# 数据集加载（模块级缓存，只读一次）
# --------------------------------------------------------------------------- #

_GERMANCREDIT_DF = None
_CREDITCARD_DF = None


def _germancredit() -> pd.DataFrame:
    """germancredit（1000 行）；数据缺失时 skip 整个模块的用例。"""
    global _GERMANCREDIT_DF
    if _GERMANCREDIT_DF is None:
        from syriskmodels.datasets import load_germancredit
        try:
            _GERMANCREDIT_DF = load_germancredit()
        except FileNotFoundError as err:
            pytest.skip(f'缺少 germancredit 数据集：{err}')
    return _GERMANCREDIT_DF


def _creditcard() -> pd.DataFrame:
    """creditcard 全量（284807 行）；数据缺失时 skip 整个模块的用例。"""
    global _CREDITCARD_DF
    if _CREDITCARD_DF is None:
        from syriskmodels.datasets import load_creditcard
        try:
            _CREDITCARD_DF = load_creditcard()
        except FileNotFoundError as err:
            pytest.skip(f'缺少 creditcard 数据集：{err}')
    return _CREDITCARD_DF


# --------------------------------------------------------------------------- #
# 轻量用例（不依赖数据集，可在 unit CI 中运行）
# --------------------------------------------------------------------------- #

class TestWOEBinUnit:
    """不依赖真实数据集的 WOEBin 用例。"""

    def test_split_vec_to_df(self):
        x = ['b%,%d', 'a', 'c%,%e']
        df = WOEBin.split_vec_to_df(x)
        assert set(df['bin_chr'].unique()) == set(x)


# --------------------------------------------------------------------------- #
# germancredit 集成用例
# --------------------------------------------------------------------------- #

@pytest.mark.integration
class TestWOEBinGermancreditIntegration:
    """germancredit 上的分箱集成用例。"""

    def test_woebin_cat_vars(self):
        df = _germancredit()
        tmp_df = pd.DataFrame({
            'variable': 'property',
            'value': df['property'],
            'y': np.where(df['creditability'] == 1, 1, 0),
        })
        woe_bin_method = WOEBinFactory.build(['quantile', 'tree'])
        binning_result = woe_bin_method(tmp_df)
        assert isinstance(binning_result, pd.DataFrame)
        assert not binning_result.empty

    def test_woebin_all_vars_germancredit(self):
        """全变量分箱（不含特殊值），验证核心路径在真实数据上可跑通。"""
        df = _germancredit()
        xs = [c for c in df.columns if c != 'creditability']
        bins = woebin(
            df,
            y='creditability',
            x=xs,
            methods=['quantile', 'tree'],
            initial_bins=20,
            bin_num_limit=5,
            no_cores=1,
        )
        woe, iv = sc_bins_to_df(bins)
        assert isinstance(woe, pd.DataFrame)
        assert not woe.empty
        assert isinstance(iv, pd.DataFrame)
        assert not iv.empty
        # 变量顺序无关，按集合比较
        assert set(iv.index) == set(xs)


# --------------------------------------------------------------------------- #
# creditcard 集成用例（大数据量 → slow）
# --------------------------------------------------------------------------- #

@pytest.mark.slow
@pytest.mark.integration
class TestRuleOptimBinIntegration:
    """RuleOptimBin 在 creditcard 全量数据上的集成用例。"""

    @pytest.fixture(autouse=True)
    def _prepare(self):
        self.df = _germancredit()
        self.df2 = _creditcard()
        pd.set_option('display.max_columns', 100)

    def test_rulebin_single_variable(self):
        dtm = self.df2[['V3', 'Class']].copy()
        dtm.rename(columns={'V3': 'value', 'Class': 'y'}, inplace=True)
        dtm['variable'] = 'V3'
        q_bin = QuantileInitBin(initial_bins=50)
        breaks = q_bin.woebin(dtm)
        r_bin = RuleOptimBin()
        r_bin.woebin(dtm, breaks)

        woebin_res = woebin(
            self.df2,
            x=['V3'],
            y='Class',
            methods=[QuantileInitBin(initial_bins=50), RuleOptimBin()],
            no_cores=1,
        )['V3']
        assert isinstance(woebin_res, pd.DataFrame)

    def test_rulebin_another_variable(self):
        dtm = self.df2[['V4', 'Class']].copy()
        dtm.rename(columns={'V4': 'value', 'Class': 'y'}, inplace=True)
        dtm['variable'] = 'V4'
        q_bin = QuantileInitBin(initial_bins=50)
        breaks = q_bin.woebin(dtm)
        r_bin = RuleOptimBin()
        r_bin.woebin(dtm, breaks)

        woebin_res = woebin(
            self.df2,
            x=['V4'],
            y='Class',
            methods=[QuantileInitBin(initial_bins=50), RuleOptimBin()],
            no_cores=1,
        )['V4']
        assert isinstance(woebin_res, pd.DataFrame)

    def test_rulebin_multiple_variables(self):
        variables = ['V' + str(i) for i in range(1, 29)]
        bins = woebin(
            self.df2,
            x=variables,
            y='Class',
            methods=[QuantileInitBin(50), RuleOptimBin()],
            no_cores=1,
        )
        woe, _ = sc_bins_to_df(bins)
        assert isinstance(woe, pd.DataFrame)
        assert not woe.empty


@pytest.mark.slow
@pytest.mark.integration
class TestWOEBinIntegration:
    """creditcard 全量数据上的分箱 / 评分卡流水线集成用例。"""

    @pytest.fixture(autouse=True)
    def _prepare(self):
        self.df = _germancredit()
        self.df2 = _creditcard()

    def test_chi2_woebin(self):
        binner = WOEBinFactory.build(['quantile', 'chi2'])
        tmp_df = pd.DataFrame({
            'variable': 'V3',
            'value': self.df2['V3'],
            'y': self.df2['Class'],
        })
        binning_result = binner(tmp_df)
        assert isinstance(binning_result, pd.DataFrame)
        assert not binning_result.empty

    def test_build_scorecard_pipeline(self, monkeypatch, tmp_path):
        """creditcard 上的端到端评分卡流水线。

        并行控制
        --------
        ``build_scorecard`` 内部经 ``woebin_psi → woebin_ply(no_cores=None)``
        可能触发 ``mp.Pool``。这里把 ``woebin_ply`` 固定为 ``no_cores=1``
        （仅并行度，不改变任何算法与结果），与 W1「测试一律 no_cores=1」的
        约束一致。``woebin`` 已通过 ``binning_kwargs={'no_cores': 1}`` 覆盖。

        历史（B-2，W2 已修复）
        ----------------------
        ``woebin_plot`` 曾因 pandas 3 的 ``groupby.apply`` 丢失分组列而抛
        ``KeyError('variable')``，本用例被迫 ``xfail(strict=False)`` 并用
        "witness" 断言兜底。B-2 修复后 xfail 与 witness 已移除，整条流水线
        （分箱/逐步回归/VIF/评分卡/PSI/Excel/绘图）必须真实全绿。

        ``build_scorecard`` 末尾会把 WOE 图写入 CWD 的 ``pic/`` 目录，
        因此通过 ``monkeypatch.chdir(tmp_path)`` 隔离到临时目录，
        避免污染仓库根目录。
        """
        import functools

        from syriskmodels.scorecard.api import transform as _transform

        # 并行度固定为 1（woebin_psi 内部会重新 import woebin_ply）
        monkeypatch.setattr(
            _transform, 'woebin_ply',
            functools.partial(_transform.woebin_ply, no_cores=1),
        )
        monkeypatch.chdir(tmp_path)

        features = self.df2.columns.tolist()[1:-1]
        excel_path = tmp_path / 'scorecard.xlsx'
        build_scorecard(
            self.df2,
            features=features,
            target='Class',
            train_filter=lambda x: x['Time'] <= 140000,
            oot_filter=lambda x: x['Time'] > 140000,
            output_excel_file=str(excel_path),
            cv=3,
            binning_kwargs={'no_cores': 1},
        )
        # 全流程成功：Excel 产物非空 + WOE 图已保存到 pic/
        assert excel_path.exists() and excel_path.stat().st_size > 0
        pic_dir = tmp_path / 'pic'
        assert pic_dir.is_dir()
        assert len(list(pic_dir.glob('*.png'))) > 0

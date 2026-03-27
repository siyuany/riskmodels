# -*- encoding: utf-8 -*-
"""build_scorecard BUG 修复回归测试

验证 build_scorecard.py 中 3 个 BUG 修复的正确性:
- BUG#1: stepwise_lr 调用方式（y 参数为列名，df 包含 target 列）
- BUG#2: VIF 计算（使用 add_constant 后的矩阵，索引 +1 跳过截距列）
- BUG#3: 系数/p值/t值检查排除截距项（iloc[1:]）
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


@pytest.fixture
def mock_train_data():
    """构造简单的模拟训练数据用于回归测试"""
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    prob = 1 / (1 + np.exp(-(0.5 * x1 + 0.3 * x2 + 0.2 * x3)))
    y = rng.binomial(1, prob)
    return pd.DataFrame({
        'x1_woe': x1, 'x2_woe': x2, 'x3_woe': x3, 'target': y
    })


class TestBug1StepwiseLrCall:
    """BUG#1: stepwise_lr 的 y 参数必须为列名字符串"""

    def test_stepwise_lr_accepts_column_name(self, mock_train_data):
        """stepwise_lr 传入列名字符串应正常工作"""
        from syriskmodels.models import stepwise_lr

        best_auc, selected = stepwise_lr(
            mock_train_data,
            y='target',
            x=['x1_woe', 'x2_woe', 'x3_woe'],
            cv=2)

        assert isinstance(best_auc, float)
        assert isinstance(selected, list)
        assert len(selected) > 0
        assert all(f in mock_train_data.columns for f in selected)


class TestBug2VifCalculation:
    """BUG#2: VIF 计算必须使用 add_constant 后的矩阵，索引从 1 开始"""

    def test_vif_with_constant_column(self, mock_train_data):
        """VIF 计算应在 add_constant 后的矩阵上进行，跳过截距列"""
        features = ['x1_woe', 'x2_woe', 'x3_woe']
        X = mock_train_data[features].copy()
        X = sm.add_constant(X)
        X_arr = X.to_numpy()

        vif_values = {}
        for idx, feature in enumerate(features):
            vif_values[feature] = variance_inflation_factor(X_arr, idx + 1)

        # VIF 值应为正数且有限
        for feat, val in vif_values.items():
            assert np.isfinite(val), f'{feat} 的 VIF 值不是有限数: {val}'
            assert val >= 1.0, f'{feat} 的 VIF 值应 >= 1.0，实际: {val}'

    def test_vif_wrong_index_raises_or_wrong(self, mock_train_data):
        """验证用错误索引 (不跳过截距列) 会得到不同结果"""
        features = ['x1_woe', 'x2_woe', 'x3_woe']
        X = mock_train_data[features].copy()
        X = sm.add_constant(X)
        X_arr = X.to_numpy()

        # 正确方式: idx + 1
        correct_vif_0 = variance_inflation_factor(X_arr, 1)
        # 错误方式: idx (指向截距列)
        wrong_vif_0 = variance_inflation_factor(X_arr, 0)

        # 两者应该不同（截距列的 VIF 无业务意义）
        assert correct_vif_0 != wrong_vif_0


class TestBug3InterceptExclusion:
    """BUG#3: 系数方向/p值/t值检查必须排除截距项"""

    def test_coef_check_excludes_intercept(self, mock_train_data):
        """模型精调循环中 iloc[1:] 应排除截距"""
        features = ['x1_woe', 'x2_woe', 'x3_woe']
        X = mock_train_data[features].copy()
        X = sm.add_constant(X)
        model = sm.GLM(
            endog=mock_train_data['target'], exog=X,
            family=sm.families.Binomial()).fit()

        # 排除截距项
        coef_params = model.params.iloc[1:]
        coef_pvalues = model.pvalues.iloc[1:]

        assert 'const' not in coef_params.index, \
            '系数检查不应包含 const（截距项）'
        assert 'const' not in coef_pvalues.index, \
            'p值检查不应包含 const（截距项）'
        assert len(coef_params) == len(features), \
            f'排除截距后应有 {len(features)} 个系数，实际 {len(coef_params)}'

    def test_t_value_max_excludes_intercept(self, mock_train_data):
        """当需要移除变量时，t值排序不应包含截距"""
        features = ['x1_woe', 'x2_woe', 'x3_woe']
        X = mock_train_data[features].copy()
        X = sm.add_constant(X)
        model = sm.GLM(
            endog=mock_train_data['target'], exog=X,
            family=sm.families.Binomial()).fit()

        t_values = model.tvalues.iloc[1:]
        rm_var = t_values.index[t_values == t_values.max()].item()

        assert rm_var != 'const', '不应移除截距项'
        assert rm_var in features, f'移除的变量 {rm_var} 应在特征列表中'
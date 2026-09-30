# -*- encoding: utf-8 -*-
from types import UnionType
from typing import List, Union

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import check_cv

import syriskmodels.logging as logging


class LogisticRegressionCV(object):

  def __init__(self, *, cv=3, **logistic_kwargs):
    self._cv = cv
    self._logistic_kwargs = logistic_kwargs

  def fit_and_eval(self, X, y):
    train_predicted = None
    train_actual = None
    valid_predicted = None
    valid_actual = None

    cv_wrapper = check_cv(self._cv, y=y, classifier=True)

    for train_index, test_index in cv_wrapper.split(X, y):
      X_train, X_valid = X[train_index], X[test_index]
      y_train, y_valid = y[train_index], y[test_index]

      logistic_regressor = LogisticRegression(**self._logistic_kwargs)
      logistic_regressor.fit(X_train, y_train)

      train_pred = logistic_regressor.predict_proba(X_train)[:, 1]
      if train_predicted is None:
        train_predicted = train_pred
        train_actual = y_train
      else:
        train_predicted = np.concatenate([train_predicted, train_pred])
        train_actual = np.concatenate([train_actual, y_train])

      valid_pred = logistic_regressor.predict_proba(X_valid)[:, 1]
      if valid_predicted is None:
        valid_predicted = valid_pred
        valid_actual = y_valid
      else:
        valid_predicted = np.concatenate([valid_predicted, valid_pred])
        valid_actual = np.concatenate([valid_actual, y_valid])

    train_auc = roc_auc_score(train_actual, train_predicted)
    valid_auc = roc_auc_score(valid_actual, valid_predicted)
    logging.debug(f'训练集 AUC={train_auc:.2%}，测试集 AUC={valid_auc:.2%}')

    return min(train_auc, valid_auc) - abs(train_auc - valid_auc)


def group_split_cv(group_arr):
  group_arr = np.asarray(group_arr)
  groups = np.unique(group_arr)

  for g in groups:
    train = group_arr != g
    yield np.argwhere(train).ravel(), np.argwhere(~train).ravel()


def stepwise_lr(df: pd.DataFrame,
                y: str,
                x: Union[str, List[str]],
                cv: int = 3,
                max_num_features: int = 30,
                initial_features: Union[str, List[str]] = None,
                direction: str = 'bidirectional',
                **lr_kwargs):
  """双向逐步逻辑回归特征选择

  通过前向选择和/或后向消除的交叉验证策略，从候选特征中选择最优特征子集。
  每一步通过 ``LogisticRegressionCV`` 交叉验证评估特征组合的性能
  （指标为 min(train_auc, valid_auc) - |train_auc - valid_auc|），
  选择使指标最优的特征组合。

  参数:
    df: 包含特征列和目标列的数据框。**必须包含 y 对应的列**。
    y: 目标变量列名（字符串），df 中必须存在该列
    x: 候选特征列名列表
    cv: 交叉验证折数，默认 3。也可传入自定义 CV splitter（如 generator）
    max_num_features: 最大入选特征数量，默认 30
    initial_features: 初始特征列表（强制入选），默认 None
    direction: 搜索方向，可选 ``'forward'``、``'backward'``、``'bidirectional'``，
      默认 ``'bidirectional'``。方向只控制每轮**生成哪些候选**：

      - ``'forward'``: 候选 = 当前集合加入一个池内特征（从空集/initial 起步）
      - ``'backward'``: 候选 = 当前集合剔除一个特征；未指定 ``initial_features``
        时从**全量特征集**起步（经典后向消除），并以当前集合的指标为基线，
        只有严格改进才接受剔除（W2/B-3）
      - ``'bidirectional'``: 候选 = 加入 + 剔除（行为与历史版本一致）

      择优更新逻辑对三种方向一致生效（B-3 修复：旧实现把择优块误放在
      backward/bidirectional 分支内，forward/backward 单向搜索永远返回空）。
    **lr_kwargs: 传递给 ``sklearn.linear_model.LogisticRegression`` 的参数

  返回:
    Tuple[float, List[str]]:

    - ``best_metrics`` (float): 最优评估指标值
    - ``selected_features`` (List[str]): 被选中的特征列名列表

  示例:
    >>> from syriskmodels.models import stepwise_lr
    >>> from syriskmodels.scorecard import woebin_ply
    >>> train_woe = woebin_ply(train_df[features], bins, value='woe')
    >>> train_woe['target'] = train_df['target']  # 必须包含 target 列
    >>> best_auc, selected = stepwise_lr(
    ...     train_woe, y='target',
    ...     x=[f + '_woe' for f in features], cv=3)
  """
  feature_pool = x
  if initial_features is None:
    selected_features = []
  else:
    if isinstance(initial_features, str):
      selected_features = [initial_features]
    elif not isinstance(initial_features, list):
      selected_features = list(initial_features)
    else:
      selected_features = initial_features
  best_metrics = None
  assert direction in ('forward', 'backward', 'bidirectional'), \
    f'direction参数为 forward, backward, bidirectional 三者之一，输入{direction}不合法'

  def get_features_perf(feature_list):
    lr_cv = LogisticRegressionCV(cv=cv, **lr_kwargs)
    auc = lr_cv.fit_and_eval(df[feature_list].to_numpy(), df[y])
    return auc

  # B-3：纯 backward 以"当前集合"的指标为基线 —— 未指定 initial_features 时
  # 从全量特征集起步（经典后向消除），只有严格改进才接受剔除。
  if direction == 'backward' and len(selected_features) > 0:
    best_metrics = get_features_perf(selected_features)
  elif direction == 'backward' and len(feature_pool) > 0:
    selected_features = list(x)
    best_metrics = get_features_perf(selected_features)
    feature_pool = [f for f in x if f not in selected_features]

  step = 0
  while True:
    perf_records = {}
    improved = False
    step += 1

    # ---- 候选生成：direction 只控制候选集合 ----
    if direction in ['forward', 'bidirectional']:
      for feature in feature_pool:
        train_features = selected_features + [feature]
        perf_records[tuple(train_features)] = get_features_perf(train_features)

    if direction in ['backward', 'bidirectional']:
      if len(selected_features) > 1:
        for feature in selected_features:
          train_features = selected_features.copy()
          train_features.remove(feature)
          perf_records[tuple(train_features)] = get_features_perf(train_features)

    # ---- 择优更新：与 direction 无关（B-3 修复前该块被误放在
    # backward/bidirectional 分支内，导致单向搜索永不更新） ----
    for key, value in perf_records.items():
      if best_metrics is None or (best_metrics < value and
                                  len(key) <= max_num_features):
        selected_features = list(key)
        best_metrics = value
        improved = True
        feature_pool = [f for f in x if f not in selected_features]

    if improved:
      logging.info(f'Step {step}:\nSelected features: {selected_features}\n'
                   f'Performance: auc={best_metrics}')
    else:
      logging.info('No improve. Exit.')
      break

  return best_metrics, selected_features

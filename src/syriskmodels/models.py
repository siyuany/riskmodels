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
    auc = lr_cv.fit_and_eval(df[feature_list].values, df[y])
    return auc

  step = 0
  while True:
    perf_records = {}
    improved = False
    step += 1

    if direction in ['forward', 'bidirectional']:
      for feature in feature_pool:
        train_features = selected_features + [feature]
        perf_records[tuple(train_features)] = get_features_perf(train_features)

      if len(selected_features) > 1:
        for feature in selected_features:
          train_features = selected_features.copy()
          train_features.remove(feature)
          perf_records[tuple(train_features)] = get_features_perf(train_features)

    if direction in ['backward', 'bidirectional']:
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

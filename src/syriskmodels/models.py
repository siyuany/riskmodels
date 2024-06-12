# -*- encoding: utf-8 -*-

import numpy as np

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
    logging.info(f'训练集 AUC={train_auc:.2%}，测试集 AUC={valid_auc:.2%}')

    return min(train_auc, valid_auc) - abs(train_auc - valid_auc)


def group_split_cv(group_arr):
  group_arr = np.asarray(group_arr)
  groups = np.unique(group_arr)

  for g in groups:
    train = group_arr != g
    yield np.argwhere(train).ravel(), np.argwhere(~train).ravel()


def stepwise_lr(df, y, x, cv=3, max_num_features=30, **lr_kwargs):
  feature_pool = x
  selected_features = []
  best_metrics = None

  def get_features_perf(feature_list):
    lr_cv = LogisticRegressionCV(cv=cv, **lr_kwargs)
    auc = lr_cv.fit_and_eval(df[feature_list].values, y)
    return auc

  step = 0
  while True:
    perf_records = {}
    improved = False
    step += 1

    for feature in feature_pool:
      train_features = selected_features + [feature]
      perf_records[tuple(train_features)] = get_features_perf(train_features)

    if len(selected_features) > 1:
      for feature in selected_features:
        train_features = selected_features.copy()
        train_features.remove(feature)
        perf_records[tuple(train_features)] = get_features_perf(train_features)

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

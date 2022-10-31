# -*- encoding: utf-8 -*-

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import check_cv

import riskmodels.logging as logging


class LogisticRegressionCV(object):

    def __init__(self, *, cv=3, **logistic_kwargs):
        self._cv = cv
        self._logistic_kwargs = logistic_kwargs

    def fit_and_eval(self, X, y):
        y_predicted = None
        y_actual = None

        cv_wrapper = check_cv(self._cv, y=y, classifier=True)

        for train_index, test_index in cv_wrapper.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            logistic_regressor = LogisticRegression(**self._logistic_kwargs)
            logistic_regressor.fit(X_train, y_train)

            y_pred = logistic_regressor.predict_proba(X_test)[:, 1]

            if y_predicted is None:
                y_predicted = y_pred
                y_actual = y_test
            else:
                y_predicted = np.concatenate([y_predicted, y_pred])
                y_actual = np.concatenate([y_actual, y_test])

        auc = roc_auc_score(y_actual, y_predicted)
        return auc


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
            perf_records[tuple(train_features)] = get_features_perf(
                train_features)

        if len(selected_features) > 1:
            for feature in selected_features:
                train_features = selected_features.copy()
                train_features.remove(feature)
                perf_records[tuple(train_features)] = get_features_perf(
                    train_features)

        for key, value in perf_records.items():
            if best_metrics is None \
                    or (best_metrics < value and len(key) <= max_num_features):
                selected_features = list(key)
                best_metrics = value
                improved = True
                feature_pool = [f for f in x if f not in selected_features]

        if improved:
            logging.info(
                f'Step {step}:\nSelected features: {selected_features}\n'
                f'Performance: auc={best_metrics}')
        else:
            logging.info('No improve. Exit.')
            break

    return best_metrics, selected_features

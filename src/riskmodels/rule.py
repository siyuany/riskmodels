# -*- encoding: utf-8 -*-

import numpy as np
import pandas as pd


def make_rule_stat(bad_cnt_fn, good_cnt_fn):

    def rule_stat(df):
        cnt = len(df)
        bad = bad_cnt_fn(df)
        good = good_cnt_fn(df)
        badrate = bad / (bad + good)
        return pd.Series({
            'cnt': cnt,
            'bad': bad,
            'good': good,
            'badrate': badrate
        })

    return rule_stat


def foil_gain(pos0, neg0, pos1, neg1):
    """FOIL增益计算"""
    if pos0 == 0:
        return -np.inf
    if neg0 == 0:
        neg0 += 0.5

    info1 = np.log2(pos0 / (pos0 + neg0))
    info2 = np.log2(pos1 / (pos1 + neg1))

    return pos0 * (info1 - info2)


def eval_rules(df, stat_fn, *rule):
    rules = list(rule)
    metrics = []

    for rule in rules:
        hit_metric = df.groupby(rule(df)).apply(stat_fn)

        if np.any(hit_metric.index):
            hit_rate = hit_metric['cnt'][
                hit_metric.index].item() / hit_metric['cnt'].sum()
            hit_bad = hit_metric['bad'][hit_metric.index].item()
            hit_good = hit_metric['good'][hit_metric.index].item()
            all_bad = hit_metric['bad'].sum()
            all_good = hit_metric['good'].sum()

            if hit_bad + hit_good == 0:
                hit_badrate = 0
                foil = np.inf
            else:
                hit_badrate = hit_bad / (hit_bad + hit_good)
                foil = foil_gain(hit_bad, hit_good, all_bad, all_good)

            metrics.append({
                'rule': rule.__name__,
                'hit': hit_rate,
                'badrate': hit_badrate,
                'foil': foil
            })
        else:
            metrics.append({
                'rule': rule.__name__,
                'hit': 0,
                'badrate': np.nan,
                'foil': 0
            })

    return pd.DataFrame(metrics)


def rule_optimize(df, stat_fn, *rule):
    rules = list(rule)
    named_rules = {rule.__name__: rule for rule in rules}
    opt_result = None
    tmp_df = df.copy()

    while len(tmp_df) > 0 and len(named_rules) > 0:
        rule_result = eval_rules(tmp_df, stat_fn, *named_rules.values())
        best_rule = rule_result.sort_values(
            'foil', ascending=False).iloc[0, ]
        best_rule_name = best_rule['rule']

        if best_rule['foil'] > 0:
            rule = named_rules.pop(best_rule_name)
            tmp_df = tmp_df[~rule(tmp_df)]

            if opt_result is None:
                opt_result = best_rule.to_frame().T

            else:
                opt_result = pd.concat(
                    [opt_result, best_rule.to_frame().T], ignore_index=True)
        else:
            break

    return opt_result

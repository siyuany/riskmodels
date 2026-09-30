#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""``woebin`` 性能基准。

用法::

    python bench/bench_binning.py --dataset germancredit --methods quantile tree \\
        --initial-bins 20 --repeats 3 --output bench/results/baseline_german.json

    python bench/bench_binning.py --dataset creditcard --nrows 50000 \\
        --methods quantile tree --initial-bins 100

约束
----
* 一律 ``no_cores=1``（见 W1 报告 B-11：默认并行路径在 spawn 下不安全，
  且在小数据上更慢）。
* 只调用公共 API ``syriskmodels.scorecard.woebin``，不触碰实现。
* 1 次 warmup + ``--repeats`` 次计时取中位数。
"""
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402  (需要先插入 bench/ 到 sys.path)
    build_payload,
    default_numeric_variables,
    default_variables,
    load_dataset,
    per_variable_seconds,
    print_case_summary,
    time_case,
    write_payload,
)

from syriskmodels.scorecard import sc_bins_to_df, woebin  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='woebin 分箱性能基准（固定 no_cores=1）',
    )
    parser.add_argument(
        '--dataset', choices=['germancredit', 'creditcard'], required=True,
        help='数据集名称',
    )
    parser.add_argument(
        '--nrows', type=int, default=None,
        help='只取前 N 行（确定性 head 抽样）；默认全量',
    )
    parser.add_argument(
        '--methods', nargs='+', default=['quantile', 'tree'],
        help='分箱方法列表，如 --methods quantile tree',
    )
    parser.add_argument(
        '--initial-bins', type=int, nargs='+', default=[20],
        help='细分箱数量，可传多个（每个生成一个用例）',
    )
    parser.add_argument(
        '--bin-num-limit', type=int, default=5,
        help='粗分箱后最大分箱数，默认 5',
    )
    parser.add_argument(
        '--variables', nargs='*', default=None,
        help='指定变量子集；默认取全部解释变量',
    )
    parser.add_argument(
        '--numeric-only', action='store_true',
        help='只使用数值型变量（creditcard 的 V1..Vn 场景）',
    )
    parser.add_argument(
        '--max-variables', type=int, default=None,
        help='数值型变量个数上限（与 --numeric-only 搭配）',
    )
    parser.add_argument('--repeats', type=int, default=3, help='计时次数，默认 3')
    parser.add_argument('--warmup', type=int, default=1, help='预热次数，默认 1')
    parser.add_argument(
        '--output', default=None,
        help='输出 JSON 路径；默认 bench/results/binning_<date>.json',
    )
    parser.add_argument('--label', default=None, help='用例前缀标签（可选）')
    return parser.parse_args(argv)


def build_cases(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """构造用例列表（纯声明，便于 dry-run 检查）。"""
    cases = []
    for initial_bins in args.initial_bins:
        cases.append({
            'methods': list(args.methods),
            'initial_bins': initial_bins,
            'bin_num_limit': args.bin_num_limit,
        })
    return cases


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    df, dataset_meta = load_dataset(args.dataset, args.nrows)

    if args.variables:
        variables = [v for v in args.variables if v in df.columns]
        missing = sorted(set(args.variables) - set(variables))
        if missing:
            print(f'[warn] 数据集中不存在这些变量，已忽略: {missing}', file=sys.stderr)
    elif args.numeric_only:
        variables = default_numeric_variables(
            df, dataset_meta['target'], limit=args.max_variables)
    else:
        variables = default_variables(df, dataset_meta['target'])
        if args.max_variables is not None:
            variables = variables[:args.max_variables]

    if not variables:
        print('[error] 没有可用变量', file=sys.stderr)
        return 2

    target = dataset_meta['target']
    label = f'{args.label}_' if args.label else ''
    print(
        f"dataset={dataset_meta['name']} rows={dataset_meta['n_rows']} "
        f"({dataset_meta['sampling']}) variables={len(variables)} "
        f"target={target} no_cores=1"
    )

    case_specs = build_cases(args)
    results: List[Dict[str, Any]] = []

    for spec in case_specs:
        case_id = (
            f"{label}{dataset_meta['name']}"
            f"_rows{dataset_meta['n_rows']}"
            f"_vars{len(variables)}"
            f"_{'+'.join(spec['methods'])}"
            f"_ib{spec['initial_bins']}"
            f"_lim{spec['bin_num_limit']}"
        )
        shape: Dict[str, Any] = {}

        def run_binning() -> Dict[str, Any]:
            return woebin(
                df,
                y=target,
                x=variables,
                methods=spec['methods'],
                initial_bins=spec['initial_bins'],
                bin_num_limit=spec['bin_num_limit'],
                no_cores=1,
            )

        def capture(bins: Dict[str, Any]) -> None:
            woe, iv = sc_bins_to_df(bins)
            shape['n_bins'] = int(0 if woe is None else len(woe))
            shape['n_iv_rows'] = int(0 if iv is None else len(iv))
            shape['n_skipped'] = int(
                sum(1 for v in bins.values() if isinstance(v, str)))
            shape['skipped_statuses'] = sorted({
                v for v in bins.values() if isinstance(v, str)
            })

        timing = time_case(
            run_binning,
            repeats=args.repeats,
            warmup=args.warmup,
            on_result=capture,
        )

        case = {
            'case_id': case_id,
            'dataset': dataset_meta['name'],
            'n_rows': dataset_meta['n_rows'],
            'sampling': dataset_meta['sampling'],
            'target': target,
            'n_variables': len(variables),
            'variables': list(variables),
            'methods': list(spec['methods']),
            'initial_bins': spec['initial_bins'],
            'bin_num_limit': spec['bin_num_limit'],
            'no_cores': 1,
            'timing': timing,
            'per_variable_seconds': per_variable_seconds(
                timing['median_seconds'], len(variables)),
            **shape,
        }
        results.append(case)
        print_case_summary(case)

    payload = build_payload(
        suite='binning',
        cases=results,
        extra={'dataset': dataset_meta, 'args': vars(args)},
    )
    path = write_payload(payload, args.output)
    print(f'\nbaseline written: {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

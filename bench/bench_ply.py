#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""``woebin_ply`` 性能基准。

流程：先在（抽样后的）数据上 ``woebin`` 一次得到 bins，再对同一份数据
反复执行 ``woebin_ply``（``no_cores=1``），测量 WOE 转换耗时。

用法::

    python bench/bench_ply.py --dataset creditcard --nrows 50000 \\
        --max-variables 12 --repeats 3 --value woe

    python bench/bench_ply.py --bins-from bench/results/xxx.json  # 复用已有分箱
"""
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    build_payload,
    default_numeric_variables,
    default_variables,
    load_dataset,
    per_variable_seconds,
    print_case_summary,
    time_case,
    write_payload,
)

from syriskmodels.scorecard import woebin, woebin_ply  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='woebin_ply 转换性能基准（固定 no_cores=1）',
    )
    parser.add_argument(
        '--dataset', choices=['germancredit', 'creditcard'], default='creditcard',
        help='数据集名称',
    )
    parser.add_argument('--nrows', type=int, default=50000,
                        help='只取前 N 行（确定性 head 抽样）')
    parser.add_argument('--max-variables', type=int, default=12,
                        help='参与转换的变量个数上限')
    parser.add_argument('--variables', nargs='*', default=None,
                        help='指定变量子集')
    parser.add_argument(
        '--all-variables', action='store_true',
        help='使用全部解释变量（默认只取数值型变量，最多 --max-variables 个）',
    )
    parser.add_argument(
        '--value', choices=['woe', 'index', 'bin'], default='woe',
        help='woebin_ply 的 value 参数',
    )
    parser.add_argument(
        '--binning-methods', nargs='+', default=['quantile', 'tree'],
        help='生成 bins 时使用的分箱方法',
    )
    parser.add_argument('--initial-bins', type=int, default=20,
                        help='生成 bins 时的细分箱数量')
    parser.add_argument('--bin-num-limit', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--output', default=None)
    parser.add_argument('--label', default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    label = f'{args.label}_' if args.label else ''

    df, dataset_meta = load_dataset(args.dataset, args.nrows)
    target = dataset_meta['target']

    if args.variables:
        variables = [v for v in args.variables if v in df.columns]
    elif args.all_variables:
        variables = default_variables(df, target)[:args.max_variables]
    else:
        variables = default_numeric_variables(df, target, limit=args.max_variables)
        if not variables:
            variables = default_variables(df, target)[:args.max_variables]

    if not variables:
        print('[error] 没有可用变量', file=sys.stderr)
        return 2

    print(
        f"dataset={dataset_meta['name']} rows={dataset_meta['n_rows']} "
        f"({dataset_meta['sampling']}) variables={len(variables)} "
        f"value={args.value} no_cores=1"
    )

    # ---- 分箱（一次性，不计入基准） ----
    bins = woebin(
        df, y=target, x=variables,
        methods=args.binning_methods,
        initial_bins=args.initial_bins,
        bin_num_limit=args.bin_num_limit,
        no_cores=1,
    )
    usable = [v for v in variables if not isinstance(bins.get(v), str)]
    print(f'  binning done: {len(usable)}/{len(variables)} 变量可转换')

    if not usable:
        print('[error] 没有任何变量成功分箱', file=sys.stderr)
        return 2

    shape: Dict[str, Any] = {}

    def run_ply():
        return woebin_ply(df[usable], {v: bins[v] for v in usable},
                          no_cores=1, value=args.value)

    def capture(result) -> None:
        shape['n_output_rows'] = int(result.shape[0])
        shape['n_output_cols'] = int(result.shape[1])
        shape['n_nan_cells'] = int(result.isna().to_numpy().sum())

    timing = time_case(run_ply, repeats=args.repeats, warmup=args.warmup,
                       on_result=capture)

    case_id = (
        f"{label}{dataset_meta['name']}"
        f"_rows{dataset_meta['n_rows']}"
        f"_vars{len(usable)}"
        f"_ply-{args.value}"
    )
    case = {
        'case_id': case_id,
        'dataset': dataset_meta['name'],
        'n_rows': dataset_meta['n_rows'],
        'sampling': dataset_meta['sampling'],
        'target': target,
        'n_variables': len(usable),
        'variables': list(usable),
        'value': args.value,
        'binning_methods': list(args.binning_methods),
        'initial_bins': args.initial_bins,
        'bin_num_limit': args.bin_num_limit,
        'no_cores': 1,
        'timing': timing,
        'per_variable_seconds': per_variable_seconds(
            timing['median_seconds'], len(usable)),
        'rows_per_second': round(
            dataset_meta['n_rows'] / timing['median_seconds'], 1
        ) if timing['median_seconds'] > 0 else None,
        **shape,
    }
    print_case_summary(case)

    payload = build_payload(
        suite='ply',
        cases=[case],
        extra={'dataset': dataset_meta, 'args': vars(args)},
    )
    path = write_payload(payload, args.output)
    print(f'\nbaseline written: {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

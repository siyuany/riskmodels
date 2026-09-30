#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""一键运行 W1 固定基准用例，输出单一 baseline JSON。

固定用例（见 docs/plans/w1-baseline-report.md）
-----------------------------------------------
1. germancredit 全变量（20 个）
   - ``quantile`` / ``quantile+tree`` / ``quantile+chi2``，``initial_bins=20``
2. creditcard 抽样 50k，7 个数值变量
   - ``quantile+tree`` 与 ``quantile+chi2``，``initial_bins`` 20 / 100
   - ``initial_bins=500`` 耗时显著更长，默认关闭，用 ``--include-ib500`` 打开
3. ``woebin_ply``：50k × 12 变量，``value='woe'``，``no_cores=1``

用法::

    PYTHONHASHSEED=0 python bench/run_all.py
    PYTHONHASHSEED=0 python bench/run_all.py --repeats 3 --warmup 1 \\
        --output bench/results/baseline_20260101.json
    PYTHONHASHSEED=0 python bench/run_all.py --quick          # 冒烟（repeats=1）
    PYTHONHASHSEED=0 python bench/run_all.py --include-ib500  # 打开重用例
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench_binning  # noqa: E402
import bench_ply  # noqa: E402
from common import build_payload, write_payload  # noqa: E402


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='W1 性能基线：一键运行固定用例')
    parser.add_argument('--repeats', type=int, default=3, help='计时次数，默认 3')
    parser.add_argument('--warmup', type=int, default=1, help='预热次数，默认 1')
    parser.add_argument(
        '--output', default=None,
        help='输出 JSON；默认 bench/results/baseline_<date>.json',
    )
    parser.add_argument('--german-rows', type=int, default=None,
                        help='germancredit 行数上限（默认全量 1000）')
    parser.add_argument('--credit-rows', type=int, default=50000,
                        help='creditcard 抽样行数，默认 50000')
    parser.add_argument('--credit-vars', type=int, default=7,
                        help='creditcard 数值变量个数，默认 7')
    parser.add_argument('--ply-vars', type=int, default=12,
                        help='woebin_ply 变量个数，默认 12')
    parser.add_argument('--include-ib500', action='store_true',
                        help='包含 initial_bins=500 的重用例（耗时显著更长）')
    parser.add_argument('--quick', action='store_true',
                        help='冒烟模式：repeats=1、warmup=0、跳过 ib>=100 用例')
    return parser.parse_args(argv)


def _run_suite(module, argv: List[str], run_dir: Path) -> Dict[str, Any]:
    """以子命令方式调用 bench 脚本，并读取其写出的 JSON。"""
    out_path = run_dir / f'{module.__name__}.json'
    full_argv = argv + ['--output', str(out_path)]
    print(f'\n=== {" ".join([module.__name__] + full_argv)} ===')
    code = module.main(full_argv)
    if code != 0:
        raise SystemExit(f'{module.__name__} 退出码 {code}')
    import json
    with out_path.open(encoding='utf-8') as fh:
        return json.load(fh)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    repeats = 1 if args.quick else args.repeats
    warmup = 0 if args.quick else args.warmup
    credit_initial_bins = [20] if args.quick else [20, 100]
    if args.include_ib500:
        credit_initial_bins = credit_initial_bins + [500]

    stamp = datetime.now().strftime('%Y%m%d')
    run_dir = Path(__file__).resolve().parent / 'results' / '.parts'
    run_dir.mkdir(parents=True, exist_ok=True)

    common_timing = ['--repeats', str(repeats), '--warmup', str(warmup)]
    datasets: Dict[str, Any] = {}
    all_cases: List[Dict[str, Any]] = []

    # ---- 1. germancredit：全变量三种方法组合 ----
    german_base = ['--dataset', 'germancredit'] + common_timing
    if args.german_rows:
        german_base += ['--nrows', str(args.german_rows)]

    for methods in (['quantile'], ['quantile', 'tree'], ['quantile', 'chi2']):
        payload = _run_suite(
            bench_binning,
            german_base + ['--methods'] + methods + ['--initial-bins', '20'],
            run_dir,
        )
        all_cases.extend(payload['cases'])
        datasets['germancredit'] = payload['dataset']

    # ---- 2. creditcard：50k 抽样，7 个数值变量 ----
    for methods in (['quantile', 'tree'], ['quantile', 'chi2']):
        payload = _run_suite(
            bench_binning,
            ['--dataset', 'creditcard', '--nrows', str(args.credit_rows),
             '--numeric-only', '--max-variables', str(args.credit_vars)]
            + common_timing
            + ['--methods'] + methods
            + ['--initial-bins'] + [str(b) for b in credit_initial_bins],
            run_dir,
        )
        all_cases.extend(payload['cases'])
        datasets['creditcard'] = payload['dataset']

    # ---- 3. woebin_ply：50k × 12 变量 ----
    ply_payload = _run_suite(
        bench_ply,
        ['--dataset', 'creditcard', '--nrows', str(args.credit_rows),
         '--max-variables', str(args.ply_vars), '--value', 'woe']
        + common_timing,
        run_dir,
    )
    all_cases.extend(ply_payload['cases'])

    payload = build_payload(
        suite='baseline',
        cases=all_cases,
        extra={
            'datasets': datasets,
            'run_args': vars(args),
            'effective_repeats': repeats,
            'effective_warmup': warmup,
            'creditcard_initial_bins': credit_initial_bins,
            'notes': [
                '所有用例 no_cores=1',
                'median 为 repeats 次计时的中位数，含 1 次 warmup（warmup 不计入）',
                'creditcard 采用确定性 head 抽样，sampling 字段标注 full/head',
            ],
        },
    )

    output = args.output
    if output is None:
        output = str(Path('bench') / 'results' / f'baseline_{stamp}.json')
    path = write_payload(payload, output)

    print('\n=== W1 性能基线汇总 ===')
    print(f'{"case_id":<56s} {"median(s)":>10s} {"per_var(s)":>11s}')
    for case in all_cases:
        print(
            f"{case['case_id']:<56s} "
            f"{case['timing']['median_seconds']:10.3f} "
            f"{case.get('per_variable_seconds', float('nan')):11.4f}"
        )
    print(f'\nbaseline written: {path}')
    print(f'cases: {len(all_cases)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

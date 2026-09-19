#!/usr/bin/env python3
"""Run the PRNN workflow and export exactly 23 worksheets to revised_model_data.xlsx."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['full', 'check'], default='full',
                        help='full: complete study; check: small real-data software test.')
    parser.add_argument('--data', type=Path, default=None,
                        help='Default: dataset/dataset_all.csv or dataset_all.csv.gz, with case-insensitive filename matching.')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--device', choices=['auto', 'cpu', 'gpu'], default=None)
    parser.add_argument('--precision', choices=['float32', 'mixed_float16'], default=None)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--sample-cases', type=int, default=256,
                        help='Number of full-data training cases sampled in check mode (128-2048).')
    parser.add_argument('--check-epochs', type=int, default=None,
                        help='1-10 epochs per candidate for check mode only; defaults to 3.')
    parser.add_argument('--no-resume', action='store_true',
                        help='Ignore earlier completed runs; reuse identical runs within this invocation.')
    parser.add_argument('--allow-different-dataset', action='store_true',
                        help='Allow a deliberately changed dataset while retaining row-count and domain checks.')
    args = parser.parse_args()
    if args.check_epochs is not None and args.mode != 'check':
        parser.error('--check-epochs is only valid with --mode check.')
    from prnn.config import ROOT, RunOptions
    from prnn.pipeline import run_pipeline
    output = args.output or ROOT / ('outputs' if args.mode == 'full' else f'outputs_{args.mode}')
    epochs = args.check_epochs if args.check_epochs is not None else (3 if args.mode == 'check' else 70)
    device = args.device or ('gpu' if args.mode == 'full' else 'auto')
    precision = args.precision or ('mixed_float16' if args.mode == 'full' else 'float32')
    options = RunOptions(
        mode=args.mode, output=output, data=args.data,
        epochs=epochs, device=device, precision=precision,
        threads=args.threads, resume=not args.no_resume,
        strict_reference=not args.allow_different_dataset,
        sample_cases=args.sample_cases,
    )
    try:
        report = run_pipeline(options)
    except (ValueError, RuntimeError, OSError, ImportError, FloatingPointError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    print(f"Workbook (23 worksheets): {report['workbook']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

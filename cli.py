#!/usr/bin/env python3
"""ConvoBench CLI for running comprehensive benchmark sweeps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from convobench.store import RunStore
from convobench.sweep import SweepConfig, run_sweep


def main():
    parser = argparse.ArgumentParser(
        prog="convobench",
        description="Run comprehensive benchmark sweeps for ConvoBench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # sweep command
    sweep_parser = subparsers.add_parser("sweep", help="Run a full benchmark sweep")
    sweep_parser.add_argument("--runs", type=int, default=1, help="Runs per scenario/baseline/model combo")
    sweep_parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducibility")
    sweep_parser.add_argument("--output", type=str, default=None, help="Path to save JSON results")
    sweep_parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    if args.command == "sweep":
        store = RunStore()
        config = SweepConfig(
            runs_per_combo=args.runs,
            seed=args.seed,
            verbose=not args.quiet,
        )

        result = run_sweep(config, store=store)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({
                "sweep_id": result.sweep_id,
                "started_at": result.started_at,
                "finished_at": result.finished_at,
                "total_combos": result.total_combos,
                "completed": result.completed,
                "results": result.results,
            }, indent=2, default=str), encoding="utf-8")
            print(f"Results saved to {out_path}")

        print(f"\nSweep {result.sweep_id} complete: {result.completed}/{result.total_combos} succeeded")
        print("View results in the frontend at http://localhost:3000/benchmarks")


if __name__ == "__main__":
    main()

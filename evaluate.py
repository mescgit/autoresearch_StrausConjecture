# -*- coding: utf-8 -*-
"""
evaluate.py — Runs one experiment and compares to best. Do NOT modify.

Usage:
    python evaluate.py          # run one experiment
    python evaluate.py --summary  # print experiment history
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import copy

# Fix Windows cp1252 stdout so Unicode box-drawing characters render correctly
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from prepare import load_best, load_experiment_log, print_summary, BEST_FILE

SEARCH_FILE = "search.py"
SEARCH_BACKUP = "search.py.bak"


def run_experiment() -> int:
    """Run search.py as a subprocess and return max_n_verified."""
    print(f"\n{'═'*50}")
    print(f"  Starting experiment...")
    print(f"  Current best: {load_best():,}")
    print(f"{'═'*50}\n")

    t0 = time.time()
    env = copy.copy(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        [sys.executable, SEARCH_FILE],
        capture_output=False,
        text=True,
        env=env,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n✗ search.py crashed (exit code {result.returncode})")
        return None

    # Read result from log (last entry)
    experiments = load_experiment_log()
    if not experiments:
        return None

    last = experiments[-1]
    max_n = last["max_n_verified"]
    print(f"\n  Elapsed: {elapsed:.1f}s | max_n_verified: {max_n:,}")
    return max_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true", help="Print experiment summary")
    parser.add_argument("--auto", type=int, default=1, metavar="N",
                        help="Run N experiments automatically (default: 1)")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    for i in range(args.auto):
        if args.auto > 1:
            print(f"\n[Auto-run {i+1}/{args.auto}]")

        prev_best = load_best()

        # Backup current search.py before running
        shutil.copy(SEARCH_FILE, SEARCH_BACKUP)

        max_n = run_experiment()

        if max_n is None:
            print("  Experiment failed — restoring backup")
            shutil.copy(SEARCH_BACKUP, SEARCH_FILE)
            continue

        if max_n > prev_best:
            print(f"  ✓ Improvement: {prev_best:,} → {max_n:,}")
        else:
            print(f"  No improvement this run.")


if __name__ == "__main__":
    main()

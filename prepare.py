"""
prepare.py — Constants, utilities, logging. Do NOT modify.
"""

import json
import os
import time
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
TIME_BUDGET_SECONDS = 300   # 5 minutes per experiment, wall clock
LOG_FILE = "experiments.jsonl"
BEST_FILE = "best.json"

# ── Logging ───────────────────────────────────────────────────────────────────

def log_result(max_n_verified: int, counterexample: int = None):
    """Append experiment result to the log file."""
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "max_n_verified": max_n_verified,
        "counterexample": counterexample,
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Update best if improved
    best = load_best()
    if max_n_verified > best:
        with open(BEST_FILE, "w") as f:
            json.dump({"max_n_verified": max_n_verified}, f)
        print(f"✓ New best: {max_n_verified:,}")
    else:
        print(f"  No improvement (best: {best:,}, this run: {max_n_verified:,})")


def load_best() -> int:
    """Load the current best max_n_verified."""
    if os.path.exists(BEST_FILE):
        with open(BEST_FILE) as f:
            return json.load(f).get("max_n_verified", 0)
    return 0


def load_experiment_log() -> list:
    """Load all past experiment results."""
    if not os.path.exists(LOG_FILE):
        return []
    results = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def print_summary():
    """Print a summary of all experiments."""
    experiments = load_experiment_log()
    if not experiments:
        print("No experiments yet.")
        return

    print(f"\n{'─'*50}")
    print(f"  Experiment History ({len(experiments)} runs)")
    print(f"{'─'*50}")
    for i, exp in enumerate(experiments):
        marker = "★" if exp["max_n_verified"] == max(e["max_n_verified"] for e in experiments) else " "
        cp = f"  ⚠ COUNTEREXAMPLE at n={exp['counterexample']}" if exp.get("counterexample") else ""
        print(f"  {marker} Run {i+1:3d} | {exp['timestamp']} | max_n = {exp['max_n_verified']:>12,}{cp}")
    print(f"{'─'*50}")
    print(f"  Best: {load_best():,}")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    print_summary()

# erdos-autoresearch

An autoresearch setup (inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch))
for pushing the computational frontier of the **Erdős–Straus Conjecture**.

## The Conjecture

For every integer n ≥ 2, there exist positive integers a, b, c such that:

```
4/n = 1/a + 1/b + 1/c
```

Verified up to ~10^14. Never proven. Never disproven.

## The Idea

Give an AI agent a search codebase and let it experiment autonomously overnight.
It modifies `search.py`, runs for 5 minutes, checks if it verified more n than before,
and repeats. You wake up to a log of experiments and (hopefully) a new personal record.

## Files

| File | Purpose | Edit? |
|------|---------|-------|
| `search.py` | The search algorithm | ✅ Agent edits this |
| `program.md` | Agent instructions | ✅ You edit this |
| `prepare.py` | Constants + logging utilities | ❌ Fixed |
| `evaluate.py` | Runs experiments + tracks best | ❌ Fixed |

## Quick Start

```bash
# Requirements: Python 3.10+, pip
pip install numpy

# Test a single run (takes 5 minutes)
python evaluate.py

# See experiment history
python evaluate.py --summary

# Run 10 experiments back to back
python evaluate.py --auto 10
```

## Running with an Agent

Point Claude (or any coding agent) at this repo and say:

```
Have a look at program.md and let's kick off a new experiment!
```

The agent will:
1. Read `program.md` for context
2. Check past results in `experiments.jsonl`
3. Propose an improvement to `search.py`
4. Run `python evaluate.py`
5. Repeat

## Metric

**`max_n_verified`** — the largest n for which the conjecture has been verified in this session.
Higher is better. Each 5-minute run tries to push this number up.

## What to Expect

| Approach | ~n/5min |
|----------|---------|
| Baseline pure Python | 10K–50K |
| Numpy vectorized | 100K–500K |
| Numba JIT compiled | 1M–10M |
| Residue shortcuts + Numba | 10M+ |

## If a Counterexample Is Found

`find_representation()` returns `None` → logged immediately with the value of n.
This would be one of the most significant mathematical discoveries of the century.
Please email everyone.

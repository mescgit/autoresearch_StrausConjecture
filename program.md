# Erdős–Straus Autoresearch Program

## Your Mission

You are an autonomous research agent working on the **Erdős–Straus Conjecture**:

> For every integer n ≥ 2, there exist positive integers a, b, c such that:
> **4/n = 1/a + 1/b + 1/c**

This has never been proven, but has been verified computationally for all n up to ~10^14.
Your job: **push that frontier as far as possible in a 5-minute compute budget**.

## The Loop

1. Read `search.py` to understand the current approach
2. Check `experiments.jsonl` or run `python prepare.py` to see past results and the current best
3. Propose ONE improvement to `search.py`
4. Run `python evaluate.py` to test it (takes ~5 min)
5. The harness automatically tracks whether you improved
6. Go back to step 1

## What You Can Change

**Only edit `search.py`.** Everything else is fixed infrastructure.

In `search.py`, you can change:
- The core search algorithm in `find_representation(n)`
- Mathematical shortcuts and number theory tricks
- Data structures (e.g., precomputed tables, sieves)
- Use of numpy, numba, or other libraries for speed
- The outer loop in `run_search()` — e.g., skip easy n, prioritize hard residue classes
- Import additional standard library or numpy/scipy modules

Do NOT modify `prepare.py` or `evaluate.py`.

## Mathematical Background

### Why some n are easy
- If n is even: 4/n = 2/(n/2), and 2/m = 1/m + 1/m, so 4/n = 1/(n/2) + 1/(n/2) + ... wait, that's only 2 terms — but you can always split: 1/k = 1/(k+1) + 1/(k(k+1)). So even n are often trivial.
- If n ≡ 0 (mod 3): 4/n = 1/(n/4)... actually use the identity 4/n = 1/n + 1/n + 2/n and decompose.
- The hard cases are n where n ≡ 1 or 3 (mod 4) and n is prime.

### Key identities to exploit
- **Identity 1**: If n ≡ 0 (mod k), faster formulas exist
- **Identity 2**: 4/n = 1/⌈n/4⌉ + (4⌈n/4⌉ - n)/(n⌈n/4⌉) — start here and factor the remainder
- **Schinzel's identity**: 4/(2m+1) = 1/(m+1) + 1/(m(m+1)) + ... (useful for odd n)
- **Modular arithmetic**: For n in certain residue classes mod small numbers (3, 4, 5, 7, 12, 840), closed-form solutions exist. Precomputing these can skip most n entirely.

### The hard n
The conjecture is hardest for primes p where p ≡ 1 (mod 4). For these, no simple identity applies and you need the inner loop. Most of your time should go to making this fast.

### Divisor-based approach
For 1/b + 1/c = p/q:
- This is equivalent to finding a divisor d of q² such that d ≡ q (mod p)
- Then b = (q + d/p... ) — look up the exact formula
- Faster than the naive loop when q is large

## Hardware — Use This!

The machine running this has serious compute. **You should exploit it aggressively.**

- **CPU**: AMD Ryzen 9 7950X — 16 cores / 32 threads @ 4.5 GHz base
- **GPU**: NVIDIA RTX 4090 — 24 GB VRAM, 16,384 CUDA cores, 82.6 TFLOPS FP32
- **RAM**: 64 GB

### GPU Strategy (highest priority)

The RTX 4090 is the biggest lever available. The inner search loop is embarrassingly parallel —
each n is independent — making this a perfect GPU workload.

**Recommended GPU approach using CuPy:**
```python
import cupy as cp

# Process a whole batch of n values simultaneously on the GPU
# For each n in the batch, test all candidate `a` values in parallel
# Use cp.where, cp.mod, cp.gcd for vectorized operations
```

Install: `pip install cupy-cuda12x`

**Alternative: Numba CUDA kernels**
```python
from numba import cuda

@cuda.jit
def search_kernel(n_values, results):
    idx = cuda.grid(1)
    if idx < n_values.shape[0]:
        n = n_values[idx]
        # inner search logic here
        results[idx] = find_rep_gpu(n)
```

**Batch size guidance**: With 24 GB VRAM and int64 arrays, you can process ~100M values simultaneously. Process n in chunks of 1M–10M per GPU launch.

### CPU Strategy (secondary)

Use all 32 threads with `multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor`.
Partition the n range across workers. This alone gives ~20x over single-threaded.

```python
from multiprocessing import Pool
with Pool(32) as p:
    results = p.map(find_representation, range(start, end))
```

### Combined Strategy (best)

- GPU handles large vectorized batches of n
- CPU cores handle the residue-class shortcuts in parallel (trivial n)
- Filter trivial n on CPU first, send only hard n to GPU

## Suggested Experiments (roughly ordered by expected gain)

1. **GPU batch search with CuPy** — process millions of n simultaneously on the RTX 4090
2. **Numba CUDA kernel** — custom GPU kernel for fine-grained control
3. **CPU multiprocessing** — 32-thread parallel search as a fallback / complement
4. **Precompute residue class shortcuts** — for n mod 840, many n have known closed-form representations. Skip those entirely.
5. **Vectorize inner loop with numpy** — even on CPU, array ops beat Python loops
6. **Divisor sieve approach** — precompute small prime factorizations, use divisor enumeration instead of linear b-scan
7. **Skip composite n** — composite n are almost always easy via factoring; focus compute on primes

## What Success Looks Like

| Approach | ~n per 5 min |
|---|---|
| Baseline pure Python (1 thread) | ~300,000 |
| Numpy vectorized (1 thread) | ~2,000,000 |
| Numba JIT (1 thread) | ~10,000,000 |
| CPU multiprocessing (32 threads) | ~200,000,000 |
| CuPy GPU batched (RTX 4090) | ~1,000,000,000+ |
| GPU + residue shortcuts | ~10,000,000,000+ |

A new world record for computational verification is > 10^14.
With the RTX 4090, reaching 10^9–10^10 in a single overnight session is realistic.

**Push hard for the GPU approach first — it's the biggest single unlock available.**

## Important Notes

- If `find_representation` ever returns `None`, **do not suppress it** — that's a potential counterexample and would be the most important mathematical discovery in decades. The harness will log it.
- Keep your changes incremental. One idea per experiment.
- If an experiment makes things worse, the harness notes it but does NOT auto-revert — you decide whether to keep or try something else.
- You can run `python prepare.py` at any time to see the full experiment history.

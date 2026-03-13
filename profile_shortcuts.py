# -*- coding: utf-8 -*-
"""
profile_shortcuts.py - Profile the ~2% of hard_n not resolved by shortcuts.

For each unsolved n, records:
  - residue mod 840 (which of the 6 hard classes)
  - residues mod small primes/composites (5, 7, 11, 13, 17, 19, 23, 24, 120)
  - the smallest a_off that solves it (found by the deep kernel)
  - the k value used in the solution

Run once; takes ~1 batch (~0.5s on GPU) plus a CPU pass over the ~27K survivors.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import numpy as np
import cupy as cp
from collections import Counter, defaultdict
from prepare import load_best

# ---- replicate just what we need from search.py ----
HARD_RESIDUES = {1, 121, 169, 289, 361, 529}

def get_is_easy_table():
    is_easy = np.ones(840, dtype=bool)
    for r in HARD_RESIDUES:
        is_easy[r] = False
    return is_easy

IS_EASY_TABLE = get_is_easy_table()
NUM_SHORTCUTS = 6

KERNEL_CODE = r"""
__device__ __forceinline__ bool divides_q2(long long n, long long a, long long d) {
    unsigned long long nd = (unsigned long long)(n % d);
    unsigned long long ad = (unsigned long long)(a % d);
    unsigned long long qd = (nd * ad) % (unsigned long long)d;
    return (qd * qd) % (unsigned long long)d == 0;
}

__device__ bool try_a(long long n, long long a, int K) {
    if (a <= n / 4) return false;
    long long p = 4LL * a - n;
    if (p <= 0) return false;
    unsigned long long pu = (unsigned long long)p;
    unsigned long long rem = ((unsigned long long)(n % p) * (unsigned long long)(a % p)) % pu;
    for (int k = 1; k <= K; ++k) {
        long long d = (long long)k * p - (long long)rem;
        if (d > 0 && divides_q2(n, a, d))
            return true;
    }
    return false;
}

extern "C" __global__
void check_shortcuts_kernel(const long long* hard_n, bool* found_mask,
                             int num_hard, int K_SHORT) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)num_hard * 6) return;
    int n_idx = (int)(idx / 6);
    if (found_mask[n_idx]) return;
    long long n = hard_n[n_idx];
    int choice = (int)(idx % 6);
    long long a;
    switch (choice) {
        case 0: a = (n + 3)  / 4; break;
        case 1: a = (n + 7)  / 4; break;
        case 2: a = (n + 11) / 4; break;
        case 3: a = (n + 19) / 4; break;
        case 4: a = (n + 15) / 4; break;
        case 5: a = (n + 23) / 4; break;
        default: return;
    }
    if (try_a(n, a, K_SHORT))
        found_mask[n_idx] = true;
}

// Records the winning a_off (1-indexed) for each solved n.
// On collision (multiple a_off work) keeps the smallest.
extern "C" __global__
void find_aoff_kernel(const long long* hard_n, int* aoff_solved,
                      int num_hard, int A_RANGE, int K_RANGE) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)num_hard * A_RANGE) return;
    int n_idx = (int)(idx / A_RANGE);
    int a_off = (int)(idx % A_RANGE) + 1;
    long long n = hard_n[n_idx];
    long long a = (n / 4) + a_off;
    long long p = 4LL * a_off - (n % 4);
    if (p <= 0) return;
    unsigned long long pu = (unsigned long long)p;
    unsigned long long rem = ((unsigned long long)(n % p) * (unsigned long long)(a % p)) % pu;
    for (int k = 1; k <= K_RANGE; ++k) {
        long long d = (long long)k * p - (long long)rem;
        if (d > 0 && divides_q2(n, a, d)) {
            // Atomic min to record smallest winning a_off
            atomicMin(&aoff_solved[n_idx], a_off);
            return;
        }
    }
}
"""

MODULE = cp.RawModule(code=KERNEL_CODE)
SC_KERNEL = MODULE.get_function('check_shortcuts_kernel')
AOFF_KERNEL = MODULE.get_function('find_aoff_kernel')


def run_profile():
    n_start = load_best() + 1
    if n_start < 2:
        n_start = 2
    batch_size = 200_000_000
    threads_per_block = 256

    print(f"Profiling batch starting at n={n_start:,} (size={batch_size:,})")

    # --- build hard_n ---
    n_vec = cp.arange(n_start, n_start + batch_size, dtype=cp.int64)
    r_vec = (n_vec % 840).astype(cp.int32)
    is_easy_gpu = cp.array(IS_EASY_TABLE)
    mask_hard = ~is_easy_gpu[r_vec]
    hard_n_gpu = n_vec[cp.where(mask_hard)[0]]
    num_hard = int(len(hard_n_gpu))
    print(f"  Total hard_n: {num_hard:,}")

    # --- shortcuts pass ---
    found_mask = cp.zeros(num_hard, dtype=bool)
    K_SHORT = 200
    total_sc = num_hard * NUM_SHORTCUTS
    grid_sc = (total_sc + threads_per_block - 1) // threads_per_block
    t0 = time.time()
    SC_KERNEL((grid_sc,), (threads_per_block,),
              (hard_n_gpu, found_mask, np.int32(num_hard), np.int32(K_SHORT)))
    cp.cuda.Stream.null.synchronize()
    sc_time = time.time() - t0

    sc_resolved = int(cp.sum(found_mask))
    unsolved_mask = ~found_mask
    num_unsolved = int(cp.sum(unsolved_mask))
    print(f"  Shortcuts resolved: {sc_resolved:,} / {num_hard:,} ({100*sc_resolved/num_hard:.3f}%)")
    print(f"  Unsolved after shortcuts: {num_unsolved:,} ({100*num_unsolved/num_hard:.3f}%)  [{sc_time:.2f}s]")

    # Pull unsolved n to CPU
    unsolved_idx = cp.where(unsolved_mask)[0]
    unsolved_n = hard_n_gpu[unsolved_idx].get()   # numpy array on CPU

    # --- find winning a_off for each unsolved n via GPU ---
    unsolved_gpu = cp.array(unsolved_n, dtype=cp.int64)
    INT_MAX = np.int32(2_000_000_000)
    aoff_solved_gpu = cp.full(num_unsolved, INT_MAX, dtype=cp.int32)

    A_RANGE, K_RANGE = 2000, 1000
    total_main = num_unsolved * A_RANGE
    grid_main = (total_main + threads_per_block - 1) // threads_per_block
    t1 = time.time()
    AOFF_KERNEL((grid_main,), (threads_per_block,),
                (unsolved_gpu, aoff_solved_gpu, np.int32(num_unsolved),
                 np.int32(A_RANGE), np.int32(K_RANGE)))
    cp.cuda.Stream.null.synchronize()
    deep_time = time.time() - t1

    aoff_solved = aoff_solved_gpu.get()   # numpy, INT_MAX means not found in A=2000
    found_in_deep = aoff_solved < INT_MAX
    not_found = ~found_in_deep
    print(f"  Deep kernel (A={A_RANGE}, K={K_RANGE}) found: {found_in_deep.sum():,} / {num_unsolved:,}  [{deep_time:.2f}s]")
    if not_found.sum():
        print(f"  Still unsolved (need wider search): {not_found.sum():,}")

    # Work only on the ones the deep kernel solved (has valid a_off)
    n_profiled = unsolved_n[found_in_deep]
    aoff_profiled = aoff_solved[found_in_deep]

    # ================================================================
    # ANALYSIS
    # ================================================================
    print(f"\n{'='*60}")
    print(f"RESIDUE ANALYSIS of {len(n_profiled):,} unsolved-by-shortcuts n")
    print(f"{'='*60}")

    # 1. Residue mod 840 (which of the 6 hard classes)
    r840 = n_profiled % 840
    r840_counts = Counter(r840.tolist())
    print(f"\n--- mod 840 ---")
    for r in sorted(r840_counts):
        cnt = r840_counts[r]
        pct = 100 * cnt / len(n_profiled)
        print(f"  r={r:4d}: {cnt:6,}  ({pct:.2f}%)")

    # Expected uniform share per residue class
    hard_res_list = sorted(HARD_RESIDUES)
    total_hard_n_per_class = {r: int(cp.sum(hard_n_gpu % 840 == r)) for r in hard_res_list}
    print(f"\n  Context: hard_n per residue class in this batch:")
    for r in hard_res_list:
        cnt_hard = total_hard_n_per_class[r]
        cnt_unsolved = r840_counts.get(r, 0)
        pct_unsolved = 100 * cnt_unsolved / cnt_hard if cnt_hard > 0 else 0
        print(f"  r={r:4d}: {cnt_hard:8,} hard_n, {cnt_unsolved:6,} unsolved ({pct_unsolved:.3f}% miss rate)")

    # 2. Residues mod small primes
    print(f"\n--- mod small primes/composites (looking for over/under-representation) ---")
    for m in [3, 4, 5, 7, 8, 11, 12, 13, 17, 19, 23, 24, 35, 40, 60, 120]:
        r_all = unsolved_n % m          # all unsolved
        r_cnt = Counter(r_all.tolist())
        # Find the most over-represented residue vs uniform
        expected = len(unsolved_n) / m
        deviations = {r: (cnt - expected) / expected for r, cnt in r_cnt.items()}
        max_dev_r = max(deviations, key=lambda x: abs(deviations[x]))
        max_dev = deviations[max_dev_r]
        # Only print if there's a notable pattern (>5% deviation)
        if abs(max_dev) > 0.05:
            # Print full breakdown
            print(f"\n  mod {m}:")
            for r in sorted(r_cnt):
                cnt = r_cnt[r]
                dev = (cnt - expected) / expected
                bar = '+' if dev > 0.1 else ('-' if dev < -0.1 else ' ')
                print(f"    r={r:3d}: {cnt:6,}  (expected {expected:6.0f}, dev={dev:+.3f}) {bar}")

    # 3. a_off distribution
    print(f"\n--- Winning a_off distribution ---")
    aoff_bins = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100, 200, 500, 1000, 2001]
    prev = 0
    for hi in aoff_bins:
        cnt = int(((aoff_profiled > prev) & (aoff_profiled <= hi)).sum())
        pct = 100 * cnt / len(aoff_profiled) if len(aoff_profiled) else 0
        label = f"a_off {prev+1:4d}..{hi:4d}"
        print(f"  {label}: {cnt:6,}  ({pct:.2f}%)")
        prev = hi

    # 4. For small a_off winners: what mod relationships hold?
    print(f"\n--- For a_off <= 10 winners: their residues mod 5, 7, 11, 13 ---")
    small_aoff_mask = aoff_profiled <= 10
    if small_aoff_mask.sum() > 0:
        n_small = n_profiled[small_aoff_mask]
        a_small = aoff_profiled[small_aoff_mask]
        print(f"  Count: {len(n_small):,}")
        for aoff_val in range(1, 11):
            mask_a = a_small == aoff_val
            if mask_a.sum() == 0:
                continue
            n_this = n_small[mask_a]
            print(f"  a_off={aoff_val} ({mask_a.sum():,} cases):", end="")
            for m in [5, 7, 11, 13]:
                r_cnt = Counter((n_this % m).tolist())
                top = sorted(r_cnt.items(), key=lambda x: -x[1])[:3]
                top_str = ", ".join(f"r={r}:{c}" for r, c in top)
                print(f"  mod{m}=[{top_str}]", end="")
            print()

    # 5. Cross-tab: residue-840 x a_off range
    print(f"\n--- Cross-tab: residue mod 840 vs a_off range ---")
    aoff_ranges = [(1, 10), (11, 50), (51, 200), (201, 1000), (1001, 2000)]
    header = f"  {'r840':>6}" + "".join(f"  {lo}-{hi:>4}" for lo, hi in aoff_ranges)
    print(header)
    for r in hard_res_list:
        mask_r = r840 == r
        n_r = n_profiled[mask_r]
        a_r = aoff_profiled[mask_r]
        row = f"  {r:6d}"
        for lo, hi in aoff_ranges:
            cnt = int(((a_r >= lo) & (a_r <= hi)).sum())
            row += f"  {cnt:6,}"
        print(row)

    # 6. Check specific algebraic candidates for new shortcuts
    print(f"\n--- Candidate new shortcuts: does a specific a_off dominate any residue class? ---")
    for r in hard_res_list:
        mask_r = r840 == r
        if mask_r.sum() == 0:
            continue
        a_r = aoff_profiled[mask_r]
        top_aoffs = Counter(a_r.tolist()).most_common(5)
        total_r = mask_r.sum()
        print(f"  r={r}: top a_off = " +
              ", ".join(f"{a}:{c}({100*c/total_r:.1f}%)" for a, c in top_aoffs))

    # 7. Modular conditions for promising a_off values
    print(f"\n--- For most common a_off values: what are the mod conditions? ---")
    all_top = Counter(aoff_profiled.tolist()).most_common(15)
    for a_val, cnt in all_top:
        mask = aoff_profiled == a_val
        n_sel = n_profiled[mask]
        pct = 100 * cnt / len(n_profiled)
        cond_parts = []
        for m in [3, 5, 7, 8, 11, 13, 17, 19, 23]:
            r_cnt = Counter((n_sel % m).tolist())
            total = len(n_sel)
            dominant = [(r, c) for r, c in r_cnt.items() if c / total > 0.6]
            if dominant:
                cond_parts.append(f"n%{m}={dominant[0][0]}")
        cond_str = " AND ".join(cond_parts) if cond_parts else "(no strong condition)"
        print(f"  a_off={a_val:4d}: {cnt:5,} cases ({pct:.2f}%) | {cond_str}")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    run_profile()

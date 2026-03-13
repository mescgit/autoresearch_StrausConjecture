# -*- coding: utf-8 -*-
"""
profile_k.py - For each deep-kernel case, find the winning (a_off, k) pair.
This tells us: for current shortcuts (a_off=1..6) that "miss" at K=200 but
succeed at K=1000, what k do they actually need? And for the top missing
a_off values (7,8,9,10), do they work within K=200?
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import cupy as cp
from collections import Counter
from prepare import load_best

HARD_RESIDUES = {1, 121, 169, 289, 361, 529}
def get_is_easy_table():
    t = np.ones(840, dtype=bool)
    for r in HARD_RESIDUES: t[r] = False
    return t
IS_EASY_TABLE = get_is_easy_table()

KERNEL_CODE = r"""
__device__ __forceinline__ bool divides_q2(long long n, long long a, long long d) {
    unsigned long long nd = (unsigned long long)(n % d);
    unsigned long long ad = (unsigned long long)(a % d);
    unsigned long long qd = (nd * ad) % (unsigned long long)d;
    return (qd * qd) % (unsigned long long)d == 0;
}

// Shortcut kernel identical to main search
__device__ bool try_a(long long n, long long a, int K) {
    if (a <= n / 4) return false;
    long long p = 4LL * a - n;
    if (p <= 0) return false;
    unsigned long long pu = (unsigned long long)p;
    unsigned long long rem = ((unsigned long long)(n % p) * (unsigned long long)(a % p)) % pu;
    for (int k = 1; k <= K; ++k) {
        long long d = (long long)k * p - (long long)rem;
        if (d > 0 && divides_q2(n, a, d)) return true;
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
    if (try_a(n, a, K_SHORT)) found_mask[n_idx] = true;
}

// Records (winning_a_off * 100000 + winning_k) packed into one int.
// Tries a_off 1..A_RANGE and records the FIRST (a_off, k) that works.
// Uses global atomicMin on (a_off * 100000 + k) to get the smallest a_off then k.
extern "C" __global__
void find_aoff_k_kernel(const long long* hard_n, int* best_aoff_k,
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
            int packed = a_off * 10001 + k;   // encodes (a_off, k), a_off dominates
            atomicMin(&best_aoff_k[n_idx], packed);
            return;
        }
    }
}
"""

MODULE = cp.RawModule(code=KERNEL_CODE)
SC_KERNEL = MODULE.get_function('check_shortcuts_kernel')
AOFF_K_KERNEL = MODULE.get_function('find_aoff_k_kernel')

def run():
    n_start = load_best() + 1
    batch_size = 200_000_000
    threads = 256

    n_vec = cp.arange(n_start, n_start + batch_size, dtype=cp.int64)
    r_vec = (n_vec % 840).astype(cp.int32)
    is_easy_gpu = cp.array(IS_EASY_TABLE)
    hard_n_gpu = n_vec[cp.where(~is_easy_gpu[r_vec])[0]]
    num_hard = int(len(hard_n_gpu))

    # shortcuts pass
    found = cp.zeros(num_hard, dtype=bool)
    SC_KERNEL(((num_hard*6+threads-1)//threads,), (threads,),
              (hard_n_gpu, found, np.int32(num_hard), np.int32(200)))
    cp.cuda.Stream.null.synchronize()

    unsolved_idx = cp.where(~found)[0]
    unsolved_gpu = hard_n_gpu[unsolved_idx]
    num_unsolved = int(len(unsolved_gpu))
    print(f"Unsolved by shortcuts: {num_unsolved:,}")

    # find winning (a_off, k) for each unsolved n
    SENTINEL = np.int32(2_000_000_000)
    best = cp.full(num_unsolved, SENTINEL, dtype=cp.int32)
    A_RANGE, K_RANGE = 20, 1000   # only need a_off 1..20 for this analysis
    AOFF_K_KERNEL(((num_unsolved*A_RANGE+threads-1)//threads,), (threads,),
                  (unsolved_gpu, best, np.int32(num_unsolved),
                   np.int32(A_RANGE), np.int32(K_RANGE)))
    cp.cuda.Stream.null.synchronize()

    best_np = best.get()
    found_mask2 = best_np < SENTINEL

    packed = best_np[found_mask2]
    aoff_arr = packed // 10001
    k_arr    = packed % 10001

    print(f"Found solution (a_off<=20, k<=1000): {found_mask2.sum():,} / {num_unsolved:,}")

    # For each a_off, show k distribution
    print(f"\n--- For each winning a_off: k distribution ---")
    print(f"{'a_off':>6}  {'count':>7}  {'k<=10':>6}  {'k<=50':>6}  {'k<=100':>6}  {'k<=200':>6}  {'k<=500':>6}  {'k<=1000':>7}  {'mean_k':>7}  {'max_k':>7}")
    for aoff_val in range(1, 21):
        mask = aoff_arr == aoff_val
        if mask.sum() == 0:
            continue
        ks = k_arr[mask]
        cnt = len(ks)
        print(f"  {aoff_val:4d}  {cnt:7,}  "
              f"{(ks<=10).sum():6,}  "
              f"{(ks<=50).sum():6,}  "
              f"{(ks<=100).sum():6,}  "
              f"{(ks<=200).sum():6,}  "
              f"{(ks<=500).sum():6,}  "
              f"{(ks<=1000).sum():7,}  "
              f"{ks.mean():7.1f}  "
              f"{ks.max():7,}")

    # What K_SHORT threshold would eliminate what fraction of deep kernel cases?
    print(f"\n--- Cumulative: % of deep cases solved if K_SHORT=X for a_off=1..6+new ---")
    print(f"Assuming we also add a_off=7,8,9,10 as shortcuts.")
    for K_thresh in [100, 200, 300, 500, 750, 1000]:
        covered_in_sc = ((aoff_arr <= 10) & (k_arr <= K_thresh)).sum()
        pct = 100 * covered_in_sc / num_unsolved
        print(f"  K_SHORT={K_thresh:4d}: covers {covered_in_sc:6,} / {num_unsolved:,} = {pct:.2f}% of deep cases")

    print(f"\nDone.")

if __name__ == "__main__":
    run()

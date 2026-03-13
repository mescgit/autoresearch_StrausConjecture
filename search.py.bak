# -*- coding: utf-8 -*-
"""
Erdos-Straus Conjecture Search - Supercharged GPU (Robust)
==========================================================
Key findings from profiling:
  - All hard n == 1 mod 24 (== 1 mod 8, 1 mod 3)
  - Shortcuts 4&5 (large-p) were proven zero-contribution:
      a=(n+1)/2 -> Q is always ODD -> d=n+1 is even -> can never divide Q^2
      a=(n+2)/3 -> Q is always ODD, d values ~k*n/3 too large
  - Replaced with a_off=4 (p=15) and a_off=6 (p=23) - even a_off -> Q even -> 4|Q^2
  - Misses are FLAT across all 6 residue classes and all a_off 1..2000

Critical arithmetic optimization (the real speedup):
  In the deep kernel: p = 4*a_off - 1 <= 7999, d = k*p - rem <= K*7999 < 8M.
  So n%d < 8M, a%d < 8M, product < 64e12 < 2^46 < uint64 max.
  Replace slow mul_mod (64-iteration loop, ~256 ops) with direct uint64 multiply.
  Same for shortcuts: all small-a_off choices have p <= 23, d <= 4600.
  Expected speedup: ~30-50x on inner loop -> ~3-4x overall batch speedup.

Timing from profiling (before this optimization):
  batch setup: 25ms, shortcuts: 901ms (16%), deep kernel: 4736ms (84%) = 5662ms total
"""

import time
import cupy as cp
import numpy as np
from prepare import TIME_BUDGET_SECONDS, log_result, load_best

HARD_RESIDUES = {1, 121, 169, 289, 361, 529}

def get_is_easy_table():
    is_easy = np.ones(840, dtype=bool)
    for r in HARD_RESIDUES:
        is_easy[r] = False
    return is_easy

IS_EASY_TABLE = get_is_easy_table()
NUM_SHORTCUTS = 10   # 6 original + 4 new (a_off=7,8,9,10)

# KERNEL_CODE must be ASCII-only: cupy writes it via cp1252 on Windows.
#
# OVERFLOW SAFETY for direct uint64 arithmetic (no mul_mod loop needed):
#   Shortcuts (10 choices, a_off<=10): p<=39, d_max = K*39.
#     Phase1 K=200: d_max=7800, product<7800^2=60.8M<2^26<2^64. SAFE.
#     Phase2 K=1000: d_max=39000, product<39000^2=1.52B<2^31<2^64. SAFE.
#   Deep kernel: p <= 4*2000-1 = 7999.  d_max = 1000*7999 = 7,999,000 < 2^23.
#     n%d, a%d < d < 8M.  (n%d * a%d) < 64e12 < 2^46 < 2^64.  SAFE.
#   Wide kernel: p <= 4*10000-1 = 39999.  d_max = 2000*39999 ~= 80M < 2^27.
#     n%d, a%d < d < 80M.  (n%d * a%d) < 6.4e15 < 2^53 < 2^64.  SAFE.
#
# TWO-PHASE SHORTCUT STRATEGY (from profiling):
#   For existing a_off=1..6: deep kernel "misses" need k=201..1000 (mean k~495).
#   For new a_off=7..10: mean k=90-137, K=200 catches 76-86% immediately.
#   Phase1 (K=200, 10 choices, all hard_n): catches 98.6% of hard_n.
#   Phase2 (K=1000, 10 choices, ~19K survivors): catches 87.5% of those.
#   Deep kernel sees only ~2,400 cases (vs 27,676 before) -> 91% reduction.
KERNEL_CODE = r"""
// Fast divisor check: no mul_mod loop needed when d is small enough.
// Safe when: (n%d) * (a%d) < 2^64  (guaranteed for all callers below).
// Ultra-wide: p<=159999, d_max=20000*159999~=3.2e9. (n%d)*(a%d)<(3.2e9)^2=1.02e19 < 2^64. SAFE.
__device__ __forceinline__ bool divides_q2(long long n, long long a, long long d) {
    unsigned long long nd = (unsigned long long)(n % d);
    unsigned long long ad = (unsigned long long)(a % d);
    unsigned long long qd = (nd * ad) % (unsigned long long)d;
    return (qd * qd) % (unsigned long long)d == 0;
}

// Check whether 4/n = 1/a + 1/b + 1/c is solvable for fixed 'a'.
// Assumes p = 4a-n is small (p <= 40000) so direct uint64 arithmetic is safe.
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

// 10 shortcuts per hard_n.  All hard n == 1 mod 24 (== 1 mod 8, 1 mod 3).
// Choices 0-3: original odd a_off.
// Choices 4-5: original even a_off.
// Choices 6-9: NEW a_off=7..10.  Profiling: these cover 36% of deep-kernel survivors
//   with K=200, and the rest at K=1000 in Phase2 (mean k=90-137 vs ~495 for a_off=1..6).
//   Overflow: p<=39, K=1000: d_max=39000, product<39000^2=1.52B<2^31<2^64. SAFE.
extern "C" __global__
void check_shortcuts_kernel(const long long* hard_n, bool* found_mask,
                             int num_hard, int K_SHORT, int NUM_SC) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)num_hard * NUM_SC) return;
    int n_idx = (int)(idx / NUM_SC);
    if (found_mask[n_idx]) return;
    long long n = hard_n[n_idx];
    int choice = (int)(idx % NUM_SC);
    long long a;
    switch (choice) {
        case 0: a = (n + 3)  / 4; break;  // a_off=1, p=3
        case 1: a = (n + 7)  / 4; break;  // a_off=2, p=7
        case 2: a = (n + 11) / 4; break;  // a_off=3, p=11; n==4mod5: 5|a, 25|Q^2
        case 3: a = (n + 19) / 4; break;  // a_off=5, p=19; n==1mod5: 5|a, 25|Q^2
        case 4: a = (n + 15) / 4; break;  // a_off=4, p=15, even a_off
        case 5: a = (n + 23) / 4; break;  // a_off=6, p=23, even a_off
        case 6: a = (n + 27) / 4; break;  // a_off=7, p=27=3^3; 8.62% of deep cases, mean k=137
        case 7: a = (n + 31) / 4; break;  // a_off=8, p=31 (prime); TOP: 16.28%, mean k=90
        case 8: a = (n + 35) / 4; break;  // a_off=9, p=35=5*7; 5.38%, mean k=148
        case 9: a = (n + 39) / 4; break;  // a_off=10, p=39=3*13; 6.16%, mean k=108
        default: return;
    }
    if (try_a(n, a, K_SHORT))
        found_mask[n_idx] = true;
}

extern "C" __global__
void check_n_kernel(const long long* hard_n, bool* found_mask,
                    int num_hard, int A_RANGE, int K_RANGE) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)num_hard * A_RANGE) return;
    int n_idx = (int)(idx / A_RANGE);
    if (found_mask[n_idx]) return;
    int a_off = (int)(idx % A_RANGE) + 1;
    long long n = hard_n[n_idx];
    long long a = (n / 4) + a_off;
    long long p = 4LL * a_off - (n % 4);
    if (p <= 0) return;
    unsigned long long pu = (unsigned long long)p;
    // rem = Q mod p, computed without mul_mod (p <= 4*A_RANGE <= 40000, safe)
    unsigned long long rem = ((unsigned long long)(n % p) * (unsigned long long)(a % p)) % pu;
    for (int k = 1; k <= K_RANGE; ++k) {
        long long d = (long long)k * p - (long long)rem;
        if (d > 0 && divides_q2(n, a, d)) {
            found_mask[n_idx] = true;
            return;
        }
    }
}

// Ultra-wide kernel: replaces CPU fallback for rare stragglers.
// Covers a_off in [A_LO+1, A_LO+a_width] with K_RANGE iterations.
// Overflow safety: p<=159999, d_max=20000*159999~=3.2e9. (n%d)*(a%d)<1.02e19<2^64. SAFE.
extern "C" __global__
void check_n_ultrawide_kernel(const long long* hard_n, bool* found_mask,
                               int num_hard, int A_LO, int A_WIDTH, int K_RANGE) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)num_hard * A_WIDTH) return;
    int n_idx = (int)(idx / A_WIDTH);
    if (found_mask[n_idx]) return;
    int a_off = A_LO + (int)(idx % A_WIDTH) + 1;
    long long n = hard_n[n_idx];
    long long a = (n / 4) + a_off;
    long long p = 4LL * a_off - (n % 4);
    if (p <= 0) return;
    unsigned long long pu = (unsigned long long)p;
    unsigned long long rem = ((unsigned long long)(n % p) * (unsigned long long)(a % p)) % pu;
    for (int k = 1; k <= K_RANGE; ++k) {
        long long d = (long long)k * p - (long long)rem;
        if (d > 0 && divides_q2(n, a, d)) {
            found_mask[n_idx] = true;
            return;
        }
    }
}
"""

MODULE = cp.RawModule(code=KERNEL_CODE)
CHECK_SHORTCUTS_KERNEL = MODULE.get_function('check_shortcuts_kernel')
CHECK_N_KERNEL = MODULE.get_function('check_n_kernel')
CHECK_N_ULTRAWIDE_KERNEL = MODULE.get_function('check_n_ultrawide_kernel')

def find_representation_thorough_cpu(n_val, a_range=20000, k_range=10000):
    """Deep fallback on CPU using pure Python arbitrary precision."""
    n = int(n_val)
    for a in [(n+3)//4, (n+7)//4, (n+11)//4, (n+19)//4, (n+15)//4, (n+23)//4]:
        if a <= n//4: continue
        p = 4*a - n
        if p <= 0: continue
        q = n*a
        rem = q % p
        for k in range(1, 2000):
            d = k * p - rem
            if d > 0 and q**2 % d == 0:
                b = (q + d) // p
                c = (q + q**2 // d) // p
                if b > 0 and c > 0 and p*b*c == q*(b + c):
                    return True
    a_min = n // 4 + 1
    for a_offset in range(1, a_range + 1):
        a = a_min + a_offset
        p = 4 * a - n
        if p <= 0: continue
        q = n * a
        rem = q % p
        for k in range(1, k_range + 1):
            d = k * p - rem
            if d > 0 and q**2 % d == 0:
                b = (q + d) // p
                c = (q + q**2 // d) // p
                if b > 0 and c > 0 and p*b*c == q*(b + c):
                    return True
    return False

_batch_count = 0

def verify_batch_gpu(n_start, batch_size):
    global _batch_count
    _batch_count += 1
    do_diag = (_batch_count <= 2) or (_batch_count % 15 == 0)

    n_vec = cp.arange(n_start, n_start + batch_size, dtype=cp.int64)
    r_vec = (n_vec % 840).astype(cp.int32)
    is_easy_gpu = cp.array(IS_EASY_TABLE)
    mask_hard = ~is_easy_gpu[r_vec]

    hard_indices = cp.where(mask_hard)[0]
    if len(hard_indices) == 0:
        return n_start + batch_size - 1, None

    hard_n = n_vec[hard_indices]
    num_hard = int(len(hard_n))
    found_mask = cp.zeros(num_hard, dtype=bool)
    threads_per_block = 256

    # 1a. Phase1 shortcuts: 10 choices x K=200 on all hard_n.
    #     Catches ~98.6% of hard_n (vs 98.06% with 6 shortcuts).
    K_SHORT1 = 200
    total_sc = num_hard * NUM_SHORTCUTS
    grid_sc = (total_sc + threads_per_block - 1) // threads_per_block
    CHECK_SHORTCUTS_KERNEL((grid_sc,), (threads_per_block,),
                           (hard_n, found_mask, np.int32(num_hard),
                            np.int32(K_SHORT1), np.int32(NUM_SHORTCUTS)))

    # 1b. Phase2 shortcuts: same 10 choices x K=1000 on survivors only.
    #     Catches ~87.5% of Phase1 survivors (profiling: a_off=1..6 need k=201..1000,
    #     a_off=7..10 remaining need k=201..1000 too). Expected ~99.87% total coverage.
    unsolved_ph2 = cp.where(~found_mask)[0]
    num_unsolved_ph2 = int(len(unsolved_ph2))
    if num_unsolved_ph2 > 0:
        hard_n_ph2 = hard_n[unsolved_ph2]
        found_mask_ph2 = cp.zeros(num_unsolved_ph2, dtype=bool)
        K_SHORT2 = 1000
        total_sc2 = num_unsolved_ph2 * NUM_SHORTCUTS
        grid_sc2 = (total_sc2 + threads_per_block - 1) // threads_per_block
        CHECK_SHORTCUTS_KERNEL((grid_sc2,), (threads_per_block,),
                               (hard_n_ph2, found_mask_ph2, np.int32(num_unsolved_ph2),
                                np.int32(K_SHORT2), np.int32(NUM_SHORTCUTS)))
        ph2_solved = int(cp.sum(found_mask_ph2))
        if ph2_solved > 0:
            found_mask[unsolved_ph2[cp.where(found_mask_ph2)[0]]] = True

    sc_resolved = int(cp.sum(found_mask))
    if do_diag:
        pct = 100.0 * sc_resolved / num_hard
        print(f"  [diag] shortcuts={pct:.2f}% ({sc_resolved}/{num_hard}), "
              f"deep handling {num_hard - sc_resolved} hard_n")

    if sc_resolved == num_hard:
        return n_start + batch_size - 1, None

    # 2. Main kernel - only on Phase2 survivors (expected ~2,400 cases vs 27,676 before).
    unsolved1 = cp.where(~found_mask)[0]
    num_unsolved1 = int(len(unsolved1))
    unsolved_n1 = hard_n[unsolved1]
    found_mask_deep = cp.zeros(num_unsolved1, dtype=bool)

    A_RANGE, K_RANGE = 2000, 1000
    total_main = num_unsolved1 * A_RANGE
    grid_main = (total_main + threads_per_block - 1) // threads_per_block
    CHECK_N_KERNEL((grid_main,), (threads_per_block,),
                   (unsolved_n1, found_mask_deep, np.int32(num_unsolved1),
                    np.int32(A_RANGE), np.int32(K_RANGE)))

    deep_solved = int(cp.sum(found_mask_deep))
    if do_diag:
        pct_d = 100.0 * deep_solved / num_unsolved1 if num_unsolved1 else 100.0
        print(f"  [diag] deep kernel: {pct_d:.2f}% of {num_unsolved1} cases")

    if deep_solved > 0:
        found_mask[unsolved1[cp.where(found_mask_deep)[0]]] = True

    if cp.all(found_mask):
        return n_start + batch_size - 1, None

    # 3. Wide kernel for rare stragglers
    unsolved2 = cp.where(~found_mask)[0]
    num_unsolved2 = int(len(unsolved2))
    unsolved_n2 = hard_n[unsolved2]
    found_mask_wide = cp.zeros(num_unsolved2, dtype=bool)

    A_RANGE_WIDE, K_RANGE_WIDE = 10000, 2000
    total_wide = num_unsolved2 * A_RANGE_WIDE
    grid_wide = (total_wide + threads_per_block - 1) // threads_per_block
    CHECK_N_KERNEL((grid_wide,), (threads_per_block,),
                   (unsolved_n2, found_mask_wide, np.int32(num_unsolved2),
                    np.int32(A_RANGE_WIDE), np.int32(K_RANGE_WIDE)))

    wide_solved = cp.where(found_mask_wide)[0]
    if len(wide_solved) > 0:
        found_mask[unsolved2[wide_solved]] = True

    if cp.all(found_mask):
        return n_start + batch_size - 1, None

    # 4. GPU ultra-wide kernel: a_off 10001..40000, K=20000.
    #    Replaces CPU fallback; safe since d_max=20000*159999~3.2e9, product<1.02e19<2^64.
    unsolved3 = cp.where(~found_mask)[0]
    num_unsolved3 = int(len(unsolved3))
    unsolved_n3 = hard_n[unsolved3]
    found_mask_ultra = cp.zeros(num_unsolved3, dtype=bool)

    A_LO_ULTRA, A_WIDTH_ULTRA, K_RANGE_ULTRA = 10000, 30000, 20000
    total_ultra = num_unsolved3 * A_WIDTH_ULTRA
    grid_ultra = (total_ultra + threads_per_block - 1) // threads_per_block
    if do_diag:
        print(f"  [diag] ultra-wide kernel: {num_unsolved3} stragglers")
    CHECK_N_ULTRAWIDE_KERNEL((grid_ultra,), (threads_per_block,),
                             (unsolved_n3, found_mask_ultra, np.int32(num_unsolved3),
                              np.int32(A_LO_ULTRA), np.int32(A_WIDTH_ULTRA), np.int32(K_RANGE_ULTRA)))

    ultra_solved = cp.where(found_mask_ultra)[0]
    if len(ultra_solved) > 0:
        found_mask[unsolved3[ultra_solved]] = True

    if cp.all(found_mask):
        return n_start + batch_size - 1, None

    # 5. CPU last-resort (should rarely/never trigger with ultra-wide GPU above)
    for idx in cp.where(~found_mask)[0]:
        n_val = int(hard_n[idx])
        print(f"GPU ultra-wide exhausted for n={n_val}, verifying on CPU...")
        if not find_representation_thorough_cpu(n_val, a_range=100000, k_range=50000):
            print(f"!!! COUNTEREXAMPLE VERIFIED at n={n_val} !!!")
            return n_val - 1, n_val
    return n_start + batch_size - 1, None

def run_search():
    start_time = time.time()
    deadline = start_time + TIME_BUDGET_SECONDS - 5

    current_n = load_best() + 1
    if current_n < 2: current_n = 2

    start_n_val = current_n
    max_verified = current_n - 1
    print(f"Starting search from n={current_n:,}...")

    batch_size = 200_000_000

    while time.time() < deadline:
        batch_max, counterexample = verify_batch_gpu(current_n, batch_size)

        if counterexample:
            log_result(batch_max, counterexample=counterexample)
            return batch_max

        max_verified = batch_max
        current_n = batch_max + 1

        elapsed = time.time() - start_time
        verified_this_run = max_verified - start_n_val + 1
        rate = verified_this_run / elapsed if elapsed > 0 else 0
        print(f"n={max_verified:,}, elapsed={elapsed:.1f}s, rate={rate:,.0f} n/s")

    log_result(max_verified)
    return max_verified

if __name__ == "__main__":
    run_search()

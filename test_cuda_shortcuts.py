import cupy as cp
import numpy as np

KERNEL_CODE = r'''
__device__ unsigned long long mul_mod(unsigned long long a, unsigned long long b, unsigned long long m) {
    unsigned long long res = 0;
    a %= m;
    while (b > 0) {
        if (b & 1) res = (res + a) % m;
        a = (a * 2) % m;
        b >>= 1;
    }
    return res;
}

extern "C" __global__
void check_shortcuts_kernel(const long long* hard_n, bool* found_mask, int num_hard) {
    int idx = blockIdx.x * (int)blockDim.x + threadIdx.x;
    if (idx >= num_hard * 3) return;
    
    int n_idx = idx / 3;
    if (found_mask[n_idx]) return;

    long long n = hard_n[n_idx];
    int a_choice = idx % 3;
    long long a;
    if (a_choice == 0) a = (n + 3) / 4;
    else if (a_choice == 1) a = (n + 1) / 2;
    else a = n;

    if (a <= n / 4) return;

    long long p = 4LL * a - n;
    if (p <= 0) return;

    long long rem = mul_mod(n % p, a % p, p);

    for (int k = 1; k <= 100; ++k) {
        long long d = (long long)k * p - rem;
        if (d > 0) {
            unsigned long long q_mod_d = mul_mod(n % d, a % d, d);
            if (mul_mod(q_mod_d, q_mod_d, d) == 0) {
                found_mask[n_idx] = true;
                return;
            }
        }
    }
}
'''

m = cp.RawModule(code=KERNEL_CODE)
check = m.get_function('check_shortcuts_kernel')

hard_n = cp.arange(6_634_591_022_441, 6_634_591_022_441 + 1_428_571, dtype=cp.int64)
mask = cp.zeros(len(hard_n), dtype=bool)

import time
t0 = time.time()
check((len(hard_n)//256 + 1,), (256,), (hard_n, mask, len(hard_n)))
cp.cuda.Stream.null.synchronize()
t1 = time.time()

print(f"CUDA time: {t1-t0:.4f}s, found: {cp.sum(mask)}")

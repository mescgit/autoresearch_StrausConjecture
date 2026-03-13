import time
import numpy as np

def test():
    hard_n_cpu = np.arange(6_634_591_022_441, 6_634_591_022_441 + 200_000_000, 840)
    print(f"Items: {len(hard_n_cpu)}")
    t0 = time.time()
    count = 0
    for n in hard_n_cpu:
        n = int(n)
        for a in [(n+3)//4, (n+1)//2, n]:
            if a <= n//4: continue
            p = 4*a - n
            q = n*a
            rem = q % p
            for k in range(1, 100):
                d = k * p - rem
                if d > 0 and q**2 % d == 0:
                    count += 1
                    break
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, found: {count}")

test()

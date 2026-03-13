import cupy as cp
m = cp.RawModule(code='extern "C" __global__ void f(long long* out) { unsigned __int128 x = 1; out[0] = (long long)x; }')
f=m.get_function('f')
out=cp.zeros(1, dtype=cp.int64)
f((1,), (1,), (out,))
print('ok')
# 11-18: KV cache formats vs exact (fp32) attention output, CPU simulation (numpy).
# Same synthetic set-up as tools/bench_flash_i8.cpp: K, V ~ N(0, 1) (optionally 8 outlier key channels x12),
# q ~ N(0, 0.15), head_dim 256, 4096 keys, 64 queries, softmax scale 1/16.  Error = rms(O - O_exact) / rms(O_exact).
import numpy as np

def e4m3(x):
    a = np.abs(x)
    e = np.floor(np.log2(np.maximum(a, 1e-30)))
    e = np.clip(e, -6, 8)
    step = 2.0 ** (e - 3)
    y = np.round(a / step) * step
    y = np.minimum(y, 448.0)
    return np.sign(x) * y

def int8_row(x):                      # one scale per key (row): max|x| / 127
    s = np.abs(x).max(axis=-1, keepdims=True) / 127.0
    s[s == 0] = 1.0
    return np.clip(np.round(x / s), -127, 127) * s

def int4_group(x, g):                 # symmetric int4 [-7, 7], one scale per g dims of a row
    sh = x.shape
    y = x.reshape(sh[0], sh[1] // g, g)
    s = np.abs(y).max(axis=-1, keepdims=True) / 7.0
    s[s == 0] = 1.0
    return (np.clip(np.round(y / s), -7, 7) * s).reshape(sh)

def int4_asym_group(x, g):            # asymmetric int4 [0, 15] with min / max per g dims
    sh = x.shape
    y = x.reshape(sh[0], sh[1] // g, g)
    lo = y.min(axis=-1, keepdims=True); hi = y.max(axis=-1, keepdims=True)
    s = (hi - lo) / 15.0
    s[s == 0] = 1.0
    return (np.clip(np.round((y - lo) / s), 0, 15) * s + lo).reshape(sh)

def attn(q, k, v):
    s = q @ k.T / 16.0
    s -= s.max(axis=-1, keepdims=True)
    p = np.exp(s); p /= p.sum(axis=-1, keepdims=True)
    return p @ v

rng = np.random.default_rng(5)
T, D, Q = 4096, 256, 64
for outl in (False, True):
    k = rng.standard_normal((T, D)).astype(np.float64)
    if outl: k[:, 7::32] *= 12.0
    v = rng.standard_normal((T, D))
    q = rng.standard_normal((Q, D)) * 0.15
    ref = attn(q, k, v)
    def err(kq, vq):
        o = attn(q, kq, vq)
        return np.sqrt(((o - ref) ** 2).mean()) / np.sqrt((ref ** 2).mean())
    rows = [
        ("FP8 K + FP8 V (today)", e4m3(k), e4m3(v)),
        ("INT8 K + FP8 V (built, GRIMOIRE_KV_K8)", int8_row(k), e4m3(v)),
        ("INT8 K + INT4 V, groups of 32", int8_row(k), int4_group(v, 32)),
        ("INT8 K + INT4 V asym, groups of 32", int8_row(k), int4_asym_group(v, 32)),
        ("INT8 K + INT4 V, groups of 64", int8_row(k), int4_group(v, 64)),
        ("FP8 K + INT4 V, groups of 32", e4m3(k), int4_group(v, 32)),
        ("INT4 K + INT4 V, groups of 32", int4_group(k, 32), int4_group(v, 32)),
    ]
    print("== keys:", "8 outlier channels x12" if outl else "plain normal")
    for nm, kq, vq in rows:
        print("  %-42s rms error %.2f%%" % (nm, 100 * err(kq, vq)))

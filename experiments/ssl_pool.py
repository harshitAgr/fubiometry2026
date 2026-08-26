"""Pure-numpy perceptual hashing + dedup/leakage filtering for the SSL unlabeled pool.

No PIL/torch import: tested under the project .venv on numpy arrays. The prep script decodes
images (PIL) and passes 8x8 grayscale arrays here. Leakage hygiene is content-hash based because
the large unlabeled pools use flat sequential filenames with no clip structure.
"""
from __future__ import annotations
import numpy as np


def ahash(gray8x8) -> int:
    """64-bit average hash: bit i set iff pixel i > mean. Input: 8x8 numeric array."""
    a = np.asarray(gray8x8, dtype=np.float64)
    bits = (a > a.mean()).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _bucket(h: int, bits: int = 16) -> int:
    return h >> (64 - bits)


def dedup(hashes, thresh: int = 3, bucket_bits: int = 16):
    """Greedy near-dup removal. Returns sorted indices to KEEP (first representative per cluster).
    Buckets by the top bucket_bits so comparisons stay local (scales to ~200k)."""
    kept, buckets = [], {}
    for i, h in enumerate(hashes):
        b = _bucket(h, bucket_bits)
        if any(hamming(h, hashes[j]) <= thresh for j in buckets.get(b, ())):
            continue
        kept.append(i)
        buckets.setdefault(b, []).append(i)
    return sorted(kept)


_POPCOUNT_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def exclude_near(pool_hashes, ref_hashes, thresh: int = 3, bucket_bits: int = 16):
    """Return sorted indices of pool_hashes NOT within `thresh` Hamming of ANY ref hash.
    Exact (no bucketing): vectorized popcount over the full reference set. `bucket_bits` is
    accepted for backward-compat but ignored. The reference set is small, so this is cheap."""
    pool = np.asarray(list(pool_hashes), dtype=np.uint64)
    refs = np.asarray(list(ref_hashes), dtype=np.uint64)
    if pool.size == 0:
        return []
    if refs.size == 0:
        return list(range(pool.size))
    keep = np.ones(pool.size, dtype=bool)
    for r in refs:                                   # ~7k refs; each step vectorized over the pool
        x = (pool ^ np.uint64(r)).view(np.uint8).reshape(-1, 8)
        dist = _POPCOUNT_LUT[x].sum(axis=1)
        keep &= dist > thresh
        if not keep.any():
            break
    return [int(i) for i in np.nonzero(keep)[0]]

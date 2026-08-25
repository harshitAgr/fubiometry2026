import numpy as np
from experiments import ssl_pool as P


def test_ahash_is_64_bits_and_deterministic():
    a = np.arange(64, dtype=np.float64).reshape(8, 8)
    h = P.ahash(a)
    assert isinstance(h, int) and 0 <= h < (1 << 64)
    assert P.ahash(a) == h


def test_ahash_thresholds_at_mean():
    a = np.zeros((8, 8)); a[4:] = 10.0
    bits = bin(P.ahash(a)).count("1")
    assert bits == 32


def test_hamming_counts_differing_bits():
    assert P.hamming(0b1011, 0b1110) == 2
    assert P.hamming(5, 5) == 0


def test_ahash_uniform_image_is_all_zero():
    assert P.ahash(np.full((8, 8), 7.0)) == 0


def test_dedup_keeps_one_per_near_duplicate_cluster():
    h = 0b1010101010101010101010101010101010101010101010101010101010101010
    far = h ^ ((1 << 40) - 1)
    keep = P.dedup([h, h, h, far], thresh=3)
    assert sorted(keep) == [0, 3]


def test_dedup_near_within_threshold_collapses():
    h = 0
    near = 0b111
    assert P.dedup([h, near], thresh=3) == [0]
    assert sorted(P.dedup([h, near], thresh=2)) == [0, 1]


def test_exclude_near_drops_pool_matching_reference():
    pool = [0, 0b1, 0b1111111]
    refs = [0]
    kept = P.exclude_near(pool, refs, thresh=1)
    assert kept == [2]


def test_exclude_near_catches_match_across_buckets():
    # pool hash 0 and ref differ by Hamming 1 via a HIGH bit -> different top-16-bit buckets.
    # The old bucketed impl missed this (kept the pool image); exact impl must EXCLUDE it.
    pool = [0]
    ref = [1 << 63]            # hamming(0, 1<<63) == 1, but top bit -> bucket mismatch
    assert P.exclude_near(pool, ref, thresh=1) == []        # excluded (within thresh)
    assert P.exclude_near(pool, ref, thresh=0) == [0]       # kept (hamming 1 > 0)


def test_exclude_near_empty_refs_keeps_all():
    assert P.exclude_near([1, 2, 3], [], thresh=2) == [0, 1, 2]

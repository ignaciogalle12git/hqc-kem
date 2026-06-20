"""
Benchmark and correctness check for the two polynomial multiplication algorithms.

Usage:
    python3 bench_poly_mul.py          # quick benchmark (1 sample)
    python3 bench_poly_mul.py --full   # statistical benchmark (5 rounds x 3 reps)

Algorithms compared:
  naive      poly_mul()           O(n^2)    -- readable, slow
  karatsuba  poly_mul_karatsuba() O(n^1.58) -- equivalent to gf2x.c from the C reference

Reference for the Karatsuba implementation:
  PQClean/crypto_kem/hqc-128/clean/gf2x.c
  https://github.com/PQClean/PQClean
"""

import argparse
import os
import sys
import timeit

sys.path.insert(0, os.path.dirname(__file__))

from hqc.hash import XOF
from hqc.params import HQC1
from hqc.poly import poly_mul, poly_mul_karatsuba
from hqc.sampling import sample_fixed_weight_encrypt, sample_fixed_weight_keygen, sample_vect


def make_vectors(n, omega, omega_r):
    y  = sample_fixed_weight_keygen(n, omega,   XOF(os.urandom(32)))
    r2 = sample_fixed_weight_encrypt(n, omega_r, XOF(os.urandom(32)))
    h  = sample_vect(n, XOF(os.urandom(32)))
    s  = sample_vect(n, XOF(os.urandom(32)))
    return y, r2, h, s


CASES = [
    ("h*y   (keygen,  sparse x dense, w=66)",  "y",  "h"),
    ("h*r2  (encrypt, sparse x dense, w=75)",  "r2", "h"),
    ("s*r2  (encrypt, sparse x dense, w=75)",  "r2", "s"),
    ("u*y   (decrypt, sparse x dense, w=66)",  "y",  "s"),
]


def run_benchmark(full: bool):
    n, omega, omega_r = HQC1.n, HQC1.omega, HQC1.omega_r
    reps   = 3 if full else 1
    rounds = 5 if full else 1

    y, r2, h, s = make_vectors(n, omega, omega_r)
    vecs = {"y": y, "r2": r2, "h": h, "s": s}

    print(f"\n{'─'*70}")
    print(f"  HQC-1  n={n}  w={omega}  wr={omega_r}")
    print(f"  {'rounds':>6} x {'reps':>4}  ->  {'naive (ms)':>12}  {'karatsuba (ms)':>16}  {'speedup':>8}")
    print(f"{'─'*70}")

    errors = 0
    for label, va, vb in CASES:
        ref = poly_mul(vecs[va], vecs[vb], n)
        got = poly_mul_karatsuba(vecs[va], vecs[vb], n)
        if ref != got:
            print(f"  ERROR: {label}")
            errors += 1
    if errors == 0:
        print(f"  Correctness verified: naive == karatsuba on all {len(CASES)} cases")
    print(f"{'─'*70}")

    total_naive = total_karat = 0.0
    for label, va, vb in CASES:
        a, b = vecs[va], vecs[vb]

        times_naive = []
        times_karat = []
        for _ in range(rounds):
            t = timeit.timeit(lambda: poly_mul(a, b, n), number=reps) / reps
            times_naive.append(t * 1000)
            t = timeit.timeit(lambda: poly_mul_karatsuba(a, b, n), number=reps) / reps
            times_karat.append(t * 1000)

        mn = min(times_naive)
        mk = min(times_karat)
        total_naive += mn
        total_karat += mk
        print(f"  {label:<42}  {mn:>10.1f}  {mk:>14.1f}  {mn/mk:>7.1f}x")

    print(f"{'─'*70}")
    print(f"  {'TOTAL (4 distinct forms, 6 muls/KEM cycle)':>42}  {total_naive:>10.1f}  {total_karat:>14.1f}  {total_naive/total_karat:>7.1f}x")
    print(f"{'─'*70}")
    print()
    print("  Note: the C reference (PQClean gf2x.c) uses the same Karatsuba")
    print("  algorithm over uint64_t; in C the speedup over naive Python is")
    print("  ~3000x. In Python the interpreter overhead dominates and the")
    print("  real speedup is ~2-3x.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="Statistical benchmark (5 rounds x 3 reps)")
    args = parser.parse_args()
    run_benchmark(full=args.full)

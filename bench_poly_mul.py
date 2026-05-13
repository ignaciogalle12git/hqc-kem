"""
Benchmark y verificación de los dos algoritmos de multiplicación de polinomios.

Uso:
    python bench_poly_mul.py          # benchmark rápido (1 muestra)
    python bench_poly_mul.py --full   # benchmark estadístico (5 muestras x 3 reps)

Algoritmos comparados
─────────────────────
  naive      poly_mul()           O(n²)     — legible, lento
  karatsuba  poly_mul_karatsuba() O(n^1.58) — equivalente al gf2x.c de la ref C

Referencia del algoritmo Karatsuba:
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

# ─────────────────────────────────────────────────────────────────────────────
# Casos representativos de multiplicación en HQC
# ─────────────────────────────────────────────────────────────────────────────

def make_vectors(n, omega, omega_r):
    y  = sample_fixed_weight_keygen(n, omega,   XOF(os.urandom(32)))  # disperso (keygen)
    r2 = sample_fixed_weight_encrypt(n, omega_r, XOF(os.urandom(32))) # disperso (encrypt)
    h  = sample_vect(n, XOF(os.urandom(32)))                          # denso uniforme
    s  = sample_vect(n, XOF(os.urandom(32)))                          # denso (≈ x + h·y)
    return y, r2, h, s


CASES = [
    ("h·y   (keygen,   disperso×denso, ω=66)",  "y",  "h"),
    ("h·r₂  (encrypt,  disperso×denso, ω=75)",  "r2", "h"),
    ("s·r₂  (encrypt,  disperso×denso, ω=75)",  "r2", "s"),
    ("u·y   (decrypt,  disperso×denso, ω=66)",  "y",  "s"),
]

# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(full: bool):
    n, omega, omega_r = HQC1.n, HQC1.omega, HQC1.omega_r
    reps   = 3 if full else 1
    rounds = 5 if full else 1

    y, r2, h, s = make_vectors(n, omega, omega_r)
    vecs = {"y": y, "r2": r2, "h": h, "s": s}

    print(f"\n{'─'*70}")
    print(f"  HQC-1  n={n}  ω={omega}  ωᵣ={omega_r}")
    print(f"  {'rounds':>6} × {'reps':>4}  →  {'naive (ms)':>12}  {'karatsuba (ms)':>16}  {'speedup':>8}")
    print(f"{'─'*70}")

    # Verificación de corrección primero
    errors = 0
    for label, va, vb in CASES:
        ref = poly_mul(vecs[va], vecs[vb], n)
        got = poly_mul_karatsuba(vecs[va], vecs[vb], n)
        if ref != got:
            print(f"  ERROR corrección: {label}")
            errors += 1
    if errors == 0:
        print(f"  Corrección verificada: naive == karatsuba en los {len(CASES)} casos ✓")
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
        print(f"  {label:<42}  {mn:>10.1f}  {mk:>14.1f}  {mn/mk:>7.1f}×")

    print(f"{'─'*70}")
    print(f"  {'TOTAL (4 formas distintas, 6 muls/ciclo KEM)':>42}  {total_naive:>10.1f}  {total_karat:>14.1f}  {total_naive/total_karat:>7.1f}×")
    print(f"{'─'*70}")
    print()
    print("  Nota: el código C de referencia (PQClean gf2x.c) usa el mismo")
    print("  algoritmo Karatsuba sobre uint64_t; en C el speedup frente al")
    print("  naive Python es ~3000×. En Python el overhead del intérprete")
    print("  domina y el speedup real es ~2-3×.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="Benchmark estadístico (5 rondas × 3 repeticiones)")
    args = parser.parse_args()
    run_benchmark(full=args.full)

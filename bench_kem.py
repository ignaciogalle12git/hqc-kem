"""
Benchmark for the three HQC-1 KEM operations: KeyGen, Encaps, Decaps.

Usage:
    python bench_kem.py             # quick benchmark  (10 iterations)
    python bench_kem.py --full      # statistical benchmark (50 iterations)

Times are reported in milliseconds (median, min, max). The last column
converts the measured time to kcycles assuming the processor clock frequency
(--cpu-mhz, default 3000 MHz) for an indicative comparison with the kcycle
figures published in the HQC specification and NIST IR 8545.

Note: Python measurements are not directly comparable with C because they
include interpreter overhead. The goal is to quantify the order of magnitude,
not to establish absolute performance figures.

Output intended for use in:
    capitulos/07_Pruebas.tex, table tab:comparativa_tiempos
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from hqc.params import HQC1
from hqc.kem import kem_keygen, kem_encaps, kem_decaps


def _bench(fn, n_iters: int) -> list[float]:
    """Run fn() n_iters times and return elapsed times in ms."""
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times


def _stats(times: list[float]) -> tuple[float, float, float]:
    """Return (median, min, max) in ms."""
    return statistics.median(times), min(times), max(times)


def run_benchmark(n_iters: int, cpu_mhz: float):
    p = HQC1

    print(f"\n{'='*78}")
    print(f"  HQC-1  n={p.n}  k={p.k}  omega={p.omega}  omega_r={p.omega_r}")
    print(f"  iterations={n_iters}   CPU={cpu_mhz} MHz (for kcycle conversion)")
    print(f"{'='*78}")
    print(f"  {'Operation':<10} {'median (ms)':>14} {'min (ms)':>10} {'max (ms)':>10} {'~kcycles':>12}")
    print(f"  {'-'*10} {'-'*14} {'-'*10} {'-'*10} {'-'*12}")

    keygen_times = _bench(lambda: kem_keygen(p), n_iters)
    med, mn, mx = _stats(keygen_times)
    print(f"  {'KeyGen':<10} {med:>14.2f} {mn:>10.2f} {mx:>10.2f} {med*cpu_mhz:>12,.0f}")

    ek, dk = kem_keygen(p)

    encaps_times = _bench(lambda: kem_encaps(ek, p), n_iters)
    med, mn, mx = _stats(encaps_times)
    print(f"  {'Encaps':<10} {med:>14.2f} {mn:>10.2f} {mx:>10.2f} {med*cpu_mhz:>12,.0f}")

    _, ct = kem_encaps(ek, p)

    decaps_times = _bench(lambda: kem_decaps(dk, ct, p), n_iters)
    med, mn, mx = _stats(decaps_times)
    print(f"  {'Decaps':<10} {med:>14.2f} {mn:>10.2f} {mx:>10.2f} {med*cpu_mhz:>12,.0f}")

    print(f"{'='*78}")
    print()
    print("  Published reference figures for HQC-1 (kcycles):")
    print(f"  C clean (HQC spec, table 7):  KeyGen=4,557   Encaps=9,116   Decaps=13,918")
    print(f"  C avx2  (HQC spec, table 8):  KeyGen=   76   Encaps=  150   Decaps=   353")
    print(f"  NIST IR 8545:                 KeyGen=  105   Encaps=  197   Decaps=   360")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="Statistical benchmark (50 iterations, several minutes)")
    parser.add_argument("--iters", type=int, default=None,
                        help="Number of iterations (overrides --full)")
    parser.add_argument("--cpu-mhz", type=float, default=3000.0,
                        help="CPU frequency in MHz for kcycle conversion (default: 3000)")
    args = parser.parse_args()

    if args.iters is not None:
        n_iters = args.iters
    elif args.full:
        n_iters = 50
    else:
        n_iters = 10

    run_benchmark(n_iters=n_iters, cpu_mhz=args.cpu_mhz)

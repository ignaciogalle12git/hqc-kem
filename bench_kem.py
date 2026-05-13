"""
Benchmark de las tres operaciones del KEM HQC-1: KeyGen, Encaps, Decaps.

Uso:
    python bench_kem.py             # benchmark rapido  (10 iteraciones)
    python bench_kem.py --full      # benchmark estadistico (50 iteraciones)

Los tiempos se reportan en milisegundos (mediana, min, max).  La columna
final convierte el tiempo medido a kciclos asumiendo la frecuencia de
reloj del procesador (--cpu-mhz, por defecto 3000 MHz) para permitir
una comparacion orientativa con los datos publicados en kciclos por la
especificacion HQC y el informe NIST IR 8545.

Nota: las medidas en Python no son comparables 1:1 con C porque incluyen
el overhead del interprete; el objetivo es cuantificar el orden de
magnitud, no establecer rendimiento absoluto.

Salida destinada a:
    capitulos/07_Pruebas.tex, tabla tab:comparativa_tiempos
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
    """Ejecuta fn() n_iters veces y devuelve la lista de tiempos en ms."""
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return times


def _stats(times: list[float]) -> tuple[float, float, float]:
    """Devuelve (median, min, max) en ms."""
    return statistics.median(times), min(times), max(times)


def run_benchmark(n_iters: int, cpu_mhz: float):
    p = HQC1

    print(f"\n{'='*78}")
    print(f"  HQC-1  n={p.n}  k={p.k}  omega={p.omega}  omega_r={p.omega_r}")
    print(f"  iteraciones={n_iters}   CPU={cpu_mhz} MHz (para conversion a kciclos)")
    print(f"{'='*78}")
    print(f"  {'Operacion':<10} {'mediana (ms)':>14} {'min (ms)':>10} {'max (ms)':>10} {'~kciclos':>12}")
    print(f"  {'-'*10} {'-'*14} {'-'*10} {'-'*10} {'-'*12}")

    # --- KeyGen --------------------------------------------------------------
    keygen_times = _bench(lambda: kem_keygen(p), n_iters)
    med, mn, mx = _stats(keygen_times)
    kcyc_kg = med * cpu_mhz
    print(f"  {'KeyGen':<10} {med:>14.2f} {mn:>10.2f} {mx:>10.2f} {kcyc_kg:>12,.0f}")

    # Generar un par de claves de referencia para los benchmarks de encaps/decaps
    ek, dk = kem_keygen(p)

    # --- Encaps --------------------------------------------------------------
    encaps_times = _bench(lambda: kem_encaps(ek, p), n_iters)
    med, mn, mx = _stats(encaps_times)
    kcyc_en = med * cpu_mhz
    print(f"  {'Encaps':<10} {med:>14.2f} {mn:>10.2f} {mx:>10.2f} {kcyc_en:>12,.0f}")

    # Generar un ciphertext de referencia
    _, ct = kem_encaps(ek, p)

    # --- Decaps --------------------------------------------------------------
    decaps_times = _bench(lambda: kem_decaps(dk, ct, p), n_iters)
    med, mn, mx = _stats(decaps_times)
    kcyc_de = med * cpu_mhz
    print(f"  {'Decaps':<10} {med:>14.2f} {mn:>10.2f} {mx:>10.2f} {kcyc_de:>12,.0f}")

    print(f"{'='*78}")
    print()
    print("  Referencias publicadas para HQC-1 (kciclos):")
    print(f"  - C clean (HQC spec, tabla 7):   KeyGen=4,557   Encaps=9,116   Decaps=13,918")
    print(f"  - C avx2  (HQC spec, tabla 8):   KeyGen=   76   Encaps=  150   Decaps=   353")
    print(f"  - NIST IR 8545           :       KeyGen=  105   Encaps=  197   Decaps=  360")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                        help="Benchmark estadistico (50 iteraciones; ~varios minutos)")
    parser.add_argument("--iters", type=int, default=None,
                        help="Numero de iteraciones (sobreescribe --full)")
    parser.add_argument("--cpu-mhz", type=float, default=3000.0,
                        help="Frecuencia de la CPU en MHz para convertir ms a kciclos (default: 3000)")
    args = parser.parse_args()

    if args.iters is not None:
        n_iters = args.iters
    elif args.full:
        n_iters = 50
    else:
        n_iters = 10

    run_benchmark(n_iters=n_iters, cpu_mhz=args.cpu_mhz)

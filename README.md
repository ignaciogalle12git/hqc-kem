# hqc-kem

Python 3.10+ reference implementation of HQC-KEM (Hamming Quasi-Cyclic Key
Encapsulation Mechanism), a code-based post-quantum KEM selected by NIST in
2025 as a standard complementary to lattice-based schemes.

Developed as a Bachelor's Thesis (TFG) at Universidad de Murcia. The goal is
a readable, auditable implementation that maps directly to the pseudocode in
the official specification rather than competing with the C reference in
performance.

Official specification: https://pqc-hqc.org

---

## Repository layout

```
hqc/
  params.py      HQC-1 parameter set and derived sizes
  hash.py        Hash functions G, H, I, J and XOF (SHAKE256)
  sampling.py    Fixed-weight vector samplers for keygen and encrypt
  poly.py        Polynomial arithmetic: naive O(n^2) and Karatsuba O(n^1.58)
  pke.py         Underlying IND-CPA PKE scheme
  kem.py         Full IND-CCA2 KEM with Fujisaki-Okamoto transform
  rmrs.py        ctypes wrapper around the reference C decoder
  drbg.py        SHAKE256 NIST DRBG used by the KAT framework
ref/
  *.c / *.h      Reed-Muller + Reed-Solomon decoder from the HQC reference
  librmrs.so     Compiled shared library (built by make)
kat/
  PQCkemKAT_2321.rsp   Official KAT vectors for HQC-1
tests/
  test_unit.py   Unit and integration tests
  test_kat.py    Byte-for-byte KAT validation
bench_poly_mul.py   Naive vs Karatsuba benchmark
bench_kem.py        KeyGen / Encaps / Decaps wall-time benchmark
```

---

## Installation

Python 3.10 or later is required. The only runtime dependency is a compiled
C shared library built from the reference decoder sources.

```bash
make ref/librmrs.so
```

The only development dependency is pytest:

```bash
pip install pytest
```

---

## Usage

### Running the test suite

```bash
make test                  # all tests (unit + KAT)
pytest tests/ -v           # same, with verbose output
pytest -m slow             # end-to-end and structural tests only
KAT_N=3 pytest tests/      # run only the first 3 KAT vectors
```

KAT tests are skipped automatically if `kat/PQCkemKAT_2321.rsp` is absent.

### Running benchmarks

```bash
make bench          # quick poly-mul benchmark (1 sample)
make bench-full     # statistical poly-mul benchmark (5 rounds x 3 reps)
make bench-kem      # quick KEM benchmark (10 iterations)
make bench-kem-full # statistical KEM benchmark (50 iterations)
```

---

## Implementation notes

**Polynomial multiplication.** The dominant operation in HQC is
multiplication in F2[x]/(x^n - 1). Two algorithms are provided: a naive
O(n^2) implementation for readability and cross-validation, and a Karatsuba
O(n^1.58) implementation equivalent to `gf2x.c` from PQClean. The KEM uses
Karatsuba by default; benchmarks show roughly 2.9x speedup over naive for
n=17669, though the Python interpreter overhead limits the gain compared to
the C reference.

**RMRS decoder.** The concatenated Reed-Muller + Reed-Solomon decoder is
taken from the HQC reference implementation and compiled into a shared
library (`ref/librmrs.so`). It is called from Python via ctypes. Reimplementing
it in pure Python would add no algorithmic insight and increase the risk of
introducing errors in a component used as the ground truth.

**KAT validation.** The implementation reproduces the first 10 vectors of
the official `PQCkemKAT_2321.rsp` file byte for byte. The KAT framework uses
a SHAKE256 DRBG (new in the HQC 2025 specification, replacing the AES-256-CTR
DRBG used in earlier NIST rounds).

**Constant-time measures.** Ciphertext comparison in Decaps uses a
byte-by-byte XOR accumulator (`_ct_equal`) to avoid short-circuit leakage.
Both the real key K and the rejection key K_bar are always computed before
the branch. Full constant-time guarantees are not achievable in CPython due
to interpreter-level timing variations.

---

## References

- HQC specification (2025): https://pqc-hqc.org
- PQClean reference implementation: https://github.com/PQClean/PQClean
- NIST IR 8545 (PQC standardization report): https://doi.org/10.6028/NIST.IR.8545

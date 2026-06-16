# hqc-kem

Python 3.10+ reference implementation of **HQC-KEM** (Hamming Quasi-Cyclic Key
Encapsulation Mechanism), a code-based post-quantum KEM selected by NIST in
2025 as a standard complementary to lattice-based schemes.

Developed as a Bachelor's Thesis (TFG) at Universidad de Granada. The primary
goal is a readable, auditable implementation that maps directly to the
pseudocode in the official specification — not to compete with the C reference
on performance.

Official specification: <https://pqc-hqc.org>

---

## Repository layout

```
hqc/
  params.py      HQC-1 parameter set and derived constants
  hash.py        Hash functions G, H, I, J and XOF (SHAKE256)
  sampling.py    Fixed-weight vector samplers for key generation and encryption
  poly.py        Polynomial arithmetic: naive O(n²) and Karatsuba O(n^1.58)
  pke.py         Underlying IND-CPA PKE scheme
  kem.py         IND-CCA2 KEM with Fujisaki–Okamoto transform
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

Python 3.10 or later is required. The only runtime dependency is the RMRS
shared library, built from the bundled C sources:

```bash
make ref/librmrs.so
```

The only development dependency is pytest:

```bash
pip install pytest
```

---

## Usage

### Test suite

```bash
make test                  # all tests (unit + KAT)
pytest tests/ -v           # same, with verbose output
pytest -m slow             # end-to-end and structural tests only
KAT_N=3 pytest tests/      # run only the first 3 KAT vectors
```

KAT tests are skipped automatically when `kat/PQCkemKAT_2321.rsp` is absent.

### Benchmarks

```bash
make bench            # poly-mul benchmark (1 sample)
make bench-full       # poly-mul benchmark (5 rounds × 3 reps)
make bench-kem        # KEM benchmark (10 iterations)
make bench-kem-full   # KEM benchmark (50 iterations)
```

---

## Implementation notes

**Polynomial multiplication.** The dominant operation in HQC is multiplication
in F₂[x]/(xⁿ − 1). Two algorithms are provided: a naive O(n²) implementation
for readability and cross-validation, and a Karatsuba O(n^1.58) implementation
equivalent to `gf2x.c` from PQClean. The KEM uses Karatsuba by default;
benchmarks show roughly 2.9× speedup over naive for n = 17669, though Python
interpreter overhead limits the gain compared to the C reference.

**RMRS decoder.** The concatenated Reed-Muller + Reed-Solomon decoder is taken
from the HQC reference implementation and compiled as a shared library
(`ref/librmrs.so`), called from Python via ctypes. Reimplementing it in pure
Python would add no algorithmic insight and would risk diverging from the
ground-truth decoder.

**KAT validation.** The implementation reproduces all 100 vectors of
`PQCkemKAT_2321.rsp` byte for byte. The KAT framework uses a SHAKE256 DRBG,
introduced in the HQC 2025 specification to replace the AES-256-CTR DRBG from
earlier NIST rounds.

**Constant-time measures.** Ciphertext comparison in Decaps uses a
byte-by-byte XOR accumulator (`_ct_equal`) to avoid short-circuit leakage.
Both the real key K and the rejection key K̄ are always computed before
branching. Full constant-time guarantees are not achievable in CPython due to
interpreter-level timing variations.

---

## References

- HQC specification (2025): <https://pqc-hqc.org>
- PQClean reference implementation: <https://github.com/PQClean/PQClean>
- NIST IR 8545 (PQC standardization report): <https://doi.org/10.6028/NIST.IR.8545>

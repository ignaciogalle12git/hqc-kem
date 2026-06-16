"""
Component-level unit tests for the HQC building blocks.

Where test_kat.py proves the whole KEM byte-for-byte against the official
vectors, this file isolates each piece and checks the property it must satisfy,
so that a failure points directly at the broken component:

  * Sampling            --> fixed-weight vectors have exactly omega ones and are
                           reproducible from a fixed XOF seed.
  * Polynomial arith.   --> multiplication in F2[x]/(x^n - 1) on hand-checkable
                           cases, plus naive-vs-Karatsuba cross-validation.
  * Hash functions      --> output lengths and domain separation of G/H/I/J.
  * RMRS codec          --> decode(encode(m)) == m (round-trip of the concatenated
                           Reed-Muller + Reed-Solomon code).
  * Full KEM cycle      --> keygen -> encaps -> decaps agreement, and the implicit
                           rejection path on a corrupted ciphertext.
  * Structural tests    --> the algebraic relations behind the hardness assumptions
                           (2-QCSD in keygen, 3-DQCSD-PT in encrypt) actually hold.

Tests marked @pytest.mark.slow run on the real HQC-1 parameters (n = 17669) and
take a few seconds each; `make test` runs everything, while `pytest -m "not slow"`
skips them for a quick check and `pytest -m slow` runs only them (marker declared
in pytest.ini).
"""

import pytest
from hqc.params import HQC1
from hqc.hash import G, H, I, J, XOF
from hqc.sampling import sample_fixed_weight_keygen, sample_vect
from hqc.poly import poly_mul, poly_mul_karatsuba, poly_add, poly_truncate, get_bit


def count_ones(v: bytearray, n: int) -> int:
    # Hamming weight: count the set bits among the first n positions of v.
    return sum(get_bit(v, i) for i in range(n))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_sample_fixed_weight_exact():
    """Sampled vector has exactly omega ones."""
    xof = XOF(b'\x00' * 32)     # deterministic seed for reproducibility
    v = sample_fixed_weight_keygen(HQC1.n, HQC1.omega, xof)
    assert count_ones(v, HQC1.n) == HQC1.omega


def test_sample_fixed_weight_deterministic():
    """Same XOF seed produces the same vector."""
    v1 = sample_fixed_weight_keygen(HQC1.n, HQC1.omega, XOF(b'\xAB' * 32))
    v2 = sample_fixed_weight_keygen(HQC1.n, HQC1.omega, XOF(b'\xAB' * 32))
    assert v1 == v2


# ---------------------------------------------------------------------------
# Polynomial arithmetic
# ---------------------------------------------------------------------------

def test_poly_mul_small():
    """
    Manual verification with n=7:
    (x^0 + x^2) * (x^0 + x^1 + x^2) mod (x^7 - 1) in F2
    = x^0 + x^1 + x^3 + x^4
    """
    n = 7
    # Polynomials are stored little-endian by bit: bit i is the coefficient of x^i.
    a = bytearray(1); a[0] = 0b00000101   # bits 0 and 2  -> x^0 + x^2
    b = bytearray(1); b[0] = 0b00000111   # bits 0, 1, 2  -> x^0 + x^1 + x^2
    res = poly_mul(a, b, n)
    expected_bits = {0, 1, 3, 4}          # worked out by hand in the docstring above
    for i in range(n):
        assert get_bit(res, i) == (1 if i in expected_bits else 0)


def test_poly_mul_identity():
    """Multiplying by the unit polynomial 1 is the identity."""
    n = 31
    import hashlib
    raw = bytearray(hashlib.sha3_256(b'test').digest()[:4]) # 4 bytes = 32 bits of pseudorandom data
    raw[-1] &= (1 << (n % 8)) - 1   # clear the high bit so raw has at most n=31 coefficients
    uno = bytearray((n + 7) // 8)
    uno[0] = 0x01                   # the unit polynomial: 1 (only the x^0 coefficient)
    resultado = poly_mul(raw, uno, n)
    assert resultado == bytearray(raw[:len(resultado)]) # raw * 1 == raw


def test_poly_truncate():
    """Bits from n1*n2 to n-1 are zeroed."""
    p = HQC1
    v = bytearray(p.n_bytes)
    for i in range(p.n):
        v[i // 8] |= 1 << (i % 8)
    result = poly_truncate(v, p.n, p.n1 * p.n2)
    for i in range(p.n1 * p.n2, p.n):
        assert get_bit(result, i) == 0


def test_poly_mul_karatsuba_small():
    """Karatsuba on the same algebraic case as test_poly_mul_small (n=7)."""
    n = 7
    a = bytearray(1); a[0] = 0b00000101
    b = bytearray(1); b[0] = 0b00000111
    res = poly_mul_karatsuba(a, b, n)
    expected_bits = {0, 1, 3, 4}
    for i in range(n):
        assert get_bit(res, i) == (1 if i in expected_bits else 0)


def test_poly_mul_karatsuba_identity():
    """Karatsuba with the unit polynomial returns the input unchanged."""
    import hashlib
    n = 31
    raw = bytearray(hashlib.sha3_256(b'karatsuba').digest()[:4])
    raw[-1] &= (1 << (n % 8)) - 1
    uno = bytearray((n + 7) // 8); uno[0] = 0x01
    res = poly_mul_karatsuba(raw, uno, n)
    assert res == bytearray(raw[:len(res)])


# ---------------------------------------------------------------------------
# Hash functions
# ---------------------------------------------------------------------------

def test_G_splits_correctly():
    """G returns exactly 32+32 bytes with K != theta."""
    K, theta = G(b'test input')
    assert len(K) == 32
    assert len(theta) == 32
    assert K != theta


def test_G_deterministic():
    """Same input produces the same output."""
    K1, t1 = G(b'same input')
    K2, t2 = G(b'same input')
    assert K1 == K2 and t1 == t2


def test_I_splits_correctly():
    """I returns exactly 32+32 bytes with seed_dk != seed_ek."""
    seed_dk, seed_ek = I(b'\x42' * 32) #
    assert len(seed_dk) == 32
    assert len(seed_ek) == 32
    assert seed_dk != seed_ek


def test_H_length():
    """H returns exactly 32 bytes."""
    assert len(H(b'any input')) == 32


def test_J_length():
    """J returns exactly 32 bytes."""
    assert len(J(b'any input')) == 32


def test_domain_separation():
    """Distinct hash functions produce distinct outputs for the same input."""
    data = b'same input' * 4
    K, _ = G(data)
    assert H(data) != K
    dk, _ = I(data[:32])
    assert H(data) != dk
    assert J(data) != H(data)


# ---------------------------------------------------------------------------
# RMRS encoder / decoder
# ---------------------------------------------------------------------------

def test_rmrs_roundtrip():
    """decode(encode(m)) == m for a random message."""
    from hqc.rmrs import encode, decode
    import os
    m = os.urandom(HQC1.k // 8)
    codeword = encode(m, HQC1.n1n2_bytes)
    assert decode(codeword, HQC1.k // 8) == m


def test_rmrs_roundtrip_zero():
    """All-zero message encodes and decodes correctly."""
    from hqc.rmrs import encode, decode
    m = bytes(HQC1.k // 8)
    assert decode(encode(m, HQC1.n1n2_bytes), HQC1.k // 8) == m


# ---------------------------------------------------------------------------
# Full KEM cycle
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_end_to_end():
    """keygen -> encaps -> decaps produces matching shared secret K."""
    from hqc.kem import _kem_keygen_det, _kem_encaps_det, kem_decaps
    import os
    seed_kem = os.urandom(HQC1.seed_bytes)
    m        = os.urandom(HQC1.k // 8)
    salt     = os.urandom(HQC1.salt_bytes)

    ek, dk    = _kem_keygen_det(seed_kem, HQC1)
    K_enc, ct = _kem_encaps_det(ek, m, salt, HQC1)
    K_dec     = kem_decaps(dk, ct, HQC1)

    assert K_enc == K_dec


@pytest.mark.slow
def test_decaps_rejects_corrupt_ct():
    """A corrupted ciphertext causes decaps to return K_bar instead of K'.

    This exercises the implicit rejection of the Fujisaki-Okamoto transform:
    instead of signalling failure, decaps re-encrypts and, on mismatch, returns
    a pseudorandom key derived from the secret sigma. The attacker cannot tell
    rejection from success, which is what gives HQC its IND-CCA2 security."""
    from hqc.kem import _kem_keygen_det, _kem_encaps_det, kem_decaps
    import os
    seed_kem = os.urandom(HQC1.seed_bytes)
    m        = os.urandom(HQC1.k // 8)
    salt     = os.urandom(HQC1.salt_bytes)

    ek, dk      = _kem_keygen_det(seed_kem, HQC1)
    K_valid, ct = _kem_encaps_det(ek, m, salt, HQC1)

    ct_corrupt = bytes([ct[0] ^ 0xFF]) + ct[1:]    # flip the first byte of the ciphertext to corrupt it
    assert kem_decaps(dk, ct_corrupt, HQC1) != K_valid


# ---------------------------------------------------------------------------
# Karatsuba cross-validation against naive
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_poly_mul_equivalence_hqc1():
    """Naive and Karatsuba produce identical results on real HQC-1 vectors.

    The naive O(n^2) routine is the easy-to-audit ground truth; this test pins
    the optimised Karatsuba path to it at the production size (n = 17669), using
    a dense h and a sparse fixed-weight y like the real key material."""
    import os
    h = sample_vect(HQC1.n, XOF(os.urandom(32)))                          # dense random operand
    y = sample_fixed_weight_keygen(HQC1.n, HQC1.omega, XOF(os.urandom(32))) # sparse, weight omega
    assert poly_mul(h, y, HQC1.n) == poly_mul_karatsuba(h, y, HQC1.n)


# ---------------------------------------------------------------------------
# Structural tests: 2-QCSD and 3-DQCSD-PT instances
#
# These do not just check that the code runs: they assert that the public data
# produced by keygen/encrypt is exactly the syndrome-decoding instance whose
# hardness HQC's security reduces to. If these relations held only by accident,
# the published key/ciphertext would not actually hide the secret.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_pke_keygen_constructs_2qcsd_instance():
    """Keygen builds a valid 2-QCSD instance: s == x + h*y.

    The public key is (h, s) with x, y secret low-weight vectors. Recovering
    (x, y) from (h, s) is the 2-Quasi-Cyclic Syndrome Decoding problem; here we
    just confirm s was computed from that relation."""
    from hqc.pke import _pke_keygen_internal
    import os
    seed_pke = os.urandom(HQC1.seed_bytes)
    ek, _dk_seed, x, y, h, s = _pke_keygen_internal(seed_pke, HQC1)
    assert s == poly_add(x, poly_mul_karatsuba(h, y, HQC1.n)) # s = x + h*y over F2[x]/(x^n - 1)


@pytest.mark.slow
def test_pke_encrypt_constructs_3dqcsd_pt_instance():
    """Encrypt builds a valid 3-DQCSD-PT instance: u == r1 + h*r2 and
    v - Encode(m) == Truncate(s*r2 + e).

    The ciphertext (u, v) masks the encoded message with the secret low-weight
    randomness (r1, r2, e). Distinguishing it from random is the 3-Decisional
    Quasi-Cyclic Syndrome Decoding problem with Parity Test. We verify both
    halves of the ciphertext match their defining equations."""
    from hqc.pke import _pke_keygen_internal, _pke_encrypt_internal
    from hqc.rmrs import encode
    import os

    seed_pke = os.urandom(HQC1.seed_bytes)
    ek, _dk_seed, _x, _y, h, s = _pke_keygen_internal(seed_pke, HQC1)

    m = os.urandom(HQC1.k // 8)
    theta = os.urandom(HQC1.seed_bytes)
    u, v, r1, r2, e = _pke_encrypt_internal(ek, m, theta, HQC1)

    # First half of the ciphertext: u = r1 + h*r2 (same QCSD shape as the key).
    assert u == poly_add(r1, poly_mul_karatsuba(h, r2, HQC1.n))

    # Second half: v = Truncate(s*r2 + e) + Encode(m), so removing the encoded

    # message must leave exactly the truncated s*r2 + e term. We obtain that
    # noise two independent ways and check they match.
    encoded_m = bytearray(encode(m, HQC1.n1n2_bytes))

    # Way 1: peel Encode(m) off the real ciphertext v (subtraction is XOR in F2).
    noise_from_v = poly_add(bytearray(v), encoded_m)

    # Way 2: rebuild the noise from its definition, Truncate(s*r2 + e).
    sr2_plus_e      = poly_add(poly_mul_karatsuba(s, r2, HQC1.n), e)
    noise_truncated = poly_truncate(sr2_plus_e, HQC1.n, HQC1.n1 * HQC1.n2)  # keep first n1*n2 bits
    noise_expected  = bytearray(noise_truncated[:HQC1.n1n2_bytes])          # trim to n1n2_bytes

    assert noise_from_v == noise_expected

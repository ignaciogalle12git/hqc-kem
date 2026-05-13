import pytest
from hqc.params import HQC1
from hqc.hash import G, H, I, J, XOF
from hqc.sampling import sample_fixed_weight_keygen, sample_vect
from hqc.poly import poly_mul, poly_mul_karatsuba, poly_add, poly_truncate, get_bit


def count_ones(v: bytearray, n: int) -> int:
    return sum(get_bit(v, i) for i in range(n))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def test_sample_fixed_weight_exact():
    """Sampled vector has exactly omega ones."""
    xof = XOF(b'\x00' * 32)
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
    a = bytearray(1); a[0] = 0b00000101   # bits 0 and 2
    b = bytearray(1); b[0] = 0b00000111   # bits 0, 1 and 2
    res = poly_mul(a, b, n)
    expected_bits = {0, 1, 3, 4}
    for i in range(n):
        assert get_bit(res, i) == (1 if i in expected_bits else 0)


def test_poly_mul_identity():
    """Multiplying by the unit polynomial 1 is the identity."""
    n = 31
    import hashlib
    raw = bytearray(hashlib.sha3_256(b'test').digest()[:4])
    raw[-1] &= (1 << (n % 8)) - 1
    uno = bytearray((n + 7) // 8)
    uno[0] = 0x01
    resultado = poly_mul(raw, uno, n)
    assert resultado == bytearray(raw[:len(resultado)])


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
    seed_dk, seed_ek = I(b'\x42' * 32)
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
    """A corrupted ciphertext causes decaps to return K_bar instead of K'."""
    from hqc.kem import _kem_keygen_det, _kem_encaps_det, kem_decaps
    import os
    seed_kem = os.urandom(HQC1.seed_bytes)
    m        = os.urandom(HQC1.k // 8)
    salt     = os.urandom(HQC1.salt_bytes)

    ek, dk      = _kem_keygen_det(seed_kem, HQC1)
    K_valid, ct = _kem_encaps_det(ek, m, salt, HQC1)

    ct_corrupt = bytes([ct[0] ^ 0xFF]) + ct[1:]
    assert kem_decaps(dk, ct_corrupt, HQC1) != K_valid


# ---------------------------------------------------------------------------
# Karatsuba cross-validation against naive
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_poly_mul_equivalence_hqc1():
    """Naive and Karatsuba produce identical results on real HQC-1 vectors."""
    import os
    h = sample_vect(HQC1.n, XOF(os.urandom(32)))
    y = sample_fixed_weight_keygen(HQC1.n, HQC1.omega, XOF(os.urandom(32)))
    assert poly_mul(h, y, HQC1.n) == poly_mul_karatsuba(h, y, HQC1.n)


# ---------------------------------------------------------------------------
# Structural tests: 2-QCSD and 3-DQCSD-PT instances
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_pke_keygen_constructs_2qcsd_instance():
    """Keygen builds a valid 2-QCSD instance: s == x + h*y."""
    from hqc.pke import _pke_keygen_internal
    import os
    seed_pke = os.urandom(HQC1.seed_bytes)
    ek, _dk_seed, x, y, h, s = _pke_keygen_internal(seed_pke, HQC1)
    assert s == poly_add(x, poly_mul_karatsuba(h, y, HQC1.n))


@pytest.mark.slow
def test_pke_encrypt_constructs_3dqcsd_pt_instance():
    """Encrypt builds a valid 3-DQCSD-PT instance: u == r1 + h*r2 and
    v - Encode(m) == Truncate(s*r2 + e)."""
    from hqc.pke import _pke_keygen_internal, _pke_encrypt_internal
    from hqc.rmrs import encode
    import os
    seed_pke = os.urandom(HQC1.seed_bytes)
    ek, _dk_seed, _x, _y, h, s = _pke_keygen_internal(seed_pke, HQC1)
    m = os.urandom(HQC1.k // 8)
    theta = os.urandom(HQC1.seed_bytes)
    u, v, r1, r2, e = _pke_encrypt_internal(ek, m, theta, HQC1)

    assert u == poly_add(r1, poly_mul_karatsuba(h, r2, HQC1.n))

    encoded_m = bytearray(encode(m, HQC1.n1n2_bytes))
    lhs = poly_add(bytearray(v), encoded_m)

    sr2_plus_e = poly_add(poly_mul_karatsuba(s, r2, HQC1.n), e)
    rhs_full = poly_truncate(sr2_plus_e, HQC1.n, HQC1.n1 * HQC1.n2)
    rhs = bytearray(rhs_full[:HQC1.n1n2_bytes])

    assert lhs == rhs

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
    """El vector muestreado tiene exactamente omega unos."""
    xof = XOF(b'\x00' * 32)
    v = sample_fixed_weight_keygen(HQC1.n, HQC1.omega, xof)
    assert count_ones(v, HQC1.n) == HQC1.omega


def test_sample_fixed_weight_deterministic():
    """El mismo XOF produce el mismo vector."""
    v1 = sample_fixed_weight_keygen(HQC1.n, HQC1.omega, XOF(b'\xAB' * 32))
    v2 = sample_fixed_weight_keygen(HQC1.n, HQC1.omega, XOF(b'\xAB' * 32))
    assert v1 == v2


# ---------------------------------------------------------------------------
# Aritmética polinómica
# ---------------------------------------------------------------------------

def test_poly_mul_small():
    """
    Verificar multiplicación con n=7.
    (x^0 + x^2) * (x^0 + x^1 + x^2) mod (x^7 - 1) en F2
    = x^0 + x^1 + x^2 + x^2 + x^3 + x^4
    = x^0 + x^1 + x^3 + x^4   (x^2 + x^2 = 0 en F2)
    """
    n = 7
    a = bytearray(1); a[0] = 0b00000101   # bits 0 y 2
    b = bytearray(1); b[0] = 0b00000111   # bits 0, 1 y 2
    res = poly_mul(a, b, n)
    expected_bits = {0, 1, 3, 4}
    for i in range(n):
        assert get_bit(res, i) == (1 if i in expected_bits else 0)


def test_poly_mul_identity():
    """Multiplicar por el polinomio unidad (1) devuelve el mismo polinomio."""
    n = 31
    # Construir un polinomio aleatorio de n=31 bits
    import hashlib
    raw = bytearray(hashlib.sha3_256(b'test').digest()[:4])
    raw[-1] &= (1 << (n % 8)) - 1   # enmascarar bits sobrantes
    # Identidad: 1 tiene solo el bit 0 activo
    uno = bytearray((n + 7) // 8)
    uno[0] = 0x01
    resultado = poly_mul(raw, uno, n)
    assert resultado == bytearray(raw[:len(resultado)])


def test_poly_truncate():
    """Los bits desde n1*n2 hasta n-1 quedan a cero."""
    p = HQC1
    v = bytearray(p.n_bytes)
    for i in range(p.n):
        v[i // 8] |= 1 << (i % 8)   # todos los bits a 1
    result = poly_truncate(v, p.n, p.n1 * p.n2)
    for i in range(p.n1 * p.n2, p.n):
        assert get_bit(result, i) == 0


# ---------------------------------------------------------------------------
# Funciones hash
# ---------------------------------------------------------------------------

def test_G_splits_correctly():
    """G devuelve exactamente 32+32 bytes."""
    K, theta = G(b'test input')
    assert len(K) == 32
    assert len(theta) == 32
    assert K != theta


def test_G_deterministic():
    """Misma entrada, misma salida."""
    K1, t1 = G(b'same input')
    K2, t2 = G(b'same input')
    assert K1 == K2 and t1 == t2


def test_I_splits_correctly():
    """I devuelve exactamente 32+32 bytes (seed_dk, seed_ek)."""
    seed_dk, seed_ek = I(b'\x42' * 32)
    assert len(seed_dk) == 32
    assert len(seed_ek) == 32
    assert seed_dk != seed_ek


def test_H_length():
    """H devuelve exactamente 32 bytes."""
    h = H(b'cualquier entrada')
    assert len(h) == 32


def test_J_length():
    """J devuelve exactamente 32 bytes."""
    j = J(b'cualquier entrada')
    assert len(j) == 32


def test_domain_separation():
    """Funciones distintas producen salidas distintas para la misma entrada."""
    data = b'misma entrada' * 4
    K, _ = G(data)
    assert H(data) != K        # G y H son distintas
    dk, _ = I(data[:32])
    assert H(data) != dk       # H e I son distintas
    assert J(data) != H(data)  # J y H son distintas


# ---------------------------------------------------------------------------
# Encoder/decoder RMRS
# ---------------------------------------------------------------------------

def test_rmrs_roundtrip():
    """encode seguido de decode recupera el mensaje original."""
    from hqc.rmrs import encode, decode
    import os
    m = os.urandom(HQC1.k // 8)   # 16 bytes aleatorios
    codeword = encode(m, HQC1.n1n2_bytes)
    m_recovered = decode(codeword, HQC1.k // 8)
    assert m_recovered == m, "RMRS: decode(encode(m)) != m"


def test_rmrs_roundtrip_zero():
    """El mensaje todo-ceros también se codifica y decodifica correctamente."""
    from hqc.rmrs import encode, decode
    m = bytes(HQC1.k // 8)
    codeword = encode(m, HQC1.n1n2_bytes)
    assert decode(codeword, HQC1.k // 8) == m


# ---------------------------------------------------------------------------
# Ciclo completo KEM (lento: ~2-3 min)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_end_to_end():
    """keygen → encaps → decaps produce la misma clave compartida K."""
    from hqc.kem import _kem_keygen_det, _kem_encaps_det, kem_decaps
    import os
    seed_kem = os.urandom(HQC1.seed_bytes)
    m        = os.urandom(HQC1.k // 8)
    salt     = os.urandom(HQC1.salt_bytes)

    ek, dk      = _kem_keygen_det(seed_kem, HQC1)
    K_enc, ct   = _kem_encaps_det(ek, m, salt, HQC1)
    K_dec       = kem_decaps(dk, ct, HQC1)

    assert K_enc == K_dec, "K de encaps y decaps no coinciden"


@pytest.mark.slow
def test_decaps_rejects_corrupt_ct():
    """Un ciphertext corrupto hace que decaps devuelva K_bar (≠ K_prime)."""
    from hqc.kem import _kem_keygen_det, _kem_encaps_det, kem_decaps
    import os
    seed_kem = os.urandom(HQC1.seed_bytes)
    m        = os.urandom(HQC1.k // 8)
    salt     = os.urandom(HQC1.salt_bytes)

    ek, dk    = _kem_keygen_det(seed_kem, HQC1)
    K_valid, ct = _kem_encaps_det(ek, m, salt, HQC1)

    # Corromper el primer byte de ct
    ct_corrupt = bytes([ct[0] ^ 0xFF]) + ct[1:]
    K_corrupt  = kem_decaps(dk, ct_corrupt, HQC1)

    assert K_corrupt != K_valid, "decaps no rechazó el ciphertext corrupto"


# ---------------------------------------------------------------------------
# Karatsuba: corrección frente al algoritmo naive (Issue #1)
# ---------------------------------------------------------------------------

def test_poly_mul_karatsuba_small():
    """Karatsuba sobre el mismo caso algebraico que test_poly_mul_small (n=7)."""
    n = 7
    a = bytearray(1); a[0] = 0b00000101
    b = bytearray(1); b[0] = 0b00000111
    res = poly_mul_karatsuba(a, b, n)
    expected_bits = {0, 1, 3, 4}
    for i in range(n):
        assert get_bit(res, i) == (1 if i in expected_bits else 0)


def test_poly_mul_karatsuba_identity():
    """Karatsuba con el polinomio unidad devuelve el mismo polinomio."""
    import hashlib
    n = 31
    raw = bytearray(hashlib.sha3_256(b'karatsuba').digest()[:4])
    raw[-1] &= (1 << (n % 8)) - 1
    uno = bytearray((n + 7) // 8); uno[0] = 0x01
    res = poly_mul_karatsuba(raw, uno, n)
    assert res == bytearray(raw[:len(res)])


@pytest.mark.slow
def test_poly_mul_equivalence_hqc1():
    """naive y Karatsuba producen el mismo resultado sobre vectores reales HQC-1."""
    import os
    h = sample_vect(HQC1.n, XOF(os.urandom(32)))
    y = sample_fixed_weight_keygen(HQC1.n, HQC1.omega, XOF(os.urandom(32)))
    assert poly_mul(h, y, HQC1.n) == poly_mul_karatsuba(h, y, HQC1.n)


# ---------------------------------------------------------------------------
# Tests estructurales: instancias 2-QCSD y 3-DQCSD-PT (Issue #3)
#
# Verifican que el código construye correctamente las instancias de los
# problemas duros sobre los que se reduce la seguridad de HQC-PKE
# (capítulo 4 del TFG).
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_pke_keygen_constructs_2qcsd_instance():
    """En keygen: s == x + h·y (instancia de 2-QCSD)."""
    from hqc.pke import _pke_keygen_internal  # variante que expone x, y, h, s
    import os
    seed_pke = os.urandom(HQC1.seed_bytes)
    ek, _dk_seed, x, y, h, s = _pke_keygen_internal(seed_pke, HQC1)
    assert s == poly_add(x, poly_mul_karatsuba(h, y, HQC1.n)), \
        "Keygen no construye una instancia válida de 2-QCSD: s != x + h·y"


@pytest.mark.slow
def test_pke_encrypt_constructs_3dqcsd_pt_instance():
    """En encrypt: u == r1 + h·r2  y  v - Encode(m) == Truncate(s·r2 + e)."""
    from hqc.pke import _pke_keygen_internal, _pke_encrypt_internal
    from hqc.rmrs import encode
    import os
    seed_pke = os.urandom(HQC1.seed_bytes)
    ek, _dk_seed, _x, _y, h, s = _pke_keygen_internal(seed_pke, HQC1)
    m = os.urandom(HQC1.k // 8)
    theta = os.urandom(HQC1.seed_bytes)
    u, v, r1, r2, e = _pke_encrypt_internal(ek, m, theta, HQC1)

    # Comprobación 1: u = r1 + h·r2  (n bits)
    assert u == poly_add(r1, poly_mul_karatsuba(h, r2, HQC1.n)), \
        "Encrypt no construye u = r1 + h·r2"

    # Comprobación 2: v - Encode(m) = Truncate(s·r2 + e)  (n1·n2 bits)
    # En F2 la resta es XOR; v y Encode(m) viven en n1n2_bytes.
    encoded_m = bytearray(encode(m, HQC1.n1n2_bytes))
    lhs = poly_add(bytearray(v), encoded_m)

    sr2_plus_e = poly_add(poly_mul_karatsuba(s, r2, HQC1.n), e)
    rhs_full = poly_truncate(sr2_plus_e, HQC1.n, HQC1.n1 * HQC1.n2)
    rhs = bytearray(rhs_full[:HQC1.n1n2_bytes])

    assert lhs == rhs, \
        "Encrypt no construye v - Encode(m) = Truncate(s·r2 + e)"

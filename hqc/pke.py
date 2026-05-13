from .params import HQCParams
from .hash import I, XOF
from .sampling import sample_vect, sample_fixed_weight_keygen, sample_fixed_weight_encrypt
from .poly import poly_add, poly_mul_karatsuba as poly_mul, poly_mul as poly_mul_naive, poly_truncate  # noqa: F401


def pke_keygen(seed_pke: bytes, p: HQCParams) -> tuple[bytes, bytes]:
    """
    Return (ek, dk_pke).
    ek  = seed_ek (32 B) || s (n_bytes)
    dk  = seed_dk (32 B)
    """
    ek, dk, _x, _y, _h, _s = _pke_keygen_internal(seed_pke, p)
    return ek, dk


def _pke_keygen_internal(seed_pke: bytes, p: HQCParams):
    """
    Internal variant of pke_keygen that also returns the intermediate vectors
    x, y, h, s for structural correctness tests (2-QCSD instance check).
    """
    seed_dk, seed_ek = I(seed_pke)

    # y is sampled before x as required by the specification; this ordering
    # also enables hardware optimisations (Antognazza et al. 2024).
    ctx_dk = XOF(seed_dk)
    y = sample_fixed_weight_keygen(p.n, p.omega, ctx_dk)
    x = sample_fixed_weight_keygen(p.n, p.omega, ctx_dk)

    ctx_ek = XOF(seed_ek)
    h = sample_vect(p.n, ctx_ek)
    s = poly_add(x, poly_mul(h, y, p.n))

    ek = seed_ek + bytes(s)
    dk = seed_dk
    return ek, dk, x, y, h, s


def pke_encrypt(ek: bytes, m: bytes, theta: bytes, p: HQCParams) -> bytes:
    """Return c_pke = u (n_bytes) || v (n1n2_bytes)."""
    u, v, _r1, _r2, _e = _pke_encrypt_internal(ek, m, theta, p)
    return bytes(u) + bytes(v)


def _pke_encrypt_internal(ek: bytes, m: bytes, theta: bytes, p: HQCParams):
    """
    Internal variant of pke_encrypt that also returns the intermediate vectors
    r1, r2, e and the separated outputs u, v for structural correctness tests
    (3-DQCSD-PT instance check).
    """
    from .rmrs import encode as rmrs_encode

    seed_ek = ek[:p.seed_bytes]
    s = bytearray(ek[p.seed_bytes:])

    ctx_ek = XOF(seed_ek)
    h = sample_vect(p.n, ctx_ek)

    # Mandatory sampling order for KAT reproducibility: r2, e, r1.
    ctx_theta = XOF(theta)
    r2 = sample_fixed_weight_encrypt(p.n, p.omega_r, ctx_theta)
    e  = sample_fixed_weight_encrypt(p.n, p.omega_e, ctx_theta)
    r1 = sample_fixed_weight_encrypt(p.n, p.omega_r, ctx_theta)

    u = poly_add(r1, poly_mul(h, r2, p.n))

    m_encoded = rmrs_encode(m, p.n1n2_bytes)
    m_bits = bytearray(m_encoded)

    sr2_e = poly_add(poly_mul(s, r2, p.n), e)
    sr2_e_trunc = poly_truncate(sr2_e, p.n, p.n1 * p.n2)
    v = poly_add(m_bits, bytearray(sr2_e_trunc[:p.n1n2_bytes]))

    return u, v, r1, r2, e


def pke_decrypt(dk: bytes, c_pke: bytes, p: HQCParams) -> bytes | None:
    """Return the recovered message, or None if the decoder fails."""
    from .rmrs import decode as rmrs_decode

    seed_dk = dk[:p.seed_bytes]

    ctx_dk = XOF(seed_dk)
    y = sample_fixed_weight_keygen(p.n, p.omega, ctx_dk)
    # x is not used in decryption

    u = bytearray(c_pke[:p.n_bytes])
    v = bytearray(c_pke[p.n_bytes:p.n_bytes + p.n1n2_bytes])

    uy = poly_truncate(poly_mul(u, y, p.n), p.n, p.n1 * p.n2)
    v_prime = poly_add(v, bytearray(uy[:p.n1n2_bytes]))

    m_recovered = rmrs_decode(bytes(v_prime), p.k // 8)
    if m_recovered is None:
        return None
    return m_recovered

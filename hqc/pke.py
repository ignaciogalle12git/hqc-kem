from .params import HQCParams
from .hash import I, XOF
from .sampling import sample_vect, sample_fixed_weight_keygen, sample_fixed_weight_encrypt
from .poly import poly_add, poly_mul, poly_truncate


def pke_keygen(seed_pke: bytes, p: HQCParams) -> tuple[bytes, bytes]:
    """
    Devuelve (ek, dk_pke).
    ek  = seed_ek (32B) || s (n_bytes)
    dk  = seed_dk (32B)
    """
    seed_dk, seed_ek = I(seed_pke)

    # Clave privada: y se muestrea ANTES que x (orden de la spec)
    ctx_dk = XOF(seed_dk)
    y = sample_fixed_weight_keygen(p.n, p.omega, ctx_dk)
    x = sample_fixed_weight_keygen(p.n, p.omega, ctx_dk)

    # Clave pública
    ctx_ek = XOF(seed_ek)
    h = sample_vect(p.n, ctx_ek)
    s = poly_add(x, poly_mul(h, y, p.n))

    ek = seed_ek + bytes(s)
    dk = seed_dk
    return ek, dk


def pke_encrypt(ek: bytes, m: bytes, theta: bytes, p: HQCParams) -> bytes:
    """
    Devuelve c_pke = u (n_bytes) || v (n1n2_bytes)
    """
    from .rmrs import encode as rmrs_encode

    seed_ek = ek[:p.seed_bytes]
    s = bytearray(ek[p.seed_bytes:])

    ctx_ek = XOF(seed_ek)
    h = sample_vect(p.n, ctx_ek)

    # Orden obligatorio: r2, e, r1
    ctx_theta = XOF(theta)
    r2 = sample_fixed_weight_encrypt(p.n, p.omega_r, ctx_theta)
    e  = sample_fixed_weight_encrypt(p.n, p.omega_e, ctx_theta)
    r1 = sample_fixed_weight_encrypt(p.n, p.omega_r, ctx_theta)

    u = poly_add(r1, poly_mul(h, r2, p.n))

    m_encoded = rmrs_encode(m, p.n1n2_bytes)
    m_bits = bytearray(m_encoded)

    sr2_e = poly_add(poly_mul(s, r2, p.n), e)
    sr2_e_trunc = poly_truncate(sr2_e, p.n, p.n1 * p.n2)
    # v vive en n1*n2 bits: recortamos el byte sobrante (ya en cero tras truncate)
    # antes de sumar con m_bits para que poly_add pueda validar longitudes.
    v = poly_add(m_bits, bytearray(sr2_e_trunc[:p.n1n2_bytes]))

    return bytes(u) + bytes(v)


def pke_decrypt(dk: bytes, c_pke: bytes, p: HQCParams) -> bytes | None:
    """
    Devuelve m recuperado, o None si el decoder falla.
    """
    from .rmrs import decode as rmrs_decode

    seed_dk = dk[:p.seed_bytes]

    ctx_dk = XOF(seed_dk)
    y = sample_fixed_weight_keygen(p.n, p.omega, ctx_dk)
    # x no se usa en decrypt

    u = bytearray(c_pke[:p.n_bytes])
    v = bytearray(c_pke[p.n_bytes:p.n_bytes + p.n1n2_bytes])

    uy = poly_truncate(poly_mul(u, y, p.n), p.n, p.n1 * p.n2)
    # v vive en n1*n2 bits; recortamos el byte sobrante de uy (ya en cero)
    v_prime = poly_add(v, bytearray(uy[:p.n1n2_bytes]))

    m_recovered = rmrs_decode(bytes(v_prime), p.k // 8)
    if m_recovered is None:
        return None
    return m_recovered

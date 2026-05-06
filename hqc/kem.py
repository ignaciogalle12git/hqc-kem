import os
from .params import HQCParams
from .hash import G, H, J, XOF
from .pke import pke_keygen, pke_encrypt, pke_decrypt


def kem_keygen(p: HQCParams) -> tuple[bytes, bytes]:
    """Devuelve (ek, dk)."""
    seed_kem = os.urandom(p.seed_bytes)

    ctx = XOF(seed_kem)
    seed_pke = ctx.get_bytes(p.seed_bytes)
    sigma    = ctx.get_bytes(p.sigma_bytes)

    ek, dk_pke = pke_keygen(seed_pke, p)

    dk = ek + dk_pke + sigma + seed_kem
    return ek, dk


def kem_encaps(ek: bytes, p: HQCParams) -> tuple[bytes, bytes]:
    """Devuelve (K, ct)."""
    m    = os.urandom(p.k // 8)
    salt = os.urandom(p.salt_bytes)

    K, theta = G(H(ek) + m + salt)

    c_pke = pke_encrypt(ek, m, theta, p)
    ct = c_pke + salt
    return K, ct


def kem_decaps(dk: bytes, ct: bytes, p: HQCParams) -> bytes:
    """Devuelve K (clave compartida)."""
    # Parsear dk
    ek      = dk[:p.ek_bytes]
    dk_pke  = dk[p.ek_bytes : p.ek_bytes + p.seed_bytes]
    sigma   = dk[p.ek_bytes + p.seed_bytes : p.ek_bytes + p.seed_bytes + p.sigma_bytes]

    # Parsear ct
    c_pke = ct[:p.n_bytes + p.n1n2_bytes]
    salt  = ct[p.n_bytes + p.n1n2_bytes:]

    m_hat = pke_decrypt(dk_pke, c_pke, p)
    decode_failed = m_hat is None

    if decode_failed:
        m_hat = b'\x00' * (p.k // 8)   # valor neutro para re-cifrado

    K_prime, theta_prime = G(H(ek) + m_hat + salt)
    c_pke_prime = pke_encrypt(ek, m_hat, theta_prime, p)

    K_bar = J(H(ek) + sigma + ct)

    # Rechazo implícito (FO): devolver K_bar si el decoder falló o si el
    # re-cifrado no coincide con ct. Se usa un flag explícito porque un m
    # legítimo puede ser todo ceros, y el chequeo anterior `m_hat == 0·…`
    # colisionaba con ese caso.
    if decode_failed or not _ct_equal(c_pke_prime, c_pke):
        return K_bar
    return K_prime


def _kem_keygen_det(seed_kem: bytes, p: HQCParams) -> tuple[bytes, bytes]:
    """Variante determinista de kem_keygen para tests KAT (seed_kem explícita)."""
    ctx = XOF(seed_kem)
    seed_pke = ctx.get_bytes(p.seed_bytes)
    sigma    = ctx.get_bytes(p.sigma_bytes)
    ek, dk_pke = pke_keygen(seed_pke, p)
    dk = ek + dk_pke + sigma + seed_kem
    return ek, dk


def _kem_encaps_det(ek: bytes, m: bytes, salt: bytes, p: HQCParams) -> tuple[bytes, bytes]:
    """Variante determinista de kem_encaps para tests KAT (m y salt explícitos)."""
    K, theta = G(H(ek) + m + salt)
    c_pke = pke_encrypt(ek, m, theta, p)
    ct = c_pke + salt
    return K, ct


def _ct_equal(a: bytes, b: bytes) -> bool:
    """Comparación en tiempo constante."""
    if len(a) != len(b):
        return False
    diff = 0
    for x, y in zip(a, b):
        diff |= x ^ y
    return diff == 0

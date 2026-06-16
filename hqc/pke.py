"""
IND-CPA public-key encryption underlying the HQC KEM. Security rests on two
syndrome-decoding assumptions: keygen produces a 2-QCSD instance and encrypt a
3-DQCSD-PT instance (the structural tests in tests/test_unit.py check both).

All arithmetic is in the ring F2[x]/(x^n - 1); poly_mul is the Karatsuba product
(poly_mul_naive is imported only so tests can cross-check it).

Adapted from src/ref/hqc.c in the official HQC implementation (v5, 2025):
    https://gitlab.com/pqc-hqc/hqc/-/blob/main/src/ref/hqc.c
pke_keygen / pke_encrypt / pke_decrypt mirror hqc_pke_keygen / _encrypt / _decrypt.
"""

from .params import HQCParams
from .hash import I, XOF
from .sampling import sample_vect, sample_fixed_weight_keygen, sample_fixed_weight_encrypt
from .poly import poly_add, poly_mul_karatsuba as poly_mul, poly_mul as poly_mul_naive, poly_truncate  # noqa: F401


def pke_keygen(seed_pke: bytes, p: HQCParams) -> tuple[bytes, bytes]:
    """Generate a PKE key pair.

    Parameters
    ----------
    seed_pke : bytes
        32-byte seed; expands into the secret (x, y) and public h.
    p : HQCParams
        HQC parameter set (e.g. HQC1).

    Returns
    -------
    tuple[bytes, bytes]
        (ek, dk_pke): ek = seed_ek (32 B) || s (n_bytes); dk_pke = seed_dk (32 B).
    """
    ek, dk, _x, _y, _h, _s = _pke_keygen_internal(seed_pke, p)
    return ek, dk


def _pke_keygen_internal(seed_pke: bytes, p: HQCParams):
    """
    Internal variant of pke_keygen that also returns the intermediate vectors
    x, y, h, s for structural correctness tests (2-QCSD instance check).
    """
    # Split the PKE seed into two independent seeds: one for the secret vectors
    # (x, y) and one for the public h.
    seed_dk, seed_ek = I(seed_pke)

    # x and y are the secret low-weight vectors (weight omega each). y is sampled
    # before x as required by the specification; this ordering also enables
    # hardware optimisations (Antognazza et al. 2024).
    ctx_dk = XOF(seed_dk)
    y = sample_fixed_weight_keygen(p.n, p.omega, ctx_dk)
    x = sample_fixed_weight_keygen(p.n, p.omega, ctx_dk)

    # h is a full-length uniform vector; s = x + h*y is the public syndrome.
    # Recovering (x, y) from the public (h, s) is the 2-QCSD problem.
    ctx_ek = XOF(seed_ek)
    h = sample_vect(p.n, ctx_ek)
    s = poly_add(x, poly_mul(h, y, p.n))

    # Public key stores seed_ek (so h can be regenerated) plus s; secret key is
    # just seed_dk (x, y are re-sampled from it on demand).
    ek = seed_ek + bytes(s) # ek = seed_ek (32 B) || s (n_bytes) -> Conatenated
    dk = seed_dk
    return ek, dk, x, y, h, s


def pke_encrypt(ek: bytes, m: bytes, theta: bytes, p: HQCParams) -> bytes:
    """Encrypt a message under the PKE public key.

    Parameters
    ----------
    ek : bytes
        Public key (seed_ek || s) from pke_keygen.
    m : bytes
        Message to encrypt, k/8 bytes.
    theta : bytes
        32-byte seed for the encryption randomness (r1, r2, e).
    p : HQCParams
        HQC parameter set (e.g. HQC1).

    Returns
    -------
    bytes
        c_pke = u (n_bytes) || v (n1n2_bytes).
    """
    u, v, _r1, _r2, _e = _pke_encrypt_internal(ek, m, theta, p)
    return bytes(u) + bytes(v)


def _pke_encrypt_internal(ek: bytes, m: bytes, theta: bytes, p: HQCParams):
    """
    Internal variant of pke_encrypt that also returns the intermediate vectors
    r1, r2, e and the separated outputs u, v for structural correctness tests
    (3-DQCSD-PT instance check).
    """
    from .rmrs import encode as rmrs_encode

    # Recover the public key parts: seed_ek regenerates h, s is read directly.
    seed_ek = ek[:p.seed_bytes]
    s = bytearray(ek[p.seed_bytes:])

    ctx_ek = XOF(seed_ek)
    h = sample_vect(p.n, ctx_ek)

    # Three secret low-weight vectors derived from theta (the FO randomness).
    # Mandatory sampling order for KAT reproducibility: r2, e, r1.
    ctx_theta = XOF(theta)
    r2 = sample_fixed_weight_encrypt(p.n, p.omega_r, ctx_theta)
    e  = sample_fixed_weight_encrypt(p.n, p.omega_e, ctx_theta)
    r1 = sample_fixed_weight_encrypt(p.n, p.omega_r, ctx_theta)

    # First ciphertext half: u = r1 + h*r2 (a fresh QCSD syndrome).
    u = poly_add(r1, poly_mul(h, r2, p.n))

    # Encode the message with the RMRS code so the decoder can later correct the
    # noise added by the s*r2 + e mask.
    m_encoded = rmrs_encode(m, p.n1n2_bytes)
    m_bits = bytearray(m_encoded)

    # Second half: v = Encode(m) + Truncate(s*r2 + e). The truncation drops the
    # tail beyond the n1*n2 code length; s*r2 + e is the low-weight noise term.
    sr2_e = poly_add(poly_mul(s, r2, p.n), e)
    sr2_e_trunc = poly_truncate(sr2_e, p.n, p.n1 * p.n2)
    v = poly_add(m_bits, bytearray(sr2_e_trunc[:p.n1n2_bytes]))

    return u, v, r1, r2, e


def pke_decrypt(dk: bytes, c_pke: bytes, p: HQCParams) -> bytes | None:
    """Decrypt a PKE ciphertext.

    Parameters
    ----------
    dk : bytes
        Secret key (seed_dk) from pke_keygen.
    c_pke : bytes
        Ciphertext u || v from pke_encrypt.
    p : HQCParams
        HQC parameter set (e.g. HQC1).

    Returns
    -------
    bytes or None
        The recovered message (k/8 bytes), or None if the decoder fails.
    """
    from .rmrs import decode as rmrs_decode

    # Re-sample the secret y from seed_dk. y is the first vector drawn in keygen,
    # so the same order must be used here; x is never needed for decryption.
    seed_dk = dk[:p.seed_bytes]

    ctx_dk = XOF(seed_dk)
    y = sample_fixed_weight_keygen(p.n, p.omega, ctx_dk)
    # x is not used in decryption

    u = bytearray(c_pke[:p.n_bytes])
    v = bytearray(c_pke[p.n_bytes:p.n_bytes + p.n1n2_bytes])

    # v' = v - Truncate(u*y) leaves Encode(m) plus a low-weight error: the cross
    # terms cancel so that u*y - (s*r2 + e) reduces to a correctable noise word.
    uy = poly_truncate(poly_mul(u, y, p.n), p.n, p.n1 * p.n2)
    v_prime = poly_add(v, bytearray(uy[:p.n1n2_bytes]))

    # The RMRS decoder removes the remaining noise; None means the error weight
    # exceeded the code's correction capacity (a decryption/decode failure).
    m_recovered = rmrs_decode(bytes(v_prime), p.k // 8)
    if m_recovered is None:
        return None
    return m_recovered

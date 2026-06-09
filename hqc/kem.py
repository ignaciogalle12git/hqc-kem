"""
IND-CCA2 KEM layer: the Fujisaki-Okamoto transform wrapped around the IND-CPA
PKE in pke.py. Keygen/encaps/decaps here turn the malleable public-key scheme
into a key-encapsulation mechanism that is safe against chosen-ciphertext attacks.

Adapted from src/common/kem.c in the official HQC implementation (v5, 2025):
    https://gitlab.com/pqc-hqc/hqc/-/blob/main/src/common/kem.c
kem_keygen / kem_encaps / kem_decaps mirror crypto_kem_keypair / _enc / _dec.
"""

import os
from .params import HQCParams
from .hash import G, H, J, XOF
from .pke import pke_keygen, pke_encrypt, pke_decrypt


def kem_keygen(p: HQCParams) -> tuple[bytes, bytes]:
    """Return (ek, dk)."""
    seed_kem = os.urandom(p.seed_bytes)

    # Expand the single KEM seed into the PKE seed and the rejection secret sigma.
    ctx = XOF(seed_kem)
    seed_pke = ctx.get_bytes(p.seed_bytes)
    sigma    = ctx.get_bytes(p.sigma_bytes)

    ek, dk_pke = pke_keygen(seed_pke, p)

    # The KEM secret key bundles everything decaps will need: the public key
    # (to re-encrypt), the PKE secret, sigma (for implicit rejection) and the
    # original seed. ek is included so H(ek) need not be recomputed from scratch.
    dk = ek + dk_pke + sigma + seed_kem
    return ek, dk


def kem_encaps(ek: bytes, p: HQCParams) -> tuple[bytes, bytes]:
    """Return (K, ct)."""
    m    = os.urandom(p.k // 8)     # the random "message" that seeds the shared secret
    salt = os.urandom(p.salt_bytes) # extra salt hashed into G, part of the 2025 spec

    # Derive both the shared secret K and the encryption randomness theta from
    # the same hash, binding the ciphertext to (ek, m, salt): this is the FO core.
    K, theta = G(H(ek) + m + salt)

    c_pke = pke_encrypt(ek, m, theta, p)
    ct = c_pke + salt   # the salt travels with the ciphertext so decaps can recompute G
    return K, ct


def kem_decaps(dk: bytes, ct: bytes, p: HQCParams) -> bytes:
    """Return the shared secret K."""
    # Parse dk back into the four parts packed by kem_keygen.
    ek      = dk[:p.ek_bytes]
    dk_pke  = dk[p.ek_bytes : p.ek_bytes + p.seed_bytes]
    sigma   = dk[p.ek_bytes + p.seed_bytes : p.ek_bytes + p.seed_bytes + p.sigma_bytes]

    # Parse ct: the PKE ciphertext (u || v) followed by the salt.
    c_pke = ct[:p.n_bytes + p.n1n2_bytes]
    salt  = ct[p.n_bytes + p.n1n2_bytes:]

    # Step 1: decrypt to recover the candidate message m_hat (None if the code
    # could not correct the errors).
    m_hat = pke_decrypt(dk_pke, c_pke, p)
    decode_failed = m_hat is None

    if decode_failed:
        m_hat = b'\x00' * (p.k // 8)   # placeholder so the re-encryption below still runs

    # Step 2: re-derive K' and theta' and re-encrypt exactly as the sender would.
    K_prime, theta_prime = G(H(ek) + m_hat + salt)
    c_pke_prime = pke_encrypt(ek, m_hat, theta_prime, p)

    # Step 3: the fallback key, derived from the secret sigma and the full ct.
    K_bar = J(H(ek) + sigma + ct)

    # Implicit rejection (FO transform): return K_bar if the decoder failed or
    # if re-encryption does not match the received ciphertext. The check uses
    # the decode_failed flag rather than testing m_hat == 0^k because a
    # legitimate message may be all zeros.
    if decode_failed or not _ct_equal(c_pke_prime, c_pke):
        return K_bar
    return K_prime


# The *_det variants below take the randomness as an explicit argument instead
# of calling os.urandom, so the KAT tests can reproduce the reference vectors
# exactly (see tests/test_kat.py). They are otherwise identical to the public API.

def _kem_keygen_det(seed_kem: bytes, p: HQCParams) -> tuple[bytes, bytes]:
    """Deterministic variant of kem_keygen for KAT tests (explicit seed_kem)."""
    ctx = XOF(seed_kem)
    seed_pke = ctx.get_bytes(p.seed_bytes)
    sigma    = ctx.get_bytes(p.sigma_bytes)
    ek, dk_pke = pke_keygen(seed_pke, p)
    dk = ek + dk_pke + sigma + seed_kem
    return ek, dk


def _kem_encaps_det(ek: bytes, m: bytes, salt: bytes, p: HQCParams) -> tuple[bytes, bytes]:
    """Deterministic variant of kem_encaps for KAT tests (explicit m and salt)."""
    K, theta = G(H(ek) + m + salt)
    c_pke = pke_encrypt(ek, m, theta, p)
    ct = c_pke + salt
    return K, ct


def _ct_equal(a: bytes, b: bytes) -> bool:
    """Constant-time byte-string comparison.

    Compares all bytes regardless of where the first difference is, so the
    running time does not leak how much of the ciphertext matched. A timing
    side-channel here could otherwise help an attacker forge ciphertexts."""
    if len(a) != len(b):
        return False
    diff = 0
    for x, y in zip(a, b):
        diff |= x ^ y     # OR-accumulate every byte difference; stays 0 only if all bytes match
    return diff == 0

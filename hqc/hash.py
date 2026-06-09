import hashlib

# Hash functions and XOF for HQC, adapted from src/common/symmetric.c in the
# official HQC implementation (v5, 2025):
#   https://gitlab.com/pqc-hqc/hqc/-/blob/main/src/common/symmetric.c
# G/H/I/J mirror hash_g/hash_h/hash_i/hash_j; XOF mirrors xof_init/xof_get_bytes.
#
# Domain separators verified against symmetric.c. Source constants:
#   HQC_PRNG_DOMAIN=0, HQC_XOF_DOMAIN=1, HQC_G_FCT_DOMAIN=0,
#   HQC_H_FCT_DOMAIN=1, HQC_I_FCT_DOMAIN=2, HQC_J_FCT_DOMAIN=3
# H and XOF share byte 0x01 but do not collide: they use different primitives
# (SHA3-256 vs SHAKE256).
XOF_DOMAIN = bytes([0x01])   # SHAKE256(seed || 0x01)
I_DOMAIN   = bytes([0x02])   # SHA3-512(data || 0x02)
G_DOMAIN   = bytes([0x00])   # SHA3-512(H(ek) || m || salt || 0x00)
H_DOMAIN   = bytes([0x01])   # SHA3-256(ek || 0x01)
J_DOMAIN   = bytes([0x03])   # SHA3-256(H(ek) || sigma || u || v || salt || 0x03)


def G(data: bytes) -> tuple[bytes, bytes]:
    """SHA3-512(data || G_DOMAIN) -> (K: 32 bytes, theta: 32 bytes)"""
    digest = hashlib.sha3_512(data + G_DOMAIN).digest() # The + symbol concatenates bytes objects in Python.
    return digest[:32], digest[32:] # Split the 64-byte digest into two halves: first 32 -> K, last 32 -> theta.


def H(data: bytes) -> bytes:
    """SHA3-256(data || H_DOMAIN) -> 32 bytes"""
    return hashlib.sha3_256(data + H_DOMAIN).digest()


def I(data: bytes) -> tuple[bytes, bytes]:
    """SHA3-512(data || I_DOMAIN) -> (seed_dk: 32 bytes, seed_ek: 32 bytes)"""
    digest = hashlib.sha3_512(data + I_DOMAIN).digest()
    return digest[:32], digest[32:] # 64-byte digest split into two seeds: first 32 -> seed_dk, last 32 -> seed_ek.


def J(data: bytes) -> bytes:
    """SHA3-256(data || J_DOMAIN) -> 32 bytes"""
    return hashlib.sha3_256(data + J_DOMAIN).digest()


class XOF:
    """SHAKE256 stream generator seeded with a fixed key."""

    def __init__(self, seed: bytes):
        self._shake = hashlib.shake_256(seed + XOF_DOMAIN)
        self._consumed = 0

    def get_bytes(self, n: int) -> bytes:
        """Return the NEXT n bytes from the pseudorandom stream."""
        # hashlib's shake has no incremental squeeze, so we re-derive the whole
        # output up to `needed` bytes each call and slice off the part not yet
        # consumed. `_consumed` tracks the stream position to emulate xof_get_bytes.
        needed = self._consumed + n
        raw = self._shake.digest(needed)
        result = raw[self._consumed:needed]
        self._consumed = needed
        return result

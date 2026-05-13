"""
SHAKE256-based PRNG from the HQC 2025 KAT framework.

Re-implementation of prng_init / prng_get_bytes from symmetric.c.
The PRNG absorbs the 48-byte seed plus domain byte 0x00 into a SHAKE256
context and produces bytes via sequential squeezing.

Reference: src/common/symmetric.c, prng_init / prng_get_bytes
"""

import hashlib


class NIST_DRBG:
    """
    SHAKE256 PRNG used by the HQC 2025 KAT generator.
    Equivalent to prng_init(seed, NULL, 48, 0) followed by prng_get_bytes().
    """

    def __init__(self, seed: bytes):
        """seed: 48 bytes (the value in the 'seed' field of the .rsp file)."""
        assert len(seed) == 48
        self._seed = seed + bytes([0x00])   # PRNG domain separator
        self._consumed = 0

    def randombytes(self, n: int) -> bytes:
        """Return n pseudorandom bytes via sequential SHAKE256 squeezing."""
        needed = self._consumed + n
        raw = hashlib.shake_256(self._seed).digest(needed)
        result = raw[self._consumed:needed]
        self._consumed = needed
        return result

"""
Deterministic pseudorandom byte generator (DRBG) for the KAT tests.

In production, HQC draws true randomness from os.urandom (the seed in keygen,
and m + salt in encaps). To reproduce the official NIST test vectors byte for
byte, that randomness must instead be deterministic: each KAT case ships a
48-byte seed, and this class stretches it into the exact same byte stream the
reference implementation produced. Same seed in -> same bytes out, always.

It works like the XOF in hash.py: the 48-byte seed plus the PRNG domain byte
0x00 are fed into SHAKE256, and randombytes() squeezes successive chunks of
that stream, tracking how many bytes have already been handed out.

Re-implements prng_init / prng_get_bytes from src/common/symmetric.c
(HQC reference, v5 2025).
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

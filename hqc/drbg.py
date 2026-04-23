"""
SHAKE256-based PRNG del framework HQC 2025.

Reimplementación del prng_init / prng_get_bytes de symmetric.c.
El PRNG se inicializa absorbiendo la semilla de 48 bytes + dominio 0x00
en un contexto SHAKE256 incremental, y genera bytes mediante squeeze.

Referencia: src/common/symmetric.c, prng_init / prng_get_bytes
"""

import hashlib


class NIST_DRBG:
    """
    SHAKE256 PRNG usado por el generador KAT de HQC 2025.
    Equivalente a prng_init(seed, NULL, 48, 0) + prng_get_bytes().
    """

    def __init__(self, seed: bytes):
        """seed: 48 bytes (los del campo 'seed' del fichero .rsp)."""
        assert len(seed) == 48
        self._seed = seed + bytes([0x00])   # dominio PRNG = 0x00
        self._consumed = 0

    def randombytes(self, n: int) -> bytes:
        """Genera n bytes pseudoaleatorios (squeeze secuencial del SHAKE256)."""
        needed = self._consumed + n
        raw = hashlib.shake_256(self._seed).digest(needed)
        result = raw[self._consumed:needed]
        self._consumed = needed
        return result

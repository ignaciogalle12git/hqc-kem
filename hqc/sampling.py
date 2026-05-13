"""
Fixed-weight vector sampling for HQC.

Two algorithms are provided:

  sample_fixed_weight_keygen  -- for x and y in keygen (vect_generate_random_support1)
    Reads 3 bytes at a time, interprets them as a 24-bit little-endian integer,
    applies Barrett reduction mod n, and rejects candidates above the rejection
    threshold to eliminate modular bias. Duplicate indices are also discarded.
    Consumption from the XOF is variable, which is acceptable for long-lived
    secret vectors where perfect uniformity is required.

  sample_fixed_weight_encrypt -- for r1, r2, e in encrypt (vect_generate_random_support2)
    Reads 4*omega bytes at once as uint32 values, computes
    support[i] = i + (rand * (n - i)) >> 32 (partial Fisher-Yates shuffle),
    then deduplicates. Introduces a negligible bias (< 0.2%), but consumes
    a fixed number of XOF bytes, which is required for KAT reproducibility.
"""

from .hash import XOF


def sample_vect(n: int, xof: XOF) -> bytearray:
    """Return a uniformly random vector of n bits drawn from xof."""
    n_bytes = (n + 7) // 8
    raw = bytearray(xof.get_bytes(n_bytes))
    remainder = n % 8
    if remainder:
        raw[-1] &= (1 << remainder) - 1
    return raw


def _barrett_reduce(x: int, n: int, n_mu: int) -> int:
    """Barrett modular reduction: return x mod n."""
    q = (x * n_mu) >> 32
    r = x - q * n
    if r >= n:
        r -= n
    return r


def _support_to_vector(support: list[int], n: int) -> bytearray:
    """Convert a list of bit indices to a packed byte vector of n bits."""
    v = bytearray((n + 7) // 8)
    for p in support:
        v[p // 8] |= 1 << (p % 8)
    return v


def sample_fixed_weight_keygen(n: int, omega: int, xof: XOF) -> bytearray:
    """
    Sample a vector of n bits with exactly omega ones for keygen (x, y).
    Implements vect_generate_random_support1: reads 3 bytes as a 24-bit
    little-endian integer, applies Barrett reduction mod n, and rejects
    candidates at or above the rejection threshold to remove modular bias.

    The rejection threshold is the largest multiple of n that fits in 2^24,
    equal to UTILS_REJECTION_THRESHOLD in ref/parameters.h (16767881 for
    n=17669). The Barrett multiplier uses a 32-bit numerator even though
    candidates are 24-bit values.
    """
    n_mu = 2**32 // n
    rejection_threshold = (1 << 24) - ((1 << 24) % n)

    support = []
    support_set = set()
    while len(support) < omega:
        b = xof.get_bytes(3)
        candidate = b[0] | (b[1] << 8) | (b[2] << 16)
        if candidate >= rejection_threshold:
            continue
        idx = _barrett_reduce(candidate, n, n_mu)
        if idx not in support_set:
            support.append(idx)
            support_set.add(idx)

    return _support_to_vector(support, n)


def sample_fixed_weight_encrypt(n: int, omega: int, xof: XOF) -> bytearray:
    """
    Sample a vector of n bits with exactly omega ones for encrypt (r1, r2, e).
    Implements vect_generate_random_support2: reads 4*omega bytes as uint32
    little-endian values, computes support[i] = i + (rand * (n-i)) >> 32
    (Algorithm 5 from Sendrier 2021), then deduplicates by replacing any
    repeated index at position i with i itself.
    """
    raw = xof.get_bytes(4 * omega)
    rand_u32 = [
        int.from_bytes(raw[4 * i: 4 * i + 4], 'little')
        for i in range(omega)
    ]

    support = [0] * omega
    for i in range(omega):
        support[i] = i + ((rand_u32[i] * (n - i)) >> 32)

    for i in range(omega - 2, -1, -1):
        if any(support[j] == support[i] for j in range(i + 1, omega)):
            support[i] = i

    return _support_to_vector(support, n)

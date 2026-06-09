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

Adapted from src/ref/vector.c in the official HQC implementation (v5, 2025):
    https://gitlab.com/pqc-hqc/hqc/-/blob/main/src/ref/vector.c
sample_fixed_weight_keygen / sample_fixed_weight_encrypt mirror
vect_generate_random_support1 / vect_generate_random_support2.
"""

from .hash import XOF


def sample_vect(n: int, xof: XOF) -> bytearray:
    """Return a uniformly random vector of n bits drawn from xof.

    Adapted from vect_set_random() in src/ref/vector.c (HQC reference)."""
    n_bytes = (n + 7) // 8 # Round up to the nearest byte
    raw = bytearray(xof.get_bytes(n_bytes))
    remainder = n % 8
    if remainder: # Mask off unused bits in the last byte if n is not a multiple of 8.
        raw[-1] &= (1 << remainder) - 1 # For example, if remainder is 3, we want to keep only the lowest 3 bits,
    return raw                           # so we AND with 0b00000111 (which is (1 << 3) - 1).


def _barrett_reduce(x: int, n: int, n_mu: int) -> int:
    """Barrett modular reduction: return x mod n using only a multiply and a shift.

    Adapted from barrett_reduce() in src/ref/vector.c (HQC reference):
    https://gitlab.com/pqc-hqc/hqc/-/blob/main/src/ref/vector.c
    """
    q = (x * n_mu) >> 32 # floor((x * n_mu) / 2^32); n_mu = floor(2^32 / n) is precomputed,
                         # so q approximates the quotient x // n without a real division.
    r = x - q * n        # remainder candidate; the estimate can leave it one multiple of n too high
    if r >= n:           # so at most one correction step is ever needed
        r -= n
    return r


def _support_to_vector(support: list[int], n: int) -> bytearray:
    """Convert a list of bit indices (the support) to a packed byte vector of n bits.

    Adapted from vect_write_support_to_vector() in src/ref/vector.c. The C version
    is branch-free for constant-time execution; here we set each bit directly for readability.
    """
    v = bytearray((n + 7) // 8)
    for p in support:
        v[p // 8] |= 1 << (p % 8) # bit p lives in byte p//8; OR in a mask that has only bit (p % 8) set.
    return v


def sample_fixed_weight_keygen(n: int, omega: int, xof: XOF) -> bytearray:
    """
    Sample a vector of n bits with exactly omega ones, for keygen (x, y).

    Adapted from vect_generate_random_support1() in src/ref/vector.c (HQC reference):
    https://gitlab.com/pqc-hqc/hqc/-/blob/main/src/ref/vector.c
    Reads 3 bytes as a 24-bit little-endian integer, applies Barrett reduction
    mod n, and rejects candidates at or above the rejection threshold to remove
    modular bias. Duplicate indices are also discarded.

    The rejection threshold is the largest multiple of n that fits in 2^24,
    equal to UTILS_REJECTION_THRESHOLD in src/ref/hqc-*/parameters.h (16767881 for
    n=17669). The Barrett multiplier uses a 32-bit numerator even though
    candidates are 24-bit values.
    """
    n_mu = 2**32 // n
    rejection_threshold = (1 << 24) - ((1 << 24) % n) # largest multiple of n strictly below 2^24

    support = []          # chosen indices, in draw order (matches the C reference)
    support_set = set()   # mirror of `support` for O(1) duplicate checks
    while len(support) < omega:
        b = xof.get_bytes(3)
        candidate = b[0] | (b[1] << 8) | (b[2] << 16) # little-endian 24-bit value
        if candidate >= rejection_threshold:          # discard the biased tail and redraw
            continue
        idx = _barrett_reduce(candidate, n, n_mu)
        if idx not in support_set:                     # keep only distinct positions
            support.append(idx)
            support_set.add(idx)

    return _support_to_vector(support, n)


def sample_fixed_weight_encrypt(n: int, omega: int, xof: XOF) -> bytearray:
    """
    Sample a vector of n bits with exactly omega ones, for encrypt (r1, r2, e).

    Adapted from vect_generate_random_support2() in src/ref/vector.c (HQC reference):
    https://gitlab.com/pqc-hqc/hqc/-/blob/main/src/ref/vector.c
    This is Algorithm 5 from Sendrier, "Secure Sampling of Constant-Weight Words"
    (2021): reads 4*omega bytes as uint32 little-endian values, computes
    support[i] = i + (rand * (n-i)) >> 32, then deduplicates by replacing any
    repeated index at position i with i itself.
    """
    raw = xof.get_bytes(4 * omega)
    rand_u32 = [
        int.from_bytes(raw[4 * i: 4 * i + 4], 'little') # i-th 32-bit draw, little-endian
        for i in range(omega)
    ]

    support = [0] * omega
    for i in range(omega):
        # Map each draw uniformly into [i, n): the partial Fisher-Yates pick for position i.
        support[i] = i + ((rand_u32[i] * (n - i)) >> 32)

    # Deduplicate from the back: if support[i] collides with any later index, fall back to i.
    for i in range(omega - 2, -1, -1):
        if any(support[j] == support[i] for j in range(i + 1, omega)):
            support[i] = i

    return _support_to_vector(support, n)

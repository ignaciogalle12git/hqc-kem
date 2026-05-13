def get_bit(v: bytearray, i: int) -> int:
    return (v[i // 8] >> (i % 8)) & 1


def set_bit(v: bytearray, i: int) -> None:
    v[i // 8] |= 1 << (i % 8)


def poly_add(a: bytearray, b: bytearray) -> bytearray:
    # Addition in F2 is XOR. Equal lengths are enforced to catch silent
    # truncation from zip when an operand is shorter than expected.
    if len(a) != len(b):
        raise ValueError(f"poly_add: length mismatch ({len(a)} != {len(b)})")
    return bytearray(x ^ y for x, y in zip(a, b))


# ──────────────────────────────────────────────────────────────────────────────
# Naive multiplication  --  O(n^2) bit operations
# ──────────────────────────────────────────────────────────────────────────────

def poly_mul(a: bytearray, b: bytearray, n: int) -> bytearray:
    """
    Compute a * b mod (x^n - 1) in F2[x]  --  naive O(n^2) algorithm.

    For each active bit i of a, XOR b rotated by i positions into the result.
    Readable and easy to audit; too slow for n=17669 in Python (~270 ms/call).
    """
    res = bytearray((n + 7) // 8)
    for i in range(n):
        if get_bit(a, i):
            for j in range(n):
                if get_bit(b, j):
                    pos = (i + j) % n
                    res[pos // 8] ^= 1 << (pos % 8)
    return res


# ──────────────────────────────────────────────────────────────────────────────
# Karatsuba multiplication  --  O(n^1.58) 64-bit word operations
# Equivalent to the algorithm in gf2x.c from the HQC reference (PQClean).
# Reference: https://github.com/PQClean/PQClean/blob/master/crypto_kem/hqc-128/clean/gf2x.c
# ──────────────────────────────────────────────────────────────────────────────

def _base_mul(a: int, b: int) -> int:
    """
    Carry-less multiply of two 64-bit words, returning a 128-bit result.
    Equivalent to base_mul() in gf2x.c (mul1 algorithm, INRIA 2006).
    The C version uses a 4-bit lookup table; Python native integers allow
    the same loop without a table.
    """
    a &= 0xFFFFFFFFFFFFFFFF
    b &= 0xFFFFFFFFFFFFFFFF
    r = 0
    for _ in range(64):
        if b & 1:
            r ^= a
        a <<= 1
        b >>= 1
    return r


def _karatsuba(a: int, b: int, size: int) -> int:
    """
    Recursive carry-less Karatsuba over blocks of `size` 64-bit words.
    a and b are Python integers (bit i is the coefficient of x^i).
    Returns an integer of 2*size*64 bits.
    Mirrors karatsuba() in gf2x.c exactly.
    """
    if size == 1:
        return _base_mul(a, b)

    size_l = (size + 1) // 2   # words in the low half (rounded up)
    size_h = size // 2          # words in the high half

    bits_l = size_l * 64
    mask_l = (1 << bits_l) - 1

    a_lo, a_hi = a & mask_l, a >> bits_l
    b_lo, b_hi = b & mask_l, b >> bits_l

    lo  = _karatsuba(a_lo,        b_lo,        size_l)
    hi  = _karatsuba(a_hi,        b_hi,        size_h)
    mid = _karatsuba(a_lo ^ a_hi, b_lo ^ b_hi, size_l)

    return lo ^ ((mid ^ lo ^ hi) << bits_l) ^ (hi << (2 * bits_l))


def _reduce_mod(x: int, n: int) -> int:
    """Reduce x mod (x^n - 1) in F2[x]: fold coefficients at positions >= n
    back by adding them at position (pos mod n)."""
    result = x & ((1 << n) - 1)
    x >>= n
    pos = 0
    while x:
        if x & 1:
            result ^= 1 << (pos % n)
        x >>= 1
        pos += 1
    return result


def poly_mul_karatsuba(a: bytearray, b: bytearray, n: int) -> bytearray:
    """
    Compute a * b mod (x^n - 1) in F2[x]  --  Karatsuba O(n^1.58).

    Implements the same algorithm as vect_mul() + karatsuba() + base_mul()
    in gf2x.c from the HQC reference (PQClean/hqc-128/clean). The key
    difference is that this version operates on arbitrary-precision Python
    integers instead of uint64_t arrays, giving the same asymptotic complexity
    with a larger constant overhead.

    In Python the real speedup over naive is ~2.9x because interpreter overhead
    dominates the algorithmic gain. The equivalent C code is ~3000x faster.
    """
    n64 = (n + 63) // 64
    a_int = int.from_bytes(a, 'little')
    b_int = int.from_bytes(b, 'little')
    prod  = _karatsuba(a_int, b_int, n64)
    res   = _reduce_mod(prod, n)
    nb    = (n + 7) // 8
    return bytearray(res.to_bytes(nb, 'little'))


# ──────────────────────────────────────────────────────────────────────────────

def poly_truncate(v: bytearray, n: int, n1n2: int) -> bytearray:
    """Zero out bits from position n1n2 to n-1, keeping only the n1n2 low bits."""
    result = bytearray(v)
    for i in range(n1n2, n):
        result[i // 8] &= ~(1 << (i % 8))
    return result

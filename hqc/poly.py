def get_bit(v: bytearray, i: int) -> int:
    """Return bit i of v as 0 or 1 (bit i is the coefficient of x^i).

    Parameters
    ----------
    v : bytearray
        Packed bit vector, little-endian by bit.
    i : int
        Bit position to read.
    """
    return (v[i // 8] >> (i % 8)) & 1


def set_bit(v: bytearray, i: int) -> None:
    """Set bit i of v to 1, in place.

    Parameters
    ----------
    v : bytearray
        Packed bit vector, little-endian by bit (modified in place).
    i : int
        Bit position to set.
    """
    v[i // 8] |= 1 << (i % 8)


def poly_add(a: bytearray, b: bytearray) -> bytearray:
    """Add two polynomials in F2[x]: bitwise XOR of two equal-length vectors.

    Parameters
    ----------
    a : bytearray
        First operand (packed bit vector).
    b : bytearray
        Second operand, must be the same length as a.

    Returns
    -------
    bytearray
        a XOR b. Raises ValueError on a length mismatch.
    """
    # Addition in F2 is XOR. Equal lengths are enforced to catch silent
    # truncation from zip when an operand is shorter than expected.
    if len(a) != len(b):
        raise ValueError(f"poly_add: length mismatch ({len(a)} != {len(b)})")
    return bytearray(x ^ y for x, y in zip(a, b)) # The ^ operator computes the bitwise XOR of two integers in Python,
                                                  # and when applied to bytes objects, it XORs each pair of corresponding bytes.
                                                  # The zip function pairs up corresponding bytes from a and b, and the generator
                                                  # expression computes the XOR for each pair, resulting in a new bytearray that
                                                  # is the sum of the two polynomials.


# ──────────────────────────────────────────────────────────────────────────────
# Naive multiplication    O(n^2) bit operations
# ──────────────────────────────────────────────────────────────────────────────

def poly_mul(a: bytearray, b: bytearray, n: int) -> bytearray:
    """
    Compute a * b mod (x^n - 1) in F2[x]  --  naive O(n^2) algorithm.

    For each active bit i of a, XOR b rotated by i positions into the result.
    Readable and easy to audit; too slow for n=17669 in Python (~270 ms/call).

    Visual example with n = 4:  a = 0011 (x+1),  b = 0110 (x^2+x)

        active bits of a: i = 0, 1      active bits of b: j = 1, 2

        i=0, j=1  ->  pos = (0+1) % 4 = 1   res ^= 0010   res = 0010
        i=0, j=2  ->  pos = (0+2) % 4 = 2   res ^= 0100   res = 0110
        i=1, j=1  ->  pos = (1+1) % 4 = 2   res ^= 0100   res = 0010  <- cancels: 1+1=0 in F2
        i=1, j=2  ->  pos = (1+2) % 4 = 3   res ^= 1000   res = 1010

        result: 1010 = x^3 + x    ( (x+1)(x^2+x) = x^3 + 2x^2 + x, and 2x^2 = 0 )

    The ring wrap-around (mod x^n - 1): with a = 1000 (x^3) and b = 0010 (x),
    pos = (3+1) % 4 = 0, so the result is 0001 = 1, because x^4 = 1.

    Parameters
    ----------
    a : bytearray
        First operand (packed bit vector, bit i = coefficient of x^i).
    b : bytearray
        Second operand (packed bit vector).
    n : int
        Ring size; the result is reduced mod (x^n - 1).

    Returns
    -------
    bytearray
        a * b mod (x^n - 1) as a packed bit vector of ceil(n/8) bytes.
    """
    res = bytearray((n + 7) // 8)
    for i in range(n):
        if get_bit(a, i):
            for j in range(n):
                if get_bit(b, j):
                    pos = (i + j) % n
                    res[pos // 8] ^= 1 << (pos % 8) # XOR the bit at position pos in the result, since addition in F2 is XOR.
                                                    # The position pos is calculated as (i + j) % n to account for the modulo operation with x^n - 1,
                                                    #  which means that any term with degree >= n wraps around to degree (pos mod n).
    return res


# ──────────────────────────────────────────────────────────────────────────────
# Karatsuba multiplication    O(n^1.58) 64-bit word operations
# Equivalent to the algorithm in gf2x.c from the HQC reference (PQClean).
# Reference: https://github.com/PQClean/PQClean/blob/master/crypto_kem/hqc-128/clean/gf2x.c
# ──────────────────────────────────────────────────────────────────────────────

def _base_mul(a: int, b: int) -> int:
    """
    Carry-less multiply of two 64-bit words, returning a 128-bit result.
    Equivalent to base_mul() in gf2x.c (mul1 algorithm, INRIA 2006).
    The C version uses a 4-bit lookup table; Python native integers allow
    the same loop without a table.
    Minimal case of Karatsuba, used when size=1 in _karatsuba().

    Visual example with 4-bit words (the real loop uses 64). It is school
    multiplication in base 2, except the partial rows are added with XOR:

        a = 1011 (x^3+x+1),  b = 0110 (x^2+x)

              1011
            x 0110
            ------
              0000      bit 0 of b = 0  ->  add nothing
             10110      bit 1 of b = 1  ->  r ^= (a << 1)
            101100      bit 2 of b = 1  ->  r ^= (a << 2)
              0000      bit 3 of b = 0  ->  add nothing
            ------
            111010      XOR column by column, no carries

        result: 111010 = x^5+x^4+x^3+x. As plain integers 11*6 = 66 = 1000010:
        a different value, because the carries corrupt the F2 coefficients.
    """
    a &= 0xFFFFFFFFFFFFFFFF # Masking a and b with 0xFFFFFFFFFFFFFFFF ensures that only the least significant 64 bits of a and b are used in the multiplication,
    b &= 0xFFFFFFFFFFFFFFFF # effectively treating them as 64-bit unsigned integers. This is important because the carry-less multiplication algorithm operates on 64-bit words,
                            #  and any bits beyond the 64th would not be relevant to the result.
    r = 0
    for _ in range(64):
        if b & 1:
            r ^= a          # If the least significant bit of b is 1, we XOR a into the result r.
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

    size_l = (size + 1) // 2   # words in the low half (rounded up) Bytes
    size_h = size // 2          # words in the high half

    bits_l = size_l * 64       # bits in the low half; the split point between low and high halves
    mask_l = (1 << bits_l) - 1 # Mask to extract the low half of the bits from a or b. It has bits_l low bits set to 1 and the rest set to 0.
                                # For example (1 << 5) - 1 would give 0b11111, which can be used to extract the low 5 bits of a number.

    a_lo, a_hi = a & mask_l, a >> bits_l # Split a into low and high halves. a_lo gets the low bits (a & mask_l), and a_hi gets the high bits (a >> bits_l).
    b_lo, b_hi = b & mask_l, b >> bits_l

    lo  = _karatsuba(a_lo,        b_lo,        size_l)  # a_lo * b_lo
    hi  = _karatsuba(a_hi,        b_hi,        size_h)  # a_hi * b_hi
    mid = _karatsuba(a_lo ^ a_hi, b_lo ^ b_hi, size_l)  # (a_lo+a_hi)(b_lo+b_hi); the +/- of Karatsuba is XOR in F2

    # Recombine: result = lo + (mid - lo - hi)*x^bits_l + hi*x^(2*bits_l), with
    # every +/- being XOR in F2. The middle term reuses lo and hi to save one multiply.
    # Multiply by X it's equal to << bits_l, and by X^2 it's << (2*bits_l).

    return lo ^ ((mid ^ lo ^ hi) << bits_l) ^ (hi << (2 * bits_l))


def _reduce_mod(x: int, n: int) -> int:
    """Reduce x mod (x^n - 1) in F2[x]: fold coefficients at positions >= n
    back by adding them at position (pos mod n)."""
    result = x & ((1 << n) - 1)  # the low n coefficients stay where they are. X AND (low n mask) zeroes out all bits from position n and above, keeping only the low n bits of x.
    x >>= n                       # everything from x^n upward must be folded back
    pos = 0
    while x:
        if x & 1: #  Now we are on the upper bits. If the least significant bit of x is 1, it means that there is a term with degree (n + pos) in the polynomial represented by x.
                  # Since we are working modulo (x^n - 1), this term can be reduced.
            result ^= 1 << (pos % n)  # x^(n+pos) == x^(pos mod n) since x^n == 1, so XOR it in there
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

    Parameters
    ----------
    a : bytearray
        First operand (packed bit vector, bit i = coefficient of x^i).
    b : bytearray
        Second operand (packed bit vector).
    n : int
        Ring size; the result is reduced mod (x^n - 1).

    Returns
    -------
    bytearray
        a * b mod (x^n - 1) as a packed bit vector of ceil(n/8) bytes,
        identical to poly_mul(a, b, n).
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
    """Zero out bits from position n1n2 to n-1, keeping only the n1n2 low bits.

    Parameters
    ----------
    v : bytearray
        Packed bit vector of n bits.
    n : int
        Total bit length of v.
    n1n2 : int
        Number of low bits to keep (the code length n1*n2).

    Returns
    -------
    bytearray
        A copy of v with bits [n1n2, n) cleared.
    """
    result = bytearray(v)
    for i in range(n1n2, n):
        result[i // 8] &= ~(1 << (i % 8))
    return result

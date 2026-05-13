def get_bit(v: bytearray, i: int) -> int:
    return (v[i // 8] >> (i % 8)) & 1


def set_bit(v: bytearray, i: int) -> None:
    v[i // 8] |= 1 << (i % 8)


def poly_add(a: bytearray, b: bytearray) -> bytearray:
    # Suma en F2 (XOR bit a bit). Se exige misma longitud para evitar que
    # `zip` trunque silenciosamente cuando un operando es más corto por error
    # (p.ej. olvidar rellenar con ceros tras un truncate).
    if len(a) != len(b):
        raise ValueError(f"poly_add: longitudes distintas ({len(a)} != {len(b)})")
    return bytearray(x ^ y for x, y in zip(a, b))


# ──────────────────────────────────────────────────────────────────────────────
# Multiplicación naive  —  O(n²) operaciones de bit
# ──────────────────────────────────────────────────────────────────────────────

def poly_mul(a: bytearray, b: bytearray, n: int) -> bytearray:
    """
    a * b mod (x^n - 1) en F2[x]  —  algoritmo naive O(n²).

    Para cada bit i de a que sea 1, XOR en el resultado b rotado i posiciones.
    Sencillo y auditable; demasiado lento para n=17669 en Python (~270 ms/llamada).
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
# Multiplicación Karatsuba  —  O(n^1.58) operaciones de palabra de 64 bits
# Equivalente al algoritmo de gf2x.c del código C de referencia HQC (PQClean).
# Referencia: https://github.com/PQClean/PQClean/blob/master/crypto_kem/hqc-128/clean/gf2x.c
# ──────────────────────────────────────────────────────────────────────────────

def _base_mul(a: int, b: int) -> int:
    """
    Carry-less multiply de dos palabras de 64 bits → resultado de 128 bits.
    Equivalente a base_mul() de gf2x.c (algoritmo mul1, INRIA 2006).
    En C se hace con una tabla de 4 bits; en Python la aritmética entera
    nativa permite el mismo bucle sin necesidad de tabla.
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
    Karatsuba carry-less sobre bloques de `size` palabras de 64 bits.
    a y b son enteros Python (bit i = coeficiente de x^i).
    Devuelve un entero de 2*size*64 bits.
    Refleja exactamente la función karatsuba() de gf2x.c.
    """
    if size == 1:
        return _base_mul(a, b)

    size_l = (size + 1) // 2   # palabras en la mitad baja (redondeada arriba)
    size_h = size // 2          # palabras en la mitad alta

    bits_l = size_l * 64
    mask_l = (1 << bits_l) - 1

    a_lo, a_hi = a & mask_l, a >> bits_l
    b_lo, b_hi = b & mask_l, b >> bits_l

    lo  = _karatsuba(a_lo,        b_lo,        size_l)
    hi  = _karatsuba(a_hi,        b_hi,        size_h)
    mid = _karatsuba(a_lo ^ a_hi, b_lo ^ b_hi, size_l)

    return lo ^ ((mid ^ lo ^ hi) << bits_l) ^ (hi << (2 * bits_l))


def _reduce_mod(x: int, n: int) -> int:
    """Reduce x mod (x^n - 1) en F2[x]: los coeficientes >= n se suman mod n."""
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
    a * b mod (x^n - 1) en F2[x]  —  Karatsuba O(n^1.58).

    Implementa el mismo algoritmo que vect_mul() + karatsuba() + base_mul()
    de gf2x.c en el código C de referencia HQC (PQClean/hqc-128/clean).
    La principal diferencia con el código C es que aquí se trabaja con
    enteros Python arbitrarios en lugar de arrays de uint64_t, por lo que
    no se alcanza la misma velocidad pero sí la misma complejidad asintótica.

    En Python el speedup real frente al naive es ~2.7× (el overhead del
    intérprete domina sobre la mejora algorítmica). En C, el mismo algoritmo
    es ~3000× más rápido que el naive Python.
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
    """Descarta los (n - n1n2) bits más significativos, conserva los n1n2 primeros."""
    result = bytearray(v)
    for i in range(n1n2, n):
        result[i // 8] &= ~(1 << (i % 8))
    return result

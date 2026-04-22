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


def poly_mul(a: bytearray, b: bytearray, n: int) -> bytearray:
    """
    res = a * b mod (x^n - 1) en F2[x]
    Para cada bit i de a que sea 1, sumar (XOR) b rotado i posiciones.
    Rotación de b en i posiciones: bit j de b -> bit (j+i) mod n de res.
    """
    res = bytearray((n + 7) // 8)
    for i in range(n):
        if get_bit(a, i):
            for j in range(n):
                if get_bit(b, j):
                    pos = (i + j) % n
                    res[pos // 8] ^= 1 << (pos % 8)
    return res


def poly_truncate(v: bytearray, n: int, n1n2: int) -> bytearray:
    """Descarta los (n - n1n2) bits más significativos, conserva los n1n2 primeros."""
    result = bytearray(v)
    for i in range(n1n2, n):
        result[i // 8] &= ~(1 << (i % 8))
    return result

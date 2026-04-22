"""
Muestreo de vectores aleatorios para HQC.

Tenemos dos algoritmos de muestreo de peso fijo:
  - sample_fixed_weight_keygen: para x e y en keygen (vect_sample_fixed_weight1)
    Lee 3 bytes a la vez, reducción Barrett, rechazo de duplicados.
  - sample_fixed_weight_encrypt: para r1, r2, e en cifrado (vect_sample_fixed_weight2)
    Lee 4·omega bytes de golpe, soporte[i] = i + (rand32·(n-i)) >> 32,
    más un paso de deduplicación.

"""

from .hash import XOF


def sample_vect(n: int, xof: XOF) -> bytearray:
    """Vector uniforme de n bits a partir de un XOF."""
    n_bytes = (n + 7) // 8
    raw = bytearray(xof.get_bytes(n_bytes))
    remainder = n % 8
    if remainder:
        raw[-1] &= (1 << remainder) - 1
    return raw


def _barrett_reduce(x: int, n: int, n_mu: int) -> int:
    """Reducción modular de Barrett: devuelve x mod n."""
    q = (x * n_mu) >> 32
    r = x - q * n
    if r >= n:
        r -= n
    return r


def _support_to_vector(support: list[int], n: int) -> bytearray:
    """Convierte una lista de índices de bits a un vector de bytes."""
    v = bytearray((n + 7) // 8)
    for p in support:
        v[p // 8] |= 1 << (p % 8)
    return v


def sample_fixed_weight_keygen(n: int, omega: int, xof: XOF) -> bytearray:
    """
    Vector de n bits con exactamente omega unos para keygen (x, y).
    Implementa vect_generate_random_support1: muestrea 3 bytes → valor 24-bit
    little-endian → reducción Barrett → rechazo si duplicado.

    El umbral de rechazo se calcula sobre el dominio real del candidato
    (24 bits = [0, 2^24)); es el mayor múltiplo de n que cabe en 2^24, lo que
    elimina el sesgo residual de la reducción modular. Para n = 17669 vale
    16767881, valor fijado como UTILS_REJECTION_THRESHOLD en ref/parameters.h.
    """
    # n_mu es el multiplicador Barrett (numerador 2^32): aunque el candidato
    # viva en 24 bits, _barrett_reduce desplaza 32 bits tras multiplicar.
    n_mu = 2**32 // n
    rejection_threshold = (1 << 24) - ((1 << 24) % n)

    support = []
    support_set = set()
    while len(support) < omega:
        b = xof.get_bytes(3)
        candidate = b[0] | (b[1] << 8) | (b[2] << 16)   # 24 bits little-endian
        if candidate >= rejection_threshold:
            continue
        idx = _barrett_reduce(candidate, n, n_mu)
        if idx not in support_set:
            support.append(idx)
            support_set.add(idx)

    return _support_to_vector(support, n)


def sample_fixed_weight_encrypt(n: int, omega: int, xof: XOF) -> bytearray:
    """
    Vector de n bits con exactamente omega unos para cifrado (r1, r2, e).
    Implementa vect_generate_random_support2: lee 4*omega bytes como uint32_t
    little-endian, soporte[i] = i + (rand * (n-i)) >> 32, y deduplicación.
    """
    raw = xof.get_bytes(4 * omega)
    rand_u32 = [
        int.from_bytes(raw[4 * i: 4 * i + 4], 'little')
        for i in range(omega)
    ]

    support = [0] * omega
    for i in range(omega):
        buff = rand_u32[i]
        support[i] = i + ((buff * (n - i)) >> 32)

    # Deduplicación: para cada i de omega-2 a 0, si support[i] ya aparece
    # en support[i+1..omega-1], se reemplaza por i.
    for i in range(omega - 2, -1, -1):
        found = any(support[j] == support[i] for j in range(i + 1, omega))
        if found:
            support[i] = i

    return _support_to_vector(support, n)

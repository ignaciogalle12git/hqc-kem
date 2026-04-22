from dataclasses import dataclass


@dataclass(frozen=True)
class HQCParams:
    n: int          # longitud del anillo
    k: int          # bits del mensaje (también |K| y |sigma| en bytes = k//8)
    omega: int      # peso de los vectores privados x, y
    omega_r: int    # peso de los vectores de ruido r1, r2
    omega_e: int    # peso del vector de ruido e (= omega_r en todas las variantes)
    delta: int      # capacidad correctora Reed-Solomon
    n1: int         # longitud código RS externo
    n2: int         # longitud código RM interno
    seed_bytes: int = 32
    salt_bytes: int = 16

    @property
    def sigma_bytes(self) -> int:   # |sigma| = k/8 bytes (VEC_K_SIZE_BYTES en ref C)
        return self.k // 8

    @property
    def ell(self) -> int:
        return self.n - self.n1 * self.n2

    @property
    def n_bytes(self) -> int:
        return (self.n + 7) // 8

    @property
    def n1n2_bytes(self) -> int:
        return (self.n1 * self.n2 + 7) // 8

    @property
    def ek_bytes(self) -> int:   # seed_ek (32) + s (n_bytes)
        return self.seed_bytes + self.n_bytes

    @property
    def dk_bytes(self) -> int:   # ek + seed_dk (32) + sigma (sigma_bytes) + seedKEM (32)
        return self.ek_bytes + self.seed_bytes + self.sigma_bytes + self.seed_bytes

    @property
    def ct_bytes(self) -> int:   # u (n_bytes) + v (n1n2_bytes) + salt (16)
        return self.n_bytes + self.n1n2_bytes + self.salt_bytes


HQC1 = HQCParams(n=17669, k=128, omega=66, omega_r=75, omega_e=75,
                 delta=15, n1=46, n2=384)

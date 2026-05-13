"""
Validación contra los vectores KAT oficiales de HQC-1.

El fichero PQCkemKAT_2321.rsp fue generado por la implementación de
referencia usando el SHAKE256 DRBG del framework NIST PQC KAT (spec 2025).
Para reproducirlo se inicializa el mismo DRBG con la semilla del vector
y se generan los mismos bytes aleatorios que usaría la referencia:
  - 32 bytes → seed_kem  (para keygen)
  - 16 bytes → m         (para encaps)
  - 16 bytes → salt      (para encaps)
"""

import os
import pytest
from hqc.params import HQC1
from hqc.kem import kem_decaps, _kem_keygen_det, _kem_encaps_det
from hqc.drbg import NIST_DRBG

KAT_FILE = os.path.join(os.path.dirname(__file__), '..', 'kat', 'PQCkemKAT_2321.rsp')

# Número de vectores a testear. Sobreescribible con: KAT_N=5 pytest ...
_KAT_N = int(os.environ.get('KAT_N', '10'))


def parse_kat(filepath: str) -> list[dict]:
    """Parser del formato count/seed/pk/sk/ct/ss del framework NIST."""
    vectors = []
    current = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, val = line.split('=', 1)
                key = key.strip()
                val = val.strip()
                if key == 'count':
                    current[key] = int(val)
                else:
                    current[key] = bytes.fromhex(val)
            if 'ss' in current and len(current) >= 6:
                vectors.append(current)
                current = {}
    return vectors


@pytest.mark.skipif(not os.path.exists(KAT_FILE), reason="KAT file not found")
@pytest.mark.parametrize("idx", range(_KAT_N))
def test_kat_hqc1_keygen(idx):
    """pk y sk coinciden con el vector KAT."""
    vectors = parse_kat(KAT_FILE)
    v = vectors[idx]

    drbg = NIST_DRBG(v['seed'])
    seed_kem = drbg.randombytes(HQC1.seed_bytes)   # 32 bytes

    ek, dk = _kem_keygen_det(seed_kem, HQC1)

    assert ek == v['pk'], f"KAT[{idx}]: pk no coincide"
    assert dk == v['sk'], f"KAT[{idx}]: sk no coincide"


@pytest.mark.skipif(not os.path.exists(KAT_FILE), reason="KAT file not found")
@pytest.mark.parametrize("idx", range(_KAT_N))
def test_kat_hqc1_encaps(idx):
    """ct y ss coinciden con el vector KAT."""
    vectors = parse_kat(KAT_FILE)
    v = vectors[idx]

    drbg = NIST_DRBG(v['seed'])
    seed_kem = drbg.randombytes(HQC1.seed_bytes)   # 32 bytes (keygen)
    m        = drbg.randombytes(HQC1.k // 8)       # 16 bytes (encaps)
    salt     = drbg.randombytes(HQC1.salt_bytes)   # 16 bytes (encaps)

    ek, _dk = _kem_keygen_det(seed_kem, HQC1)
    K, ct   = _kem_encaps_det(ek, m, salt, HQC1)

    assert ct == v['ct'], f"KAT[{idx}]: ct no coincide"
    assert K  == v['ss'], f"KAT[{idx}]: ss no coincide"


@pytest.mark.skipif(not os.path.exists(KAT_FILE), reason="KAT file not found")
@pytest.mark.parametrize("idx", range(_KAT_N))
def test_kat_hqc1_decaps(idx):
    """decaps sobre el ct del KAT produce el mismo ss."""
    vectors = parse_kat(KAT_FILE)
    v = vectors[idx]

    K = kem_decaps(v['sk'], v['ct'], HQC1)
    assert K == v['ss'], f"KAT[{idx}]: decaps ss no coincide"

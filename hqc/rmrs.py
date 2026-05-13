import ctypes
import os

_lib_path = os.path.join(os.path.dirname(__file__), '..', 'ref', 'librmrs.so')
_lib_path = os.path.abspath(_lib_path)

try:
    _rmrs = ctypes.CDLL(_lib_path)
except OSError:
    raise ImportError(
        f"ref/librmrs.so not found at {_lib_path}.\n"
        "Build it from the reference C sources with:\n"
        "  make ref/librmrs.so"
    )

# Function signatures from rmrs_wrapper.c:
# void hqc_encode(uint8_t *cdw, const uint8_t *msg)
# int  hqc_decode(uint8_t *msg, const uint8_t *cdw)
_rmrs.hqc_encode.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_rmrs.hqc_encode.restype  = None
_rmrs.hqc_decode.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_rmrs.hqc_decode.restype  = ctypes.c_int   # 0 on success


def encode(m: bytes, n1n2_bytes: int) -> bytes:
    """Encode m (k/8 bytes) into a codeword of n1*n2 bits (n1n2_bytes)."""
    codeword = ctypes.create_string_buffer(n1n2_bytes)
    _rmrs.hqc_encode(codeword, m)
    return bytes(codeword)


def decode(codeword: bytes, k_bytes: int) -> bytes | None:
    """Decode a noisy codeword (n1*n2 bits) and recover m (k/8 bytes).
    Returns m on success, or None if the decoder fails."""
    m = ctypes.create_string_buffer(k_bytes)
    result = _rmrs.hqc_decode(m, codeword)
    if result != 0:
        return None
    return bytes(m)

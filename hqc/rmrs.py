import ctypes
import os

_lib_path = os.path.join(os.path.dirname(__file__), '..', 'ref', 'librmrs.so')
_lib_path = os.path.abspath(_lib_path)

try:
    _rmrs = ctypes.CDLL(_lib_path)
except OSError:
    raise ImportError(
        f"No se encontró ref/librmrs.so en {_lib_path}.\n"
        "Descarga los fuentes de referencia de https://pqc-hqc.org y ejecuta:\n"
        "  make ref/librmrs.so"
    )

# Firmas del wrapper (rmrs_wrapper.c)
# void hqc_encode(uint8_t *cdw, const uint8_t *msg)
# int  hqc_decode(uint8_t *msg, const uint8_t *cdw)
_rmrs.hqc_encode.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_rmrs.hqc_encode.restype  = None
_rmrs.hqc_decode.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_rmrs.hqc_decode.restype  = ctypes.c_int   # 0 = OK


def encode(m: bytes, n1n2_bytes: int) -> bytes:
    """
    Codifica m (k/8 bytes) en una palabra de código (n1*n2 bits = n1n2_bytes).
    Devuelve los n1n2_bytes del codeword.
    """
    codeword = ctypes.create_string_buffer(n1n2_bytes)
    _rmrs.hqc_encode(codeword, m)
    return bytes(codeword)


def decode(codeword: bytes, k_bytes: int) -> bytes | None:
    """
    Decodifica un codeword ruidoso (n1*n2 bits) y recupera m (k/8 bytes).
    Devuelve m si tiene éxito, None si el decoder falla.
    """
    m = ctypes.create_string_buffer(k_bytes)
    result = _rmrs.hqc_decode(m, codeword)
    if result != 0:
        return None
    return bytes(m)

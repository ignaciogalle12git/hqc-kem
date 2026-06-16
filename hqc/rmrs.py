"""
RMRS = the concatenated error-correcting code HQC uses to encode messages: a
Reed-Solomon outer code over a Reed-Muller inner code. encode() adds redundancy
so the receiver can recover m even after the s*r2 + e noise added in pke_encrypt.

Unlike the rest of the package, this is a ctypes wrapper around the reference C
decoder (compiled into ref/librmrs.so), not a re-implementation. The RS/RM
decoder is intricate and constant-time-sensitive, so reusing the audited C code
keeps it as a trustworthy ground truth; the Python layer only marshals bytes.

The C side wraps code_encode / code_decode from src/common/code.c in the official
HQC implementation (v5, 2025):
    https://gitlab.com/pqc-hqc/hqc/-/blob/main/src/common/code.c
"""

import ctypes
import os

# Locate the shared library
_lib_path = os.path.join(os.path.dirname(__file__), '..', 'ref', 'librmrs.so')
_lib_path = os.path.abspath(_lib_path)

try:
    _rmrs = ctypes.CDLL(_lib_path)
except OSError:
    # Fail loudly with the build command rather than crashing later at call time.
    raise ImportError(
        f"ref/librmrs.so not found at {_lib_path}.\n"
        "Build it from the reference C sources with:\n"
        "  make ref/librmrs.so"
    )

# Declare the C signatures so ctypes marshals arguments correctly.

# Function signatures from rmrs_wrapper.c:
# void hqc_encode(uint8_t *cdw, const uint8_t *msg)
# int  hqc_decode(uint8_t *msg, const uint8_t *cdw)
_rmrs.hqc_encode.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_rmrs.hqc_encode.restype  = None
_rmrs.hqc_decode.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_rmrs.hqc_decode.restype  = ctypes.c_int   # 0 on success, non-zero on decode failure


def encode(m: bytes, n1n2_bytes: int) -> bytes:
    """Encode m (k/8 bytes) into a codeword of n1*n2 bits (n1n2_bytes)."""
    codeword = ctypes.create_string_buffer(n1n2_bytes) # Create a mutable buffer for the codeword of the specified size (n1n2_bytes).
    _rmrs.hqc_encode(codeword, m)                      # Call the C function hqc_encode to encode the message m into the codeword buffer. The C function modifies the codeword buffer in place.
    return bytes(codeword)                             # Return the codeword as a bytes object, which is immutable and suitable for further processing in Python.


def decode(codeword: bytes, k_bytes: int) -> bytes | None:
    """Decode a noisy codeword (n1*n2 bits) and recover m (k/8 bytes).
    Returns m on success, or None if the decoder fails."""
    m = ctypes.create_string_buffer(k_bytes)
    result = _rmrs.hqc_decode(m, codeword)
    if result != 0:        # non-zero return = too many errors to correct
        return None
    return bytes(m)

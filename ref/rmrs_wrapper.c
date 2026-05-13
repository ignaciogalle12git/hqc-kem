/**
 * @file rmrs_wrapper.c
 * @brief Wrapper C para el encoder/decoder concatenado Reed-Solomon + Reed-Muller.
 *
 * Expone dos funciones con tipos simples (uint8_t*) para llamar desde Python
 * via ctypes. El pipeline sigue la especificación HQC:
 *   encode: msg -> RS encode -> RM encode -> codeword
 *   decode: codeword -> RM decode -> RS decode -> msg
 */

#include <stdint.h>
#include <string.h>
#include "reed_muller.h"
#include "reed_solomon.h"
#include "parameters.h"

/**
 * Codifica msg (PARAM_K bytes) al codeword RMRS (VEC_N1N2_SIZE_BYTES bytes).
 */
void hqc_encode(uint8_t *cdw, const uint8_t *msg) {
    uint64_t rs_cdw[VEC_N1_SIZE_64];
    uint64_t rmrs_cdw[VEC_N1N2_SIZE_64];

    memset(rs_cdw,   0, sizeof(rs_cdw));
    memset(rmrs_cdw, 0, sizeof(rmrs_cdw));

    reed_solomon_encode(rs_cdw,   (const uint64_t *)msg);
    reed_muller_encode (rmrs_cdw, rs_cdw);

    memcpy(cdw, rmrs_cdw, VEC_N1N2_SIZE_BYTES);
}

/**
 * Decodifica cdw (VEC_N1N2_SIZE_BYTES bytes) a msg (PARAM_K bytes).
 * Devuelve 0 siempre (reed_solomon_decode no expone estado de error en la ref).
 */
int hqc_decode(uint8_t *msg, const uint8_t *cdw) {
    uint64_t rs_cdw[VEC_N1_SIZE_64];
    uint64_t msg64[CEIL_DIVIDE(PARAM_K, 8)];

    memset(rs_cdw, 0, sizeof(rs_cdw));
    memset(msg64,  0, sizeof(msg64));

    reed_muller_decode(rs_cdw, (const uint64_t *)cdw);
    reed_solomon_decode(msg64, rs_cdw);

    memcpy(msg, msg64, PARAM_K);
    return 0;
}

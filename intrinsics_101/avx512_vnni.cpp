#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <chrono>

int main() {
    // データの初期化
    __m512i a = _mm512_set1_epi8((uint8_t)2);  // 実質的に符号なし
    __m512i b = _mm512_set1_epi8((uint8_t)3);  // 実質的に符号なし
    __m512i c = _mm512_setzero_si512();

    // -------------------------------
    // VNNI使用コード
    // -------------------------------
    auto start_vnni = std::chrono::high_resolution_clock::now();
    __m512i result_vnni = _mm512_dpbusd_epi32(c, a, b);
    auto end_vnni = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> duration_vnni = end_vnni - start_vnni;

    // -------------------------------
    // VNNI未使用コード
    // -------------------------------
    auto start_non_vnni = std::chrono::high_resolution_clock::now();

    // INT8 → INT16 拡張
    __m256i a_lo_256 = _mm512_extracti64x4_epi64(a, 0);
    __m256i b_lo_256 = _mm512_extracti64x4_epi64(b, 0);
    __m512i a_lo_16 = _mm512_cvtepi8_epi16(a_lo_256);
    __m512i b_lo_16 = _mm512_cvtepu8_epi16(b_lo_256);

    // 乗算
    __m512i mul_lo_16 = _mm512_mullo_epi16(a_lo_16, b_lo_16);

    // INT16 → INT32 拡張
    __m256i mul_lo_256 = _mm512_extracti64x4_epi64(mul_lo_16, 0);
    __m512i mul_lo_32 = _mm512_cvtepi16_epi32(mul_lo_256);

    // 加算
    __m512i result_non_vnni = _mm512_add_epi32(c, mul_lo_32);

    auto end_non_vnni = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> duration_non_vnni = end_non_vnni - start_non_vnni;

    // -------------------------------
    // 結果表示
    // -------------------------------
    int32_t output_vnni[16];
    int32_t output_non_vnni[16];
    _mm512_storeu_si512(output_vnni, result_vnni);
    _mm512_storeu_si512(output_non_vnni, result_non_vnni);

    printf("VNNI result[0] = %d\n", output_vnni[0]);
    printf("Non-VNNI result[0] = %d\n", output_non_vnni[0]);
    printf("VNNI execution time: %.2f ns\n", duration_vnni.count());
    printf("Non-VNNI execution time: %.2f ns\n", duration_non_vnni.count());

    return 0;
}

#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <chrono>
#include <vector>
#include <cassert>

constexpr int NUM_BLOCKS = 1024;  // 64KB相当のデータ
constexpr int LOOP_COUNT = 1000;  // ベンチマークループ回数

int main() {
    std::vector<__m512i> a_blocks(NUM_BLOCKS), b_blocks(NUM_BLOCKS), c_blocks(NUM_BLOCKS);
    std::vector<__m512i> result_vnni_blocks(NUM_BLOCKS);
    std::vector<int32_t> result_manual_blocks(NUM_BLOCKS * 16, 0);  // 16 INT32 per block

    // データ初期化
    for (int i = 0; i < NUM_BLOCKS; ++i) {
        a_blocks[i] = _mm512_set1_epi8((uint8_t)2);  // unsigned
        b_blocks[i] = _mm512_set1_epi8((int8_t)3);   // signed
        c_blocks[i] = _mm512_setzero_si512();
    }

    // -------------------------------
    // VNNI使用コード
    // -------------------------------
    auto start_vnni = std::chrono::high_resolution_clock::now();
    for (int loop = 0; loop < LOOP_COUNT; ++loop) {
        for (int i = 0; i < NUM_BLOCKS; ++i) {
            result_vnni_blocks[i] = _mm512_dpbusd_epi32(c_blocks[i], a_blocks[i], b_blocks[i]);
        }
    }
    auto end_vnni = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> duration_vnni = end_vnni - start_vnni;

    // -------------------------------
    // 非VNNIコード（手動積和演算）
    // -------------------------------
    auto start_manual = std::chrono::high_resolution_clock::now();
    for (int loop = 0; loop < LOOP_COUNT; ++loop) {
        for (int i = 0; i < NUM_BLOCKS; ++i) {
            alignas(64) uint8_t a_bytes[64];
            alignas(64) int8_t b_bytes[64];
            _mm512_store_si512(a_bytes, a_blocks[i]);
            _mm512_store_si512(b_bytes, b_blocks[i]);

            for (int j = 0; j < 16; ++j) {
                int32_t sum = 0;
                for (int k = 0; k < 4; ++k) {
                    int idx = j * 4 + k;
                    sum += static_cast<uint8_t>(a_bytes[idx]) * static_cast<int8_t>(b_bytes[idx]);
                }
                result_manual_blocks[i * 16 + j] = sum;
            }
        }
    }
    auto end_manual = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::nano> duration_manual = end_manual - start_manual;

    // -------------------------------
    // 結果表示と検証
    // -------------------------------
    int32_t output_vnni[16];
    _mm512_storeu_si512(output_vnni, result_vnni_blocks[0]);

    printf("VNNI result[0] = %d\n", output_vnni[0]);
    printf("Manual result[0] = %d\n", result_manual_blocks[0]);
    printf("VNNI total time: %.2f ms\n", duration_vnni.count() / 1e6);
    printf("Manual total time: %.2f ms\n", duration_manual.count() / 1e6);
    printf("VNNI avg time per loop: %.2f ns\n", duration_vnni.count() / LOOP_COUNT);
    printf("Manual avg time per loop: %.2f ns\n", duration_manual.count() / LOOP_COUNT);

    // 結果の一致確認
    for (int i = 0; i < NUM_BLOCKS; ++i) {
        int32_t vnni_out[16];
        _mm512_storeu_si512(vnni_out, result_vnni_blocks[i]);
        for (int j = 0; j < 16; ++j) {
            assert(vnni_out[j] == result_manual_blocks[i * 16 + j]);
        }
    }
    printf("✅ All results match between VNNI and manual implementation.\n");

    return 0;
}

#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/syscall.h>
#include <iostream>
#include <chrono>

#define TILE_ROWS 16
#define TILE_COLS 64
#define MATRIX_SIZE 1024 // need to be multiples of 64
// #define MATRIX_SIZE 2048 // need to be multiples of 64
// #define MATRIX_SIZE 4096 // need to be multiples of 64

#define LOOP_COUNT 1000

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

struct __tile_config {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];
    uint8_t rows[16];
};

bool set_tiledata_use() {
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        std::cerr << "Failed to enable AMX tile data use." << std::endl;
        return false;
    }
    return true;
}

void init_tile_config(__tile_config* tileinfo) {
    tileinfo->palette_id = 1;
    tileinfo->start_row = 0;

    // only tile0 tile1 and tile2 will be used
    for (int i = 0; i < 1; ++i)
    {
        tileinfo->colsb[i] = TILE_ROWS;
        tileinfo->rows[i] =  TILE_ROWS;
    }

    for (int i = 1; i < 4; ++i)
    {
        tileinfo->colsb[i] = TILE_COLS;
        tileinfo->rows[i] =  TILE_ROWS;
    }

    _tile_loadconfig(tileinfo);
}

void init_buffer(int8_t* buf, int8_t val) {
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i)
        buf[i] = val;
}

void init_buffer32(int32_t* buf, int32_t val) {
    for (int i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i)
        buf[i] = val;
}

void run_amx(int8_t* A, int8_t* B, int32_t* C) {
    for (int i = 0; i < MATRIX_SIZE; i += TILE_ROWS) {
        for (int j = 0; j < MATRIX_SIZE; j += TILE_COLS) {
            int32_t* c_ptr = &C[i * MATRIX_SIZE + j];    // A: i行目から
            _tile_loadd(1, c_ptr, MATRIX_SIZE);

            for (int k = 0; k < MATRIX_SIZE; k += TILE_COLS) {
                int8_t* a_ptr = &A[i * MATRIX_SIZE + k];    // A: i行目から
                int8_t* b_ptr = &B[j * MATRIX_SIZE + k];    // B: 転置アクセス

                _tile_loadd(2, a_ptr, MATRIX_SIZE);         // Aブロック（行優先）
                _tile_loadd(3, b_ptr, MATRIX_SIZE);         // Bブロック（列優先）

                _tile_dpbssd(1, 2, 3);                      // 積和演算
            }

            _tile_stored(1, c_ptr, MATRIX_SIZE); // 結果を c_ptr に保存

        }
    }
}

void run_avx512_vnni(int8_t* A, int8_t* B, int32_t* C) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            __m512i acc = _mm512_setzero_si512();
            for (int k = 0; k < MATRIX_SIZE; k += 64) {
                // Aの1行から64要素をロード
                const __m512i vec_a = _mm512_loadu_si512((const __m512i*)(A + i * MATRIX_SIZE + k));
                // Bの1列から64要素をロード（Bは転置されていると仮定）
                const __m512i vec_b = _mm512_loadu_si512((const __m512i*)(B + j * MATRIX_SIZE + k));
                acc = _mm512_dpbusd_epi32(acc, vec_a, vec_b);
            }
            C[i * MATRIX_SIZE + j] = _mm512_reduce_add_epi32(acc);
        }
    }
}

int main() {
    if (!set_tiledata_use()) return -1;

    alignas(64) int8_t A[MATRIX_SIZE * MATRIX_SIZE];
    alignas(64) int8_t B[MATRIX_SIZE * MATRIX_SIZE];
    alignas(64) int32_t C_amx[MATRIX_SIZE * MATRIX_SIZE], C_vnni[MATRIX_SIZE * MATRIX_SIZE];

    init_buffer(A, 1);
    init_buffer(B, 1);

    using namespace std::chrono;

    // AMX timing
    __tile_config cfg = {};
    init_tile_config(&cfg);
    auto start_amx = high_resolution_clock::now();
    for (int i = 0; i < LOOP_COUNT; ++i) {
        init_buffer32(C_amx, 0);
        run_amx(A, B, C_amx);
    }
    _tile_release();
    auto end_amx = high_resolution_clock::now();
    auto amx_time_ms = duration_cast<std::chrono::milliseconds>(end_amx - start_amx).count();

    for (int i = 0; i < 64; ++i) {
        // if (C_amx[i] != 1024) 
        std::cout << "Wrong sum: " << C_amx[i] << std::endl;
    }
    std::cout << "AMX Time (1000 loops)        : " << amx_time_ms << " ms" << std::endl;

    // AVX512 VNNI timing
    auto start_vnni = high_resolution_clock::now();
    for (int i = 0; i < LOOP_COUNT; ++i) {
        init_buffer32(C_vnni, 0);
        run_avx512_vnni(A, B, C_vnni);
    }
    auto end_vnni = high_resolution_clock::now();
    auto vnni_time_ms = duration_cast<std::chrono::milliseconds>(end_vnni - start_vnni).count();

    for (int i = 0; i < MATRIX_SIZE*MATRIX_SIZE; ++i) {
        if (C_vnni[i] != 1024) std::cout << "Wrong sum: " << C_vnni[i] << std::endl;
    }
    std::cout << "AVX512 VNNI Time (1000 loops): " << vnni_time_ms << " ms" << std::endl;

    return 0;
}

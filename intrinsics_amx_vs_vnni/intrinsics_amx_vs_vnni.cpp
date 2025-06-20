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
#define MAX 1024

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

    for (int i = 0; i < 8; ++i) {
        tileinfo->colsb[i] = TILE_COLS;
        tileinfo->rows[i] = TILE_ROWS;
    }

    _tile_loadconfig(tileinfo);
}

void init_buffer(int8_t* buf, int8_t val) {
    for (int i = 0; i < MAX * MAX; ++i)
        buf[i] = val;
}

void init_buffer32(int32_t* buf, int32_t val) {
    for (int i = 0; i < MAX * MAX; ++i)
        buf[i] = val;
}

void run_amx(int8_t* A, int8_t* B, int32_t* C) {
    __tile_config cfg = {};
    init_tile_config(&cfg);

    for (int i = 0; i < MAX; i += TILE_ROWS) {
        for (int j = 0; j < MAX; j += TILE_ROWS) {
            for (int k = 0; k < MAX; k += TILE_COLS) {
                _tile_loadd(1, &C[i * MAX + j], MAX * sizeof(int32_t));
                _tile_loadd(2, &A[i * MAX + k], MAX);
                _tile_loadd(3, &B[j * MAX + k], MAX);
                _tile_dpbssd(1, 2, 3);
                _tile_stored(1, &C[i * MAX + j], MAX * sizeof(int32_t));
            }
        }
    }
    _tile_release();
}

void run_avx512_vnni(int8_t* A, int8_t* B, int32_t* C) {
    for (int i = 0; i < MAX; i += TILE_ROWS) {
        for (int j = 0; j < MAX; j += TILE_ROWS) {
            __m512i acc = _mm512_setzero_si512();
            for (int k = 0; k < MAX; k += TILE_COLS) {
                __m512i a = _mm512_loadu_si512((__m512i*)&A[i * MAX + k]);
                __m512i b = _mm512_loadu_si512((__m512i*)&B[j * MAX + k]);
                acc = _mm512_dpbusd_epi32(acc, a, b);
            }
            C[i * MAX + j] = _mm512_reduce_add_epi32(acc);
        }
    }
}

int main() {
    std::string mode;
    std::cout << "Enter target instrument? amx or vnni: ";
    std::cin >> mode;

    if (!set_tiledata_use()) return -1;

    int8_t A[MAX * MAX];
    int8_t B[MAX * MAX];
    int32_t C_amx[MAX * MAX], C_vnni[MAX * MAX];

    init_buffer(A, 1);
    init_buffer(B, 2);

    using namespace std::chrono;

    if (mode == "amx"){
        // AMX timing
        auto start_amx = high_resolution_clock::now();
        for (int i = 0; i < LOOP_COUNT; ++i) {
            init_buffer32(C_amx, 0);
            run_amx(A, B, C_amx);
        }
        auto end_amx = high_resolution_clock::now();
        auto amx_time_ns = duration_cast<nanoseconds>(end_amx - start_amx).count();
        std::cout << "AMX Time (1000 loops)        : " << amx_time_ns << " ns" << std::endl;
    } else if (mode == "vnni") {
        // AVX512 VNNI timing
        auto start_vnni = high_resolution_clock::now();
        for (int i = 0; i < LOOP_COUNT; ++i) {
            init_buffer32(C_vnni, 0);
            run_avx512_vnni(A, B, C_vnni);
        }
        auto end_vnni = high_resolution_clock::now();
        auto vnni_time_ns = duration_cast<nanoseconds>(end_vnni - start_vnni).count();
        std::cout << "AVX512 VNNI Time (1000 loops): " << vnni_time_ns << " ns" << std::endl;
    }

    return 0;
}

#include <immintrin.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <sys/syscall.h>
#include <ctime>
#include <iostream>

#define MAX 1024
#define MAX_ROWS 16
#define MAX_COLS 64
#define STRIDE 64
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

    tileinfo->colsb[1] = MAX_COLS;
    tileinfo->rows[1] = MAX_ROWS;
    tileinfo->colsb[2] = MAX_COLS;
    tileinfo->rows[2] = MAX_ROWS;
    tileinfo->colsb[3] = MAX_ROWS * 4;
    tileinfo->rows[3] = MAX_ROWS;

    _tile_loadconfig(tileinfo);
}

void init_buffer(int8_t* buf, int8_t val) {
    for (int i = 0; i < MAX_ROWS * MAX_COLS; ++i)
        buf[i] = val;
}

void init_buffer32(int32_t* buf, int32_t val) {
    for (int i = 0; i < MAX_ROWS * MAX_ROWS; ++i)
        buf[i] = val;
}

long get_time_diff_ns(timespec start, timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000000000L + (end.tv_nsec - start.tv_nsec);
}

void run_amx(int8_t* A, int8_t* B, int32_t* C) {
    __tile_config cfg = {};
    init_tile_config(&cfg);

    _tile_loadd(2, A, STRIDE);
    _tile_loadd(3, B, STRIDE);
    _tile_loadd(1, C, STRIDE);

    _tile_dpbssd(1, 2, 3);  // C += A * B

    _tile_stored(1, C, STRIDE);
    _tile_release();
}

void run_avx512_vnni(int8_t* A, int8_t* B, int32_t* C) {
    for (int i = 0; i < MAX_ROWS; ++i) {
        for (int j = 0; j < MAX_ROWS; ++j) {
            __m512i acc = _mm512_setzero_si512();
            for (int k = 0; k < MAX_COLS; k += 64) {
                __m512i a = _mm512_loadu_si512((__m512i*)&A[i * MAX_COLS + k]);
                __m512i b = _mm512_loadu_si512((__m512i*)&B[j * MAX_COLS + k]);
                acc = _mm512_dpbusd_epi32(acc, a, b);
            }
            C[i * MAX_ROWS + j] = _mm512_reduce_add_epi32(acc);
        }
    }
}

int main() {
    if (!set_tiledata_use()) return -1;

    int8_t A[MAX], B[MAX];
    int32_t C_amx[MAX_ROWS * MAX_ROWS], C_vnni[MAX_ROWS * MAX_ROWS];

    init_buffer(A, 1);
    init_buffer(B, 2);

    timespec start, end;
    long amx_time_ns, vnni_time_ns;

    // AMX
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < LOOP_COUNT; ++i) {
        init_buffer32(C_amx, 0);
        run_amx(A, B, C_amx);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    amx_time_ns = get_time_diff_ns(start, end);

    // AVX512 VNNI
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < LOOP_COUNT; ++i) {
        init_buffer32(C_vnni, 0);
        run_avx512_vnni(A, B, C_vnni);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    vnni_time_ns = get_time_diff_ns(start, end);

    std::cout << "AMX Time (1000 loops): " << amx_time_ns << " ns" << std::endl;
    std::cout << "AVX512 VNNI Time (1000 loops): " << vnni_time_ns << " ns" << std::endl;

    return 0;
}

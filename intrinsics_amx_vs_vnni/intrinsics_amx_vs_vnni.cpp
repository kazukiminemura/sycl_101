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
#define STRIDE 64
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
        tileinfo->colsb[i] = TILE_COLS;
        tileinfo->rows[i] =  TILE_ROWS;
    }

    for (int i = 1; i < 4; ++i)
    {
        tileinfo->colsb[i] = TILE_COLS;
        tileinfo->rows[i] =  TILE_ROWS;
    }

    _tile_loadconfig(tileinfo);
}

void init_buffer(int8_t* buf, int8_t val, int32_t size) {
    for (int i = 0; i < size; ++i)
        buf[i] = val;
}

void init_buffer32(int32_t* buf, int32_t val, int32_t size) {
    for (int i = 0; i < size; ++i)
        buf[i] = val;
}


/* Print int32_t buffer */
static void print_buffer32(int32_t* buf, int32_t rows, int32_t cols)
{
   for (int i = 0; i < rows; i++) {
     for (int j = 0; j < cols; j++) {
        std::cout << buf[i * cols + j] << " ";
     }
     std::cout << std::endl;
   }
   std::cout << std::endl;
}


void run_amx(int8_t* A, int8_t* B, int32_t* C) {
    int32_t mem_ptr[MATRIX_SIZE / 4] = {0};// 256要素のint32_t配列 = 1024byte

    _tile_zero(1); // 一時的な結果をタイルに保存
    _tile_loadd(2, A, STRIDE);         // Aブロック 連続したメモリから16x64Byteの2次元の形に変換して読み込む
    _tile_loadd(3, B, STRIDE);         // Bブロック 連続したメモリから16x64Byteの2次元の形に変換して読み込む
    _tile_dpbusd(1, 2, 3);                 // 積和演算: 32ビットの16x16個のドット積になる
    _tile_stored(1, mem_ptr, STRIDE);      // 一時配列に戻す
    // print_buffer32(mem_ptr, TILE_ROWS, TILE_COLS/4);

    // Cメモリに戻す
    for (int m=0; m<16; m++)
        C[0] += mem_ptr[m*16];
    memset(mem_ptr, 0, sizeof(mem_ptr));
}

void run_amx_2d(int8_t* A, int8_t* B, int32_t* C) {
    int32_t mem_ptr[MATRIX_SIZE / 4] = {0};// 256要素のint32_t配列 = 1024byte

    // A=1028x1028 B=1028x1028 C=1028x1028 なので int8_tの1024要素ごとに計算を行う
    for (int i = 0; i < MATRIX_SIZE; i+=TILE_ROWS) {
        for (int j = 0; j < MATRIX_SIZE; j+=TILE_ROWS) {
            _tile_zero(1); // 一時的な結果をタイルに保存

            for (int k = 0; k < MATRIX_SIZE; k += 64) {
                int8_t* a_ptr = &A[i * MATRIX_SIZE + k];    // A: i行目から
                int8_t* b_ptr = &B[j * MATRIX_SIZE + k];    // B: (転置されていると仮定）
                
                _tile_loadd(2, a_ptr, MATRIX_SIZE);         // Aブロック 連続したメモリから16x64Byteの2次元の形に変換して読み込む
                _tile_loadd(3, b_ptr, MATRIX_SIZE);         // Bブロック 連続したメモリから16x64Byteの2次元の形に変換して読み込む
                _tile_dpbusd(1, 2, 3);                      // 積和演算(tile1 += A * B): 32ビットの16x16個のドット積になる
            }
            _tile_stored(1, mem_ptr, STRIDE);      // 一時配列に戻す
            // print_buffer32(mem_ptr, TILE_ROWS, TILE_COLS/4);

            // mem_ptr を C に書き戻す
            for (int m = 0; m < TILE_ROWS; ++m) {
                for (int n = 0; n < TILE_COLS / 4; ++n) {
                    int row = i + m;
                    int col = j + n;
                    if (row < MATRIX_SIZE && col < MATRIX_SIZE) {
                        C[row * MATRIX_SIZE + col] = mem_ptr[m * (TILE_COLS / 4) + n];
                    }
                }
            }
        }
    }

    // // call AMX for each element
    // // A=1028x1028 B=1028x1028 C=1028x1028 なので int8_tの1024要素ごとに計算を行う
    // for (int i = 0; i < MATRIX_SIZE; i++) {
    //     int8_t* a_ptr = &A[i * MATRIX_SIZE];    // A: i行目から
    //     for (int j = 0; j < MATRIX_SIZE; j++) {
    //         int8_t* b_ptr = &B[j * MATRIX_SIZE];    // B: 転置されていると仮定）

    //         _tile_zero(1); // 一時的な結果をタイルに保存
    //         _tile_loadd(2, a_ptr, STRIDE);         // Aブロック 連続したメモリから16x64Byteの2次元の形に変換して読み込む
    //         _tile_loadd(3, b_ptr, STRIDE);         // Bブロック 連続したメモリから16x64Byteの2次元の形に変換して読み込む
    //         _tile_dpbusd(1, 2, 3);                 // 積和演算: 32ビットの16x16個のドット積になる
    //         _tile_stored(1, mem_ptr, STRIDE);      // 一時配列に戻す
    //         // print_buffer32(mem_ptr, TILE_ROWS, TILE_COLS/4);
            
    //         for (int m=0; m<16; m++)
    //             C[i * MATRIX_SIZE + j] += mem_ptr[m*16];
    //     }
    // }
}


void run_avx512_vnni(int8_t* A, int8_t* B, int32_t* C) {
    // A=1x1028 B=1028x1 C=1x1
    __m512i acc = _mm512_setzero_si512();
    for (int k = 0; k < MATRIX_SIZE; k += 64) {
        // Aの1行から64要素をロード
        const __m512i vec_a = _mm512_loadu_si512((const __m512i*)(A + k));
        // Bの1列から64要素をロード（Bは転置されていると仮定）
        const __m512i vec_b = _mm512_loadu_si512((const __m512i*)(B + k));
        acc = _mm512_dpbusd_epi32(acc, vec_a, vec_b);
    }
    C[0] = _mm512_reduce_add_epi32(acc);
}

void run_avx512_vnni_2d(int8_t* A, int8_t* B, int32_t* C) {
    // A=1028x1028 B=1028x1028 C=1028x1028 なので int8_tの64要素ごとに計算を行う
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            __m512i acc = _mm512_setzero_si512();
            for (int k = 0; k < MATRIX_SIZE; k += 64) {
                // Aの1行からint8_tの64要素をロード
                const __m512i vec_a = _mm512_loadu_si512((const __m512i*)(A + i * MATRIX_SIZE + k));
                // Bの1行からint8_tの64要素をロード（Bは転置されていると仮定）
                const __m512i vec_b = _mm512_loadu_si512((const __m512i*)(B + j * MATRIX_SIZE + k));
                acc = _mm512_dpbusd_epi32(acc, vec_a, vec_b);
            }
            C[i * MATRIX_SIZE + j] = _mm512_reduce_add_epi32(acc);
        }
    }
}

int main() {
    if (!set_tiledata_use()) return -1;

    // // 1D case: A=1xMATRIX_SIZE B=1xMATRIX_SIZE C=1x1
    // alignas(64) int8_t A[MATRIX_SIZE];
    // alignas(64) int8_t B[MATRIX_SIZE];
    // alignas(64) int32_t C_amx[1], C_vnni[1];

    // init_buffer(A, 1);
    // init_buffer(B, 1);

    // using namespace std::chrono;

    // // AMX timing
    // __tile_config cfg = {};
    // init_tile_config(&cfg);
    // auto start_amx = high_resolution_clock::now();
    // for (int i = 0; i < LOOP_COUNT; ++i) {
    //     init_buffer32(C_amx, 0);
    //     run_amx(A, B, C_amx);
    // }
    // _tile_release();
    // auto end_amx = high_resolution_clock::now();
    // auto amx_time_ms = duration_cast<std::chrono::nanoseconds>(end_amx - start_amx).count();

    // for (int i = 0; i < MATRIX_SIZE; ++i) {
    //     if (C_amx[i] != 1024) std::cout << "Wrong sum: " << C_amx[i] << std::endl;
    // }
    // std::cout << "AMX Time (1000 loops)        : " << amx_time_ms << " ns" << std::endl;

    // // AVX512 VNNI timing
    // auto start_vnni = high_resolution_clock::now();
    // for (int i = 0; i < LOOP_COUNT; ++i) {
    //     init_buffer32(C_vnni, 0);
    //     run_avx512_vnni(A, B, C_vnni);
    // }
    // auto end_vnni = high_resolution_clock::now();
    // auto vnni_time_ms = duration_cast<std::chrono::nanoseconds>(end_vnni - start_vnni).count();

    // for (int i = 0; i < MATRIX_SIZE; ++i) {
    //     if (C_vnni[i] != 1024) std::cout << "Wrong sum: " << C_vnni[i] << std::endl;
    // }
    // std::cout << "AVX512 VNNI Time (1000 loops): " << vnni_time_ms << " ns" << std::endl;

    //
    // 2D case: A=MATRIX_SIZExMATRIX_SIZE B=MATRIX_SIZExMATRIX_SIZE C=MATRIX_SIZExMATRIX_SIZE
    //
    int32_t MATRIX_SIZE_2D = MATRIX_SIZE * MATRIX_SIZE;
    alignas(64) int8_t A[MATRIX_SIZE * MATRIX_SIZE];
    alignas(64) int8_t B[MATRIX_SIZE * MATRIX_SIZE];
    alignas(64) int32_t C_amx[MATRIX_SIZE * MATRIX_SIZE], C_vnni[MATRIX_SIZE * MATRIX_SIZE];

    init_buffer(A, 1, MATRIX_SIZE_2D);
    init_buffer(B, 1, MATRIX_SIZE_2D);

    using namespace std::chrono;

    // AMX timing
    __tile_config cfg = {};
    init_tile_config(&cfg);
    auto start_amx = high_resolution_clock::now();
    for (int i = 0; i < LOOP_COUNT; ++i) {
        init_buffer32(C_amx, 0, MATRIX_SIZE_2D);
        run_amx_2d(A, B, C_amx);
    }
    _tile_release();
    auto end_amx = high_resolution_clock::now();
    auto amx_time_ms = duration_cast<std::chrono::milliseconds>(end_amx - start_amx).count();

    for (int i = 0; i < MATRIX_SIZE; ++i) {
        if (C_amx[i] != 1024) std::cout << "Wrong sum: " << C_amx[i] << std::endl;
    }
    std::cout << "2D AMX Time (1000 loops)        : " << amx_time_ms << " ms" << std::endl;

    // AVX512 VNNI timing
    auto start_vnni = high_resolution_clock::now();
    for (int i = 0; i < LOOP_COUNT; ++i) {
        init_buffer32(C_vnni, 0, MATRIX_SIZE_2D);
        run_avx512_vnni_2d(A, B, C_vnni);
    }
    auto end_vnni = high_resolution_clock::now();
    auto vnni_time_ms = duration_cast<std::chrono::milliseconds>(end_vnni - start_vnni).count();

    for (int i = 0; i < MATRIX_SIZE; ++i) {
        if (C_vnni[i] != 1024) std::cout << "Wrong sum: " << C_vnni[i] << std::endl;
    }
    std::cout << "2D Case: AVX512 VNNI Time (1000 loops): " << vnni_time_ms << " ms" << std::endl;

    return 0;
}

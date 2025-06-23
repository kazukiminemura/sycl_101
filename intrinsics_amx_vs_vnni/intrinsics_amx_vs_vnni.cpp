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

using namespace std::chrono;

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

// 単一の16x64ブロックに対してAMXを使ってドット積を計算し、結果をC[0]に加算する
void run_amx(int8_t* A, int8_t* B, int32_t* C) {
    int32_t mem_ptr[MATRIX_SIZE / 4] = {0}; // 一時バッファ（16x16 = 256要素）

    _tile_zero(1);                          // 出力タイルをゼロ初期化
    _tile_loadd(2, A, STRIDE);              // A: 16行×64列のブロックをタイル2にロード
    _tile_loadd(3, B, STRIDE);              // B: 16行×64列のブロックをタイル3にロード（転置済み）
    _tile_dpbusd(1, 2, 3);                  // タイル1に A×B のドット積を加算（16x16の結果）
    _tile_stored(1, mem_ptr, STRIDE);       // 結果を一時バッファにストア

    // 結果の左端列（mem_ptrの各行の先頭）をC[0]に加算
    for (int m = 0; m < 16; m++)
        C[0] += mem_ptr[m * 16];

    memset(mem_ptr, 0, sizeof(mem_ptr));    // バッファをクリア
}

// AMXを使って1024x1024行列の積 C = A × B を計算（Bは転置済み）
void run_amx_2d(int8_t* A, int8_t* B, int32_t* C) {
    int32_t mem_ptr[MATRIX_SIZE / 4] = {0}; // 一時バッファ（16x16）

    // C の各 16x16 ブロックに対して計算
    for (int i = 0; i < MATRIX_SIZE; i += TILE_ROWS) {
        for (int j = 0; j < MATRIX_SIZE; j += TILE_ROWS) {
            _tile_zero(1); // 出力タイルを初期化

            // Aのi行目とBのj列目（転置済み）に対して、16x64要素ずつ積和演算
            for (int k = 0; k < MATRIX_SIZE; k += TILE_COLS) {
                int8_t* a_ptr = &A[i * MATRIX_SIZE + k]; // Aの(i,k)から16x64ブロック
                int8_t* b_ptr = &B[j * MATRIX_SIZE + k]; // Bの(j,k)から16x64ブロック（転置済み）
                _tile_loadd(2, a_ptr, MATRIX_SIZE);      // Aブロックをタイル2にロード
                _tile_loadd(3, b_ptr, MATRIX_SIZE);      // Bブロックをタイル3にロード
                _tile_dpbusd(1, 2, 3);                   // タイル1に積和演算を加算
            }
            _tile_stored(1, mem_ptr, STRIDE); // 結果を一時バッファに保存

            // 一時バッファをC行列に書き戻す（16x16ブロック）
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
}

// AVX-512 VNNI を使って A(1x1024) × B(1024x1) の内積を計算し、C[0] に格納
void run_avx512_vnni(int8_t* A, int8_t* B, int32_t* C) {
    __m512i acc = _mm512_setzero_si512(); // アキュムレータ初期化

    for (int k = 0; k < MATRIX_SIZE; k += 64) {
        __m512i vec_a = _mm512_loadu_si512((const __m512i*)(A + k)); // Aの64要素
        __m512i vec_b = _mm512_loadu_si512((const __m512i*)(B + k)); // Bの64要素（転置済み）
        acc = _mm512_dpbusd_epi32(acc, vec_a, vec_b);                // ドット積を加算
    }

    C[0] = _mm512_reduce_add_epi32(acc); // 結果をスカラーにして格納
}

// AVX-512 VNNI を使って C = A × B を計算（A, B は1024x1024、Bは転置済み）
void run_avx512_vnni_2d(int8_t* A, int8_t* B, int32_t* C) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            __m512i acc = _mm512_setzero_si512(); // アキュムレータ初期化

            for (int k = 0; k < MATRIX_SIZE; k += 64) {
                const __m512i vec_a = _mm512_loadu_si512((const __m512i*)(A + i * MATRIX_SIZE + k)); // Aのi行
                const __m512i vec_b = _mm512_loadu_si512((const __m512i*)(B + j * MATRIX_SIZE + k)); // Bのj行（転置）
                acc = _mm512_dpbusd_epi32(acc, vec_a, vec_b); // ドット積を加算
            }

            C[i * MATRIX_SIZE + j] = _mm512_reduce_add_epi32(acc); // 結果をCに格納
        }
    }
}

int main() {
    if (!set_tiledata_use()) return -1;

    // ------------------------------
    // Case 1: ベクトル内積の性能比較（AMX vs AVX512 VNNI）
    // ------------------------------

    alignas(64) int8_t vector_a[MATRIX_SIZE];
    alignas(64) int8_t vector_b[MATRIX_SIZE];
    alignas(64) int32_t result_amx_scalar[1], result_vnni_scalar[1];

    // ベクトル A, B を初期化（すべての要素を 1 に設定）
    init_buffer(vector_a, 1, MATRIX_SIZE);
    init_buffer(vector_b, 1, MATRIX_SIZE);

    // ------------------------------
    // AMX による内積計算のベンチマーク
    // ------------------------------
    __tile_config tile_config_scalar = {};
    init_tile_config(&tile_config_scalar); // AMX タイル設定を初期化

    auto amx_start_time = high_resolution_clock::now();
    for (int loop = 0; loop < LOOP_COUNT; ++loop) {
        init_buffer32(result_amx_scalar, 0, 1); // 出力スカラーをゼロ初期化
        run_amx(vector_a, vector_b, result_amx_scalar); // AMX による内積計算
    }
    _tile_release(); // タイル設定を解放
    auto amx_end_time = high_resolution_clock::now();
    auto amx_duration_ns = duration_cast<nanoseconds>(amx_end_time - amx_start_time).count();

    // 結果検証（A, B がすべて 1 のとき、内積は 1024 になるはず）
    if (result_amx_scalar[0] != 1024) {
        std::cout << "[AMX] ❌ 計算結果が不正です: " << result_amx_scalar[0] << std::endl;
    } else {
        std::cout << "[AMX] ✅ 計算結果は正しいです（1024）" << std::endl;
    }
    std::cout << "[AMX] ベクトル内積（" << LOOP_COUNT << " 回繰り返し）にかかった時間: "
            << amx_duration_ns << " ns" << std::endl;

    // ------------------------------
    // AVX512 VNNI による内積計算のベンチマーク
    // ------------------------------
    auto vnni_start_time = high_resolution_clock::now();
    for (int loop = 0; loop < LOOP_COUNT; ++loop) {
        init_buffer32(result_vnni_scalar, 0, 1); // 出力スカラーをゼロ初期化
        run_avx512_vnni(vector_a, vector_b, result_vnni_scalar); // AVX512 による内積計算
    }
    auto vnni_end_time = high_resolution_clock::now();
    auto vnni_duration_ns = duration_cast<nanoseconds>(vnni_end_time - vnni_start_time).count();

    // 結果検証
    if (result_vnni_scalar[0] != 1024) {
        std::cout << "[VNNI] ❌ 計算結果が不正です: " << result_vnni_scalar[0] << std::endl;
    } else {
        std::cout << "[VNNI] ✅ 計算結果は正しいです（1024）" << std::endl;
    }
    std::cout << "[VNNI] ベクトル内積（" << LOOP_COUNT << " 回繰り返し）にかかった時間: "
            << vnni_duration_ns << " ns" << std::endl;


    // ------------------------------
    // Case 2: 行列積の性能比較（AMX vs AVX512 VNNI）
    // ------------------------------

    int32_t matrix_elements = MATRIX_SIZE * MATRIX_SIZE;

    alignas(64) int8_t matrix_a[matrix_elements];
    alignas(64) int8_t matrix_b[matrix_elements];
    alignas(64) int32_t result_amx_matrix[matrix_elements], result_vnni_matrix[matrix_elements];

    // 行列 A, B を初期化（すべての要素を 1 に設定）
    init_buffer(matrix_a, 1, matrix_elements);
    init_buffer(matrix_b, 1, matrix_elements);

    // ------------------------------
    // AMX による行列積のベンチマーク
    // ------------------------------
    __tile_config tile_config_matrix = {};
    init_tile_config(&tile_config_matrix); // AMX タイル設定を初期化

    auto amx_start_time2 = high_resolution_clock::now();
    for (int loop = 0; loop < LOOP_COUNT; ++loop) {
        init_buffer32(result_amx_matrix, 0, matrix_elements); // 出力行列をゼロ初期化
        run_amx_2d(matrix_a, matrix_b, result_amx_matrix);    // AMX による行列積
    }
    _tile_release(); // タイル設定を解放
    auto amx_end_time2 = high_resolution_clock::now();
    auto amx_duration_ms2 = duration_cast<milliseconds>(amx_end_time2 - amx_start_time2).count();

    // 結果検証（各行の先頭要素が 1024 であることを確認）
    bool amx_correct = true;
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        if (result_amx_matrix[i] != 1024) {
            std::cout << "[AMX] ❌ 計算誤り: C[" << i << "] = " << result_amx_matrix[i] << std::endl;
            amx_correct = false;
            break;
        }
    }
    if (amx_correct)
        std::cout << "[AMX] ✅ 計算結果は正しいです（各行の先頭 = 1024）" << std::endl;

    std::cout << "[AMX] 行列積（" << LOOP_COUNT << " 回繰り返し）にかかった時間: "
            << amx_duration_ms2 << " ms" << std::endl;

    // ------------------------------
    // AVX512 VNNI による行列積のベンチマーク
    // ------------------------------
    auto vnni_start_time2 = high_resolution_clock::now();
    for (int loop = 0; loop < LOOP_COUNT; ++loop) {
        init_buffer32(result_vnni_matrix, 0, matrix_elements); // 出力行列をゼロ初期化
        run_avx512_vnni_2d(matrix_a, matrix_b, result_vnni_matrix); // AVX512 による行列積
    }
    auto vnni_end_time2 = high_resolution_clock::now();
    auto vnni_duration_ms2 = duration_cast<milliseconds>(vnni_end_time2 - vnni_start_time2).count();

    // 結果検証
    bool vnni_correct = true;
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        if (result_vnni_matrix[i] != 1024) {
            std::cout << "[VNNI] ❌ 計算誤り: C[" << i << "] = " << result_vnni_matrix[i] << std::endl;
            vnni_correct = false;
            break;
        }
    }
    if (vnni_correct)
        std::cout << "[VNNI] ✅ 計算結果は正しいです（各行の先頭 = 1024）" << std::endl;

    std::cout << "[VNNI] 行列積（" << LOOP_COUNT << " 回繰り返し）にかかった時間: "
            << vnni_duration_ms2 << " ms" << std::endl;

    return 0;
}

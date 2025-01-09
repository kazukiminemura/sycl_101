# 説明
## タイル分割（Tiling）  
行列をタイル（小ブロック）に分割し、共有メモリにロードして計算を効率化します。  

## 並列処理
nd_range を使って行列全体をスレッドに分散させ、並列に計算を行います。 

## 共有メモリ  
accessor を使用して共有メモリにタイルデータを格納し、メモリアクセスを効率化します。   

GPUを最大限に利用するには、以下のポイントを考慮する必要があります： 
スレッド並列性の最大化：GPU上のスレッド数を最大限に利用するようにタスクを分割します。 
メモリアクセスの効率化：連続的なメモリアクセス（coalesced memory access）を行い、メモリ帯域を効率よく利用します。 
共有メモリの使用：データをスレッド間で効率よく共有するためにローカルメモリ（共有メモリ）を活用します。  


以下は、SYCLを使って行列積を計算するコードです。行列積は計算負荷が高く、GPUの性能を引き出す典型的なタスクです。   

行列積（Matrix Multiplication）のコード   
```
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono> // 実行時間測定用

using namespace cl::sycl;

constexpr size_t TILE_SIZE = 16; // タイルのサイズ（共有メモリ用）

// GPUで行列積を計算するカーネル
void matrix_multiply(queue &q, const float *A, const float *B, float *C, size_t N) {
    // デバイスメモリの確保
    float *d_A = malloc_device<float>(N * N, q);
    float *d_B = malloc_device<float>(N * N, q);
    float *d_C = malloc_device<float>(N * N, q);

    // ホストからデバイスへのデータ転送
    q.memcpy(d_A, A, sizeof(float) * N * N).wait();
    q.memcpy(d_B, B, sizeof(float) * N * N).wait();

    // カーネルの実行
    auto start = std::chrono::high_resolution_clock::now();

    q.submit([&](handler &h) {
        // 共有メモリの定義
        accessor<float, 2, access::mode::read_write, access::target::local> tile_A({TILE_SIZE, TILE_SIZE}, h);
        accessor<float, 2, access::mode::read_write, access::target::local> tile_B({TILE_SIZE, TILE_SIZE}, h);

        h.parallel_for(nd_range<2>({N, N}, {TILE_SIZE, TILE_SIZE}), [=](nd_item<2> item) {
            size_t row = item.get_global_id(0);
            size_t col = item.get_global_id(1);

            float sum = 0.0f;

            for (size_t t = 0; t < N / TILE_SIZE; ++t) {
                // タイルを共有メモリにロード
                tile_A[item.get_local_id(0)][item.get_local_id(1)] =
                    d_A[row * N + t * TILE_SIZE + item.get_local_id(1)];
                tile_B[item.get_local_id(0)][item.get_local_id(1)] =
                    d_B[(t * TILE_SIZE + item.get_local_id(0)) * N + col];

                item.barrier(access::fence_space::local_space);

                // タイル内で計算
                for (size_t k = 0; k < TILE_SIZE; ++k) {
                    sum += tile_A[item.get_local_id(0)][k] * tile_B[k][item.get_local_id(1)];
                }

                item.barrier(access::fence_space::local_space);
            }

            // 結果を出力
            d_C[row * N + col] = sum;
        });
    }).wait();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // 実行時間を表示
    std::cout << "Execution time: " << diff.count() << " seconds" << std::endl;

    // デバイスからホストへの結果コピー
    q.memcpy(C, d_C, sizeof(float) * N * N).wait();

    // メモリ解放
    free(d_A, q);
    free(d_B, q);
    free(d_C, q);
}

int main() {
    constexpr size_t N = 1024; // 行列のサイズ (N x N)

    // 行列の初期化
    std::vector<float> A(N * N, 1.0f); // 行列A (全て1.0)
    std::vector<float> B(N * N, 1.0f); // 行列B (全て1.0)
    std::vector<float> C(N * N, 0.0f); // 結果を格納する行列C

    // SYCLデバイスキューの作成
    queue q{gpu_selector{}, [](exception_list e_list) {
                for (std::exception_ptr const &e : e_list) {
                    try {
                        std::rethrow_exception(e);
                    } catch (std::exception const &e) {
                        std::cerr << "SYCL exception: " << e.what() << "\n";
                    }
                }
            }};

    std::cout << "Running on: " 
              << q.get_device().get_info<info::device::name>() 
              << std::endl;

    // 行列積を計算
    matrix_multiply(q, A.data(), B.data(), C.data(), N);

    // 結果の一部を表示
    std::cout << "Result sample (first 10 elements of the first row): ";
    for (size_t i = 0; i < 10; ++i) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

説明  
タイル分割（Tiling）  
行列をタイル（小ブロック）に分割し、共有メモリにロードして計算を効率化します。  

並列処理  
nd_range を使って行列全体をスレッドに分散させ、並列に計算を行います。 

共有メモリ  
accessor を使用して共有メモリにタイルデータを格納し、メモリアクセスを効率化します。 
  
# 実行手順
## コンパイル
```
icpx -fsycl -O2 -o sycl_matrix_multiply_gpu sycl_matrix_multiply_gpu.cpp
```

## 実行
``` ./matrix_multiply ```

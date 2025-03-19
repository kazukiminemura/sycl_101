#include <sycl/sycl.hpp>
#include <chrono> // 実行時間測定用
#include <ittnotify.h>

using namespace sycl;

constexpr int matrixSize = 1024;
constexpr int iterations = 16;
constexpr size_t TILE_SIZE = 16; // タイルのサイズ（共有メモリ用）


// Even more parallel matrix multiplication
void even_more_parallel_matrix_multiply(
    std::vector<float> vecA, std::vector<float> vecB, std::vector<float> vecC,
    const int M, const int N, const int K) {

    auto start = std::chrono::high_resolution_clock::now();
    queue q;
    // std::cout << "Running on: " << q.get_device().get_info<info::device::name>()static_cast<size_t>(M) << std::endl;

    buffer bufA{vecA};  // M * K elements
    buffer bufB{vecB};  // K * N elements
    buffer bufC{vecC};  // M * N elements
    q.submit([&](handler &h) {
        accessor matrixA{bufA, h};
        accessor matrixB{bufB, h};
        accessor matrixC{bufC, h};
  
        h.parallel_for(range{static_cast<size_t>(M), static_cast<size_t>(N)}, [=](id<2> idx) {
            int m = idx[0];
            int n = idx[1];
        
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += matrixA[m * K + k] * matrixB[k * N + n];
            }
            matrixC[m * N + n] = sum;
        });
    }).wait();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    // 実行時間を表示
    std::cout << "Even more parallel matrix multiplication: " << diff.count() << " seconds" << std::endl;

    host_accessor results{bufC};
    for(int m = 0; m < M; m++){
        for(int n = 0; n < N; n++){
            assert(results[m*N + n] == K);
        }
    }
}

// Inefficient matrix multiplication
void single_workitem_per_workgroup_parallel_matrix_multiply(
    std::vector<float> vecA, std::vector<float> vecB, std::vector<float> vecC,
    const int M, const int N, const int K) {

    auto start = std::chrono::high_resolution_clock::now();
    queue q;
    // std::cout << "Running on: " << q.get_device().get_info<info::device::name>() << std::endl;

    buffer bufA{vecA};  // M * K elements
    buffer bufB{vecB};  // K * N elements
    buffer bufC{vecC};  // M * N elements
    q.submit([&](handler &h) {
        accessor matrixA{bufA, h};
        accessor matrixB{bufB, h};
        accessor matrixC{bufC, h};
  
        h.parallel_for(nd_range<1>{M, 1}, [=](nd_item<1> idx) {
            int m = idx.get_global_id(0);
            for (int n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += matrixA[m * K + k] * matrixB[k * N + n];
                }
                matrixC[m * N + n] = sum;
            }
        });
    }).wait();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    // 実行時間を表示
    std::cout << "single workitem per workgroup parallel matrix multiplication: " << diff.count() << " seconds" << std::endl;

    host_accessor results{bufC};
    for(int m = 0; m < M; m++){
        for(int n = 0; n < N; n++){
            assert(results[m*N + n] == K);
        }
    }
}

int main() {
    const int M = matrixSize;
    const int N = matrixSize;
    const int K = matrixSize;

    // 行列の初期化
    std::vector<float> A(M*K, 1.0f); // 行列A (全て1.0)
    std::vector<float> B(K*N, 1.0f); // 行列B (全て1.0)
    std::vector<float> C(M*N, 0.0f); // 結果を格納する行列C

    queue q;
    std::cout << "Running on: " << q.get_device().get_info<info::device::name>()static_cast<size_t>(M) << std::endl;

    __itt_pt_region region1 = __itt_pt_region_create("even_more_parallel_matrix_multiply_region");
    __itt_pt_region region2 = __itt_pt_region_create("single_workitem_per_workgroup_parallel_matrix_region");

    for (int i = 0; i < 100; i++) {
        // Even more parallel matrix multiplication
        __itt_mark_pt_region_begin(region1);
        even_more_parallel_matrix_multiply(A, B, C, M, N, K);
        __itt_mark_pt_region_end(region1);
        
        // Inefficient matrix multiplication
        __itt_mark_pt_region_begin(region2);
        single_workitem_per_workgroup_parallel_matrix_multiply(A, B, C, M, N, K);
        __itt_mark_pt_region_end(region2);
    }

    return 0;
}

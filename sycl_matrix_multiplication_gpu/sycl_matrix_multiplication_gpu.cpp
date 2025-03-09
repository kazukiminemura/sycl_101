#include <sycl/sycl.hpp>
#include <chrono> // 実行時間測定用

using namespace sycl;

constexpr int matrixSize = 1024;
constexpr int iterations = 16;
constexpr size_t TILE_SIZE = 16; // タイルのサイズ（共有メモリ用）

// sequential matrix multiplication
void sequential_matrix_multiply(
    std::vector<float> vecA, std::vector<float> vecB, std::vector<float> vecC,
    const int M, const int N, const int K) {
    // 行列のサイズを表示
    std::cout << "M: " << M << ", N: " << N << ", K: " << K << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += vecA[m * K + k] * vecB[k * N + n];
            }
            vecC[m * N + n] = sum;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    // 実行時間を表示
    std::cout << "sequential matrix multiplication: " << diff.count() << " seconds" << std::endl;

    for(int m = 0; m < M; m++){
        for(int n = 0; n < N; n++){
            // std::cout << vecC[m*N + n] << std::endl;
            assert(vecC[m*N + n] == K);
        }
    }
}

// somewhat parallel matric multiplication
void somewhat_parallel_matrix_multiply(
    std::vector<float> vecA, std::vector<float> vecB, std::vector<float> vecC,
    const int M, const int N, const int K) {

    auto start = std::chrono::high_resolution_clock::now();
    queue q;
    std::cout << "Running on: " << q.get_device().get_info<info::device::name>() << std::endl;

    buffer bufA{vecA};  // M * K elements
    buffer bufB{vecB};  // K * N elements
    buffer bufC{vecC};  // M * N elements
    q.submit([&](handler &h) {
        accessor matrixA{bufA, h};
        accessor matrixB{bufB, h};
        accessor matrixC{bufC, h};
  
        h.parallel_for(range{static_cast<size_t>(M)}, [=](id<1> idx) {
            int m = idx;
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
    std::cout << "somewhat parallel matrix multiplication: " << diff.count() << " seconds" << std::endl;

    host_accessor results{bufC};
    for(int m = 0; m < M; m++){
        for(int n = 0; n < N; n++){
            assert(results[m*N + n] == (float)K);
        }
    }
}

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

// Column first matrix multiplication: better locality
void column_parallel_matrix_multiply(
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
  
        h.parallel_for(range{static_cast<size_t>(N)}, [=](id<1> idx) {
            int n = idx[0];
            for (int m = 0; m < M; ++m) {
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
    std::cout << "column parallel matrix multiplication: " << diff.count() << " seconds" << std::endl;

    host_accessor results{bufC};
    for(int m = 0; m < M; m++){
        for(int n = 0; n < N; n++){
            assert(results[m*N + n] == K);
        }
    }

}

// Tiled matrix multiplication
void tiled_matrix_multiply(
    std::vector<float> vecA, std::vector<float> vecB, std::vector<float> vecC,
    const int M, const int N, const int K) {

    // カーネルの実行
    auto start = std::chrono::high_resolution_clock::now();
    queue q;

    // デバイスメモリの確保
    float *matrixA = malloc_shared<float>(vecA.size(), q);
    float *matrixB = malloc_shared<float>(vecB.size(), q);
    float *matrixC = malloc_shared<float>(vecC.size(), q);

    // ホストからデバイスへのデータ転送
    q.memcpy(matrixA, vecA.data(), sizeof(float) * vecA.size()).wait();
    q.memcpy(matrixB, vecB.data(), sizeof(float) * vecB.size()).wait();

    q.submit([&](handler &h) {
        // ローカルメモリの確保
        local_accessor<float> tile_A({TILE_SIZE}, h);
        // local_acccessor tile_B({TILE_SIZE * TILE_SIZE}, h);

        h.parallel_for(nd_range<2>({static_cast<size_t>(M), static_cast<size_t>(N)}, {1, TILE_SIZE}), [=](nd_item<2> item) {
            // Indices in the glovba matrix
            size_t m = item.get_global_id(0);
            size_t n = item.get_global_id(1);

            // index in the local index space:
            size_t local_row = item.get_local_id(1);

            float sum = 0.0f;
            for (size_t t = 0; t < K; t+=TILE_SIZE) {
                // マトリックスタイルをmatrixAからローカルメモリにコピーし、すべてのワークアイテムが一貫したデータを持つようにする
                tile_A[local_row] = matrixA[m * K + t + local_row];
                item.barrier();

                //　ローカルタイルとmatrixB（グローバルメモリ）を使ってsumを計算
                for (size_t k = 0; k < TILE_SIZE; ++k) {
                    sum += tile_A[k] * matrixB[(t + k)*N + n];
                }
                item.barrier();
            }

            // 結果を出力
            matrixC[m * N + n] = sum;
        });
    }).wait();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    // 実行時間を表示
    std::cout << "Tiled matrix mulpilication: " << diff.count() << " seconds" << std::endl;

    for(int m = 0; m < M; m++){
        for(int n = 0; n < N; n++){
            assert(matrixC[m*N + n] == K);
        }
    }

    // メモリ解放
    free(matrixA, q);
    free(matrixB, q);
    free(matrixC, q);
}

int main() {
    const int M = matrixSize;
    const int N = matrixSize;
    const int K = matrixSize;

    // 行列の初期化
    std::vector<float> A(M*K, 1.0f); // 行列A (全て1.0)
    std::vector<float> B(K*N, 1.0f); // 行列B (全て1.0)
    std::vector<float> C(M*N, 0.0f); // 結果を格納する行列C

    // sequential matrix multiplication
    sequential_matrix_multiply(A, B, C, M, N, K);
    // somewhat parallel matrix multiplication
    somewhat_parallel_matrix_multiply(A, B, C, M, N, K);
    // Even more parallel matrix multiplication
    even_more_parallel_matrix_multiply(A, B, C, M, N, K);
    // Inefficient matrix multiplication
    single_workitem_per_workgroup_parallel_matrix_multiply(A, B, C, M, N, K);
    // Column first matrix multiplication
    column_parallel_matrix_multiply(A, B, C, M, N, K);

    // Tiled matrix multiplication
    tiled_matrix_multiply(A, B, C, M, N, K);

    return 0;
}

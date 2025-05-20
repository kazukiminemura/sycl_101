#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <mkl.h>

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K>\n";
        return 1;
    }

    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);
    constexpr int iterations = 1000;

    int num_threads = mkl_get_max_threads();
    std::cout << "Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K << "\n";
    std::cout << "Using " << num_threads << " MKL threads\n";

    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 1.0f);
    std::vector<float> C(M * N, 0.0f);

    // Warm-up
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f,
                A.data(), K,
                B.data(), N,
                0.0f,
                C.data(), N);

    // Timed iterations
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        std::fill(C.begin(), C.end(), 0.0f);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K,
                    1.0f,
                    A.data(), K,
                    B.data(), N,
                    0.0f,
                    C.data(), N);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    double avg_time = elapsed.count() / iterations;
    double tflops = (2.0 * M * N * K) / (avg_time * 1e12);

    std::cout << "Average execution time over " << iterations << " runs: " << avg_time << " seconds\n";
    std::cout << "SGEMM Performance: " << tflops << " TFLOPs\n";

    return 0;
}

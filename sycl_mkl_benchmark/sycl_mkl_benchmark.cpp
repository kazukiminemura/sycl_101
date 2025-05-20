#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>  // for std::atoi

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <M> <N> <K>\n";
        return 1;
    }

    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);
    constexpr int iterations = 1000;

    sycl::queue q;
    auto device = q.get_device();

    std::cout << "Device: " << device.get_info<sycl::info::device::name>() << "\n";
    std::cout << "Max compute units (threads): " << device.get_info<sycl::info::device::max_compute_units>() << "\n";
    std::cout << "Matrix dimensions: M=" << M << ", N=" << N << ", K=" << K << "\n";

    std::vector<oneapi::mkl::bfloat16> A(M * K, oneapi::mkl::bfloat16(1.0f));
    std::vector<oneapi::mkl::bfloat16> B(K * N, oneapi::mkl::bfloat16(1.0f));
    std::vector<float> C(M * N, 0.0f);

    oneapi::mkl::bfloat16* d_A = sycl::malloc_device<oneapi::mkl::bfloat16>(M * K, q);
    oneapi::mkl::bfloat16* d_B = sycl::malloc_device<oneapi::mkl::bfloat16>(K * N, q);
    float* d_C = sycl::malloc_device<float>(M * N, q);

    q.memcpy(d_A, A.data(), sizeof(oneapi::mkl::bfloat16) * M * K).wait();
    q.memcpy(d_B, B.data(), sizeof(oneapi::mkl::bfloat16) * K * N).wait();
    q.memset(d_C, 0, sizeof(float) * M * N).wait();

    // Warm-up
    oneapi::mkl::blas::row_major::gemm(
        q,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        M, N, K,
        1.0f,
        d_A, K,
        d_B, N,
        0.0f,
        d_C, N
    ).wait();

    // Timed iterations
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        oneapi::mkl::blas::row_major::gemm(
            q,
            oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans,
            M, N, K,
            1.0f,
            d_A, K,
            d_B, N,
            0.0f,
            d_C, N
        ).wait();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    double avg_time = elapsed.count() / iterations;
    double tflops = (2.0 * M * N * K) / (avg_time * 1e12);

    std::cout << "Average execution time over " << iterations << " runs: " << avg_time << " seconds\n";
    std::cout << "BF16 Performance: " << tflops << " TFLOPs\n";

    sycl::free(d_A, q);
    sycl::free(d_B, q);
    sycl::free(d_C, q);

    return 0;
}

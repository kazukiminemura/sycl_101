#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>

using namespace sycl;
using namespace std::chrono;

template<int subgroup_size>
void run_kernel(int* A, int* B, int* C, const int N, queue q){
    sycl::device device = q.get_device();
    size_t max_work_group_size = device.get_info<sycl::info::device::max_work_group_size>();

    auto start = high_resolution_clock::now();

    // カーネル実行
    q.submit([&](handler& h) {
        h.parallel_for(nd_range<1>(N, max_work_group_size),
            [=](nd_item<1> item) [[intel::reqd_sub_group_size(subgroup_size)]] {
            int i = item.get_global_id(0);
            C[i] = A[i] + B[i];
        });
    }).wait();

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    std::cout << "Subgroup size: " << subgroup_size << ", Execution time: " << duration << " us" << std::endl;
}

int main() {
    const int N = 1024 * 1024;  // 1M要素
    queue q;
    std::cout << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    // USMメモリの確保
    int* A = malloc_shared<int>(N, q);
    int* B = malloc_shared<int>(N, q);
    int* C = malloc_shared<int>(N, q);

    // ホスト側でデータを初期化
    for (int i = 0; i < N; i++) {
        A[i] = 1;
        B[i] = 2;
    }

    // 異なるサブグループサイズで実行
    std::cout << "Running kernel with different subgroup sizes..." << std::endl;
    run_kernel<8>(A, B, C, N, q);  // 最小のサブグループサイズ
    run_kernel<16>(A, B, C, N, q); // サブグループサイズ16
    run_kernel<32>(A, B, C, N, q); // サブグループサイズ32

    // USMメモリの解放
    free(A, q);
    free(B, q);
    free(C, q);

    return 0;
}

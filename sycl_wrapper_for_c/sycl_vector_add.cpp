#include <sycl/sycl.hpp>
#include <iostream>

// SYCLを使ったベクトル加算を行う関数
extern "C" void vector_add(const float *a, const float *b, float *c, int n) {
    sycl::queue q;

    // デバイスメモリの割り当て
    float *d_a = sycl::malloc_device<float>(n, q);
    float *d_b = sycl::malloc_device<float>(n, q);
    float *d_c = sycl::malloc_device<float>(n, q);

    // ホストからデバイスにデータをコピー
    q.memcpy(d_a, a, n * sizeof(float)).wait();
    q.memcpy(d_b, b, n * sizeof(float)).wait();

    // SYCLカーネルでベクトル加算を実行
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
        int i = idx[0];
        d_c[i] = d_a[i] + d_b[i];
    }).wait();

    // 結果をホストにコピー
    q.memcpy(c, d_c, n * sizeof(float)).wait();

    // デバイスメモリを解放
    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);
}

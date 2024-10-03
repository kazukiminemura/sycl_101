#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace sycl;

int main()
{
    const size_t N = 512; // 行列のサイズ
    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 2.0f);
    std::vector<float> C(N * N, 0.0f);

    try
    {
        queue q;

        buffer<float, 2> a_buf(A.data(), range<2>(N, N));
        buffer<float, 2> b_buf(B.data(), range<2>(N, N));
        buffer<float, 2> c_buf(C.data(), range<2>(N, N));

        auto start = std::chrono::high_resolution_clock::now();

        q.submit([&](handler &h)
                 {
            auto a = a_buf.get_access<access::mode::read>(h);
            auto b = b_buf.get_access<access::mode::read>(h);
            auto c = c_buf.get_access<access::mode::write>(h);

            h.parallel_for<class matmul_basic>(range<2>(N, N), [=](id<2> idx) {
                size_t row = idx[0];
                size_t col = idx[1];
                float sum = 0.0f;
                for (size_t k = 0; k < N; k++) {
                    sum += a[row][k] * b[k][col];
                }
                c[row][col] = sum;
            }); }).wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "実行時間（基本実装）: " << duration.count() << " ms" << std::endl;
    }
    catch (exception const &e)
    {
        std::cerr << "SYCL例外が発生しました: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
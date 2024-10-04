#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>

using namespace sycl;

int main() {
    constexpr unsigned Size = 1024 * 128;

    try {
        // キューの作成（デバイスを選択）
        queue q;
        auto dev = q.get_device();
        std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

        // USMメモリの割り当て
        float* A = malloc_shared<float>(Size, q);
        float* B = malloc_shared<float>(Size, q);
        float* C = malloc_shared<float>(Size, q);

        // ホスト側でデータを初期化
        for (unsigned int i = 0; i < Size; ++i) {
            A[i] = 1.0f;
            B[i] = 2.0f;
            C[i] = 0.0f;
        }

        auto start = std::chrono::high_resolution_clock::now();
      
        // カーネルの実行
        q.submit([&](handler& cgh) {
            cgh.parallel_for<class vector_add>(
                range<1>(Size), [=](id<1> idx) {
                    unsigned int i = idx[0];
                    C[i] = A[i] + B[i];
                });
        });

        q.wait();
      
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "実行時間（基本実装）: " << duration.count() << " ms" << std::endl;

        // 結果の検証
        bool passed = true;
        for (size_t i = 0; i < Size; i++) {
            if (C[i] != A[i] + B[i]) {
                passed = false;
                break;
            }
        }

        if (passed) {
            std::cout << "計算結果は正しいです。" << std::endl;
        } else {
            std::cout << "計算結果に誤りがあります。" << std::endl;
        }

        // USMメモリの解放
        sycl::free(A, q);
        sycl::free(B, q);
        sycl::free(C, q);

    } catch (sycl::exception const& e) {
        std::cerr << "SYCL例外が発生しました: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

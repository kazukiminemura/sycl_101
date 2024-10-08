#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <iostream>
#include <chrono>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

int main() {
    constexpr unsigned Size = 1024 * 128;
    // SIMDサイズを定義
    constexpr unsigned VL = 16;

    try {
        // キューの作成（GPUデバイスを選択）
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
            cgh.parallel_for<class esimd_vector_add>(
                range<1>(Size / VL), [=](id<1> i) SYCL_ESIMD_KERNEL {
                    // オフセットの計算
                    unsigned int offset = i * VL;

                    // メモリからロード
                    simd<float, VL> va;
                    simd<float, VL> vb;
                    va.copy_from(A + offset);
                    vb.copy_from(B + offset);

                    // 要素ごとの加算
                    simd<float, VL> vc = va + vb;

                    // メモリにストア
                    vc.copy_to(C + offset);
                });
        });

        q.wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "実行時間（ESIMD実装）: " << duration.count() << " ms" << std::endl;
        
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

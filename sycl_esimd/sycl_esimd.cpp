#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <iostream>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

int main() {
    constexpr unsigned int size = 1024;
    std::vector<float> A(size, 1.0f);
    std::vector<float> B(size, 2.0f);
    std::vector<float> C(size, 0.0f);

    try {
        queue q;

        // バッファの作成
        buffer<float, 1> bufA(A.data(), range<1>(size));
        buffer<float, 1> bufB(B.data(), range<1>(size));
        buffer<float, 1> bufC(C.data(), range<1>(size));

        q.submit([&](handler& cgh) {
            // アクセサの作成
            auto accA = bufA.get_access<access::mode::read>(cgh);
            auto accB = bufB.get_access<access::mode::read>(cgh);
            auto accC = bufC.get_access<access::mode::write>(cgh);

            cgh.parallel_for<class esimd_vector_add>(
                range<1>(size / 16), [=](id<1> i) SYCL_ESIMD_KERNEL {
                    // SIMDサイズを定義
                    constexpr unsigned int VL = 16;

                    // ベクトルのオフセットを計算
                    unsigned int offset = i[0] * VL;

                    // メモリからロード
                    simd<float, VL> va;
                    simd<float, VL> vb;
                    va.copy_from(accA.get_pointer() + offset);
                    vb.copy_from(accB.get_pointer() + offset);

                    // 要素ごとの加算
                    simd<float, VL> vc = va + vb;

                    // メモリにストア
                    vc.copy_to(accC.get_pointer() + offset);
                });
        }).wait();

        // 結果の検証
        bool passed = true;
        for (size_t i = 0; i < size; i++) {
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

    } catch (exception const& e) {
        std::cerr << "SYCL例外が発生しました: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

# サンプルコード: 行列乗算のプロファイリングと最適化
このサンプルコードでは、行列乗算を実装し、プロファイラを使用してボトルネックを特定します。その後、メモリアクセスパターンと計算量を最適化してパフォーマンスを向上させます。  

## ステップ1: ベーシックな行列乗算の実装   
まず、基本的な行列乗算をSYCLで実装します。  

```
cpp
Copy code
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace sycl;

int main() {
    const size_t N = 512; // 行列のサイズ
    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 2.0f);
    std::vector<float> C(N * N, 0.0f);

    try {
        queue q{ default_selector{} };

        buffer<float, 2> a_buf(A.data(), range<2>(N, N));
        buffer<float, 2> b_buf(B.data(), range<2>(N, N));
        buffer<float, 2> c_buf(C.data(), range<2>(N, N));

        auto start = std::chrono::high_resolution_clock::now();

        q.submit([&](handler& h) {
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
            });
        }).wait();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "実行時間（基本実装）: " << duration.count() << " ms" << std::endl;

    } catch (exception const& e) {
        std::cerr << "SYCL例外が発生しました: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

# コードの説明
行列サイズの設定: N = 512で512x512の行列を使用。    
データの初期化: 行列AとBを初期化し、結果を格納する行列Cを用意。 
バッファの作成: buffer<float, 2>で2次元のバッファを作成。   
カーネルの実行: 
parallel_forを使用して各要素の計算を並列化。    
内部ループでkを用いて行列乗算を計算。   
実行時間の計測: std::chronoを使用してカーネルの実行時間を計測。 
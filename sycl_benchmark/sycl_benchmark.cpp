#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>  // 実行時間測定用

using namespace sycl;

int main() {
    // ベクトルのサイズ（大きいほど負荷が高くなる）
    const size_t vector_size = 1 << 26; // 約67M要素

    // SYCLデバイスキューの作成（GPU優先、無い場合はデフォルト）
    queue q{default_selector{}, [](exception_list e_list) {
                for (std::exception_ptr const &e : e_list) {
                    try {
                        std::rethrow_exception(e);
                    } catch (std::exception const &e) {
                        std::cerr << "SYCL exception: " << e.what() << "\n";
                    }
                }
            }};

    std::cout << "Running on: " 
              << q.get_device().get_info<sycl::info::device::name>() 
              << std::endl;

    // ホスト側のデータ
    std::vector<float> a(vector_size, 1.0f); // すべて1.0
    std::vector<float> b(vector_size, 2.0f); // すべて2.0
    std::vector<float> c(vector_size, 0.0f); // 結果を格納

    // デバイスメモリの確保
    float *d_a = malloc_device<float>(vector_size, q);
    float *d_b = malloc_device<float>(vector_size, q);
    float *d_c = malloc_device<float>(vector_size, q);

    // ホストからデバイスへのデータコピー
    q.memcpy(d_a, a.data(), sizeof(float) * vector_size).wait();
    q.memcpy(d_b, b.data(), sizeof(float) * vector_size).wait();

    // ベンチマーク測定開始
    auto start = std::chrono::high_resolution_clock::now();

    // ベクトル加算カーネル
    q.parallel_for(sycl::range<1>(vector_size), [=](sycl::id<1> i) {
         d_c[i] = d_a[i] + d_b[i];
    }).wait(); // 実行完了を待機

    // ベンチマーク測定終了
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // デバイスからホストへのデータコピー
    q.memcpy(c.data(), d_c, sizeof(float) * vector_size).wait();

    // 実行時間を表示
    std::cout << "Execution time: " << diff.count() << " seconds" << std::endl;

    // パフォーマンスを計算 (GB/s)
    double data_size = 3.0 * vector_size * sizeof(float) / 1e9; // 読み書きの合計
    double bandwidth = data_size / diff.count();
    std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;

    // 結果確認（最初の10要素）
    std::cout << "Sample result: ";
    for (size_t i = 0; i < 10; ++i) {
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    // メモリ解放
    free(d_a, q);
    free(d_b, q);
    free(d_c, q);

    return 0;
}

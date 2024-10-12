#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>  // 実行時間を計測するために追加

using namespace sycl;
using namespace std::chrono;  // 実行時間計測に便利なstd::chrono名前空間

// 簡単なAI推論の例（1層のDense Layer）
void dense_layer(
                queue& q,
                float* input,
                float* weights,
                float* bias,
                float* output,
                int input_size,
                int output_size) {
    // SYCLカーネルを実行
    q.submit([&](handler& h) {
        h.parallel_for(range<1>(output_size), [=](id<1> idx) {
            int i = idx[0];
            float sum = bias[i];  // バイアスを初期値として使用
            for (int j = 0; j < input_size; j++) {
                sum += input[j] * weights[i * input_size + j];  // 重み行列と入力ベクトルの積
            }
            output[i] = sum;  // 結果を出力
        });
    }).wait();
}

int main() {
    // デバイスキューの作成
    queue q;
    std::cout << "Running on " << q.get_device().get_info<info::device::name>() << "\n";

    // ネットワークパラメータの定義
    const int input_size = 4;
    const int output_size = 512;

    // データを定義（例として固定の値を使用）
    std::vector<float> input_data = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> weights_data(input_size * output_size, 0.5f);  // 4x512の重み行列を0.5で初期化
    std::vector<float> bias_data(output_size, 1.0f);  // バイアス値を1で初期化
    std::vector<float> output_data(output_size, 0.0);  // 出力データを初期化


    // USMでメモリを確保
    float* input = malloc_shared<float>(input_size, q);
    float* weights = malloc_shared<float>(input_size * output_size, q);
    float* bias = malloc_shared<float>(output_size, q);
    float* output = malloc_shared<float>(output_size, q);

    // データをコピー
    std::copy(input_data.begin(), input_data.end(), input);
    std::copy(weights_data.begin(), weights_data.end(), weights);
    std::copy(bias_data.begin(), bias_data.end(), bias);

    // 実行時間の計測開始
    auto start = high_resolution_clock::now();
    // 推論実行
    dense_layer(q, input, weights, bias, output, input_size, output_size);
    // 実行時間の計測終了
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start).count();
    
    // 実行時間を表示
    std::cout << "実行時間(オリジナル): " << duration << " microseconds" << std::endl;

    // 結果を表示
    std::cout << "Output: ";
    for (int i = 0; i < output_size; i++) {
        std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    // メモリを解放
    free(input, q);
    free(weights, q);
    free(bias, q);
    free(output, q);

    return 0;
}

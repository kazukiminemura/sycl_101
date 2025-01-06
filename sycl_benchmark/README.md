# 実行手順
## コンパイル
Intel oneAPI DPC++ コンパイラ (icpx) を使用します。
'''icpx -fsycl -O2 -o gpu_benchmark gpu_benchmark.cpp'''
## 実行
実行してGPU性能を計測します。
./gpu_benchmark

# 出力例
'''
Running on: Intel(R) UHD Graphics 620
Execution time: 0.056732 seconds
Bandwidth: 7.64 GB/s
Sample result: 3 3 3 3 3 3 3 3 3 3
'''
Execution time: ベクトル加算にかかった時間。
Bandwidth: データ転送と計算の合計でのメモリ帯域幅。
Sample result: 計算結果（ここではすべて 1.0 + 2.0 = 3.0）。

# カスタマイズ
## ベクトルサイズ:
const size_t vector_size を変更して負荷を増減できます。
## GPUデバイスの指定:
default_selector{} を gpu_selector{} に変更すると、明示的にGPUを選択します。

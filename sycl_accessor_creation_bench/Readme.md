このコードは、SYCLを使って並列計算を行い、2種類のアクセサ（シンプルアクセサと指定アクセサ）のパフォーマンスを比較するプログラムです。

# 主なポイント
バッファーとアクセサの作成: データを格納するバッファーと、そのデータにアクセスするアクセサを作成。  
並列計算の実行: parallel_forを使って、配列の各要素に対して並列に計算を実行。  
実行時間の計測: std::chronoを使って、各アクセサの実行時間を計測し、結果を表示。  

# Usage
```
$ icpx -fsycl sycl_accessor_creation_bench.cpp

$ a.exe
Intel(R) Iris(R) Xe Graphics
Simple accessor duration: 0.0188568 seconds
Specified accessor duration: 0.0011152 seconds
```

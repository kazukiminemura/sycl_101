# Usage
icpx -fsycl sycl_expressing_parallelism.cpp


# Results
```
a.exe
Device: Intel(R) Iris(R) Xe Graphics
parallel_for duration: 1.09972 seconds
parallel_for_work_group duration: 0.790371 seconds
```

# コード説明
このコードは、SYCLを使用してparallel_forとparallel_for_work_groupの2つの異なるカーネルの性能を比較するプログラムです。以下に各部分の説明を示します：

## ヘッダーファイルのインクルード:
```
#include <sycl/sycl.hpp>
#include <iostream>
#include <chrono>
```
SYCL、標準入出力、時間計測のためのヘッダーファイルをインクルードしています。
## 名前空間の使用
```
using namespace sycl;
```
## メイン関数
```
int main() {
    const size_t N = 1024;
    const size_t B = 4;
    std::vector<int> data1(N, 1);
    std::vector<int> data2(N, 1);
配列のサイズNとブロックサイズBを定義し、2つのベクトルdata1とdata2を初期化します。
```
## デバイスとキューの設定
```
queue q;
std::cout << "Device: " << q.get_device().get_info<info::device::name>() << std::endl;
```
SYCLキューを作成し、使用するデバイスの名前を表示します。

# バッファの作成
```
buffer<int, 1> buf1(data1.data(), range<1>(N));
buffer<int, 1> buf2(data2.data(), range<1>(N));
```
ベクトルdata1とdata2を基にバッファbuf1とbuf2を作成します。
# parallel_forを使ったカーネル
```
auto start1 = std::chrono::high_resolution_clock::now();
for (size_t i=0; i<10000; ++i){
  q.submit(& {
      auto acc = buf1.get_access<access::mode::read_write>(h);
      h.parallel_for(range<1>(N), = {
          acc[idx] = 1;
      });
  }).wait();
}
auto end1 = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> duration1 = end1 - start1;
```
parallel_forを使って、バッファbuf1の各要素を1に設定するカーネルを10000回実行し、実行時間を計測します。
# parallel_for_work_groupを使ったカーネル
```
auto start2 = std::chrono::high_resolution_clock::now();
range num_group{N / B};
range group_size{B};
for (size_t i=0; i<10000; ++i){
  q.submit(& {
      auto acc = buf2.get_access<access::mode::read_write>(h);
      h.parallel_for_work_group(num_group, group_size, ={
        int jb = grp.get_id(0);
          grp.parallel_for_work_item(& {
            int j = jb * B + item.get_local_id(0);
              acc[j] = 1;
          });
      });
  }).wait();
}
auto end2 = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> duration2 = end2 - start2;
```
parallel_for_work_groupを使って、バッファbuf2の各要素を1に設定するカーネルを10000回実行し、実行時間を計測します。

## 実行時間の表示
```
std::cout << "parallel_for duration: " << duration1.count() << " seconds\n";
std::cout << "parallel_for_work_group duration: " << duration2.count() << " seconds\n";
```
各カーネルの実行時間を表示します。

# 結果の確認
```
host_accessor buf1_hacc(buf1, read_only);
host_accessor buf2_hacc(buf2, read_only);
for (size_t i=0; i<N; ++i){
  if (buf1_hacc[i] != 1){
      std::cout << "something wrong in parallel_for " << std::endl;
      break;
    }
}
for (size_t i=0; i<N; ++i){
  if (buf2_hacc[i] != 1){
      std::cout << "something wrong in parallel_for_work_group" << std::endl;
      break;
    }
}
```
バッファbuf1とbuf2の内容を確認し、正しく設定されているかをチェックします。

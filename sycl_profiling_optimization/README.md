# サンプルコード: 行列乗算のプロファイリングと最適化
このサンプルコードでは、行列乗算を実装し、プロファイラを使用してボトルネックを特定します。その後、メモリアクセスパターンと計算量を最適化してパフォーマンスを向上させます。  

## ステップ1: ベーシックな行列乗算の実装   
まず、基本的な行列乗算をSYCLで実装します。  

```
#include <sycl/sycl.hpp>
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
        queue q;

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


## ステップ2: プロファイラを使用してボトルネックを特定
### プロファイラの使用方法    
    
Intel VTune Profilerのインストール: Intel oneAPIツールキットに含まれるVTune Profilerを使用します。    
    
プログラムのビルド: デバッグ情報を含めてビルドします。    

```
icpx -fsycl -g -O2 sycl_basic_matmul.cpp -o sycl_basic_matmul
```
### プロファイリングの実行:
```
vtune -collect hotspots -result-dir vtune_result ./sycl_basic_matmul
```
### 結果の分析:
VTune ProfilerのGUIまたはCLIを使用して結果を確認。    
ホットスポット（実行時間の多くを占める部分）を特定。    
### 予想される結果    
内部ループ（for (size_t k = 0; k < N; k++)）がボトルネックであることが判明。    
メモリアクセスが非効率である可能性がある。    

## ステップ3: 最適化の実施
### 最適化1: ループタイル化によるメモリアクセスの改善
```
// カーネル内での変更
constexpr size_t TILE_SIZE = 16;

h.parallel_for<class matmul_tiled>(
    nd_range<2>({ N, N }, { TILE_SIZE, TILE_SIZE }),
    [=](nd_item<2> item) {

        size_t row = item.get_global_id(0);
        size_t col = item.get_global_id(1);

        float sum = 0.0f;
        for (size_t k = 0; k < N; k++) {
            sum += a[row][k] * b[k][col];
        }
        c[row][col] = sum;
    });
```
### 変更点
nd_rangeを使用して、ワークグループを定義。  
ワークグループサイズを{ TILE_SIZE, TILE_SIZE }に設定。    
### 効果
ワークグループ内でのデータの局所性が向上。    
しかし、このままではまだローカルメモリを活用していない。    
## 最適化2: ローカルメモリの活用    
```
h.parallel_for<class matmul_optimized>(
    nd_range<2>({ N, N }, { TILE_SIZE, TILE_SIZE }),
    [=](nd_item<2> item) {

        size_t row = item.get_global_id(0);
        size_t col = item.get_global_id(1);
        size_t local_row = item.get_local_id(0);
        size_t local_col = item.get_local_id(1);

        // ローカルメモリの定義
        local_accessor<float, 2> a_tile(range<2>(TILE_SIZE, TILE_SIZE), h);
        local_accessor<float, 2> b_tile(range<2>(TILE_SIZE, TILE_SIZE), h);

        float sum = 0.0f;

        for (size_t k = 0; k < N; k += TILE_SIZE) {
            // ローカルメモリにタイルをロード
            a_tile[local_row][local_col] = a[row][k + local_col];
            b_tile[local_row][local_col] = b[k + local_row][col];

            item.barrier(access::fence_space::local_space);

            // タイル内で計算
            for (size_t n = 0; n < TILE_SIZE; n++) {
                sum += a_tile[local_row][n] * b_tile[n][local_col];
            }

            item.barrier(access::fence_space::local_space);
        }

        c[row][col] = sum;
    });
```
### 変更点
ローカルアクセサ（local_accessor）**を使用してローカルメモリを確保。    
データをローカルメモリにロードし、ワークグループ内で共有。    
バリア（item.barrier）を使用して同期。    
### 効果
グローバルメモリへのアクセスを削減。    
メモリアクセスパターンを最適化し、キャッシュ効率を向上。    

## プロファイリング結果の再確認
最適化後のコードを再度プロファイルし、ボトルネックが改善されたか確認します。    
    
プロファイリングの実行    

```
vtune -collect hotspots -result-dir vtune_result_optimized ./sycl_tiling_localmemory
```
### 結果の分析
メモリアクセスの効率が向上していることを確認。    
依然としてボトルネックが存在する場合、さらなる最適化を検討。    

## 追加の最適化    
### 最適化3: ループアンローリング    
ループアンローリングを適用して、ループオーバーヘッドを削減できます。    
```
#pragma unroll
for (size_t n = 0; n < TILE_SIZE; n++) {
    sum += a_tile[local_row][n] * b_tile[n][local_col];
}
```
`#pragma unroll`を使用して、コンパイラにループのアンローリングを指示。    
パフォーマンスが向上する場合がありますが、逆効果になることもあるため、プロファイリングで効果を確認。  

## 実行例
```
$ icpx -fsycl sycl_basic_matmul.cpp -o sycl_basic_matmu
$ icpx -fsycl sycl_tiling.cpp -o sycl_tiling
$ icpx -fsycl sycl_tiling_localmemory.cpp -o sycl_tiling_localmemory
$ icpx -fsycl sycl_tiling_localmemory_loopunroling.cpp -o sycl_tiling_localmemory_loopunrolin
$ ./sycl_basic_matmul 
実行時間（基本実装）: 125 ms
$ ./sycl_tiling
実行時間（タイリング実装）: 74 ms
$ ./sycl_tiling_localmemory
実行時間（タイリング、ローカルメモリ実装）: 73 ms
$ ./sycl_tiling_localmemory_loopunroling 
実行時間（タイリング、ローカルメモリ、アンローリング実装）: 2671 m
```

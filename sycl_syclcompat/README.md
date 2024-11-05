# サンプルコード：syclcompatを使ったベクトル加算
## コードの説明
### syclcompatを使ったメモリ割り当て
sc::malloc はデバイス上のメモリを割り当てる関数で、cudaMallocと似た働きをします。
sc::free でデバイスメモリを解放します。
### syclcompatを使ったデータ転送
sc::memcpy を使って、ホストとデバイス間のデータ転送を行います（cudaMemcpyと似ています）。
### syclcompatを使ったカーネル起動
カーネルは sc::launch_kernel を使って起動し、CUDAのようにグリッドサイズとブロックサイズ、カーネル引数を指定します。
vector_add は2つのベクトルの要素を1つずつ加算して結果を保存するカーネル関数です。
### デバイスの同期
sc::device_synchronize() は、デバイス上の全ての処理が終了するまで待機する関数です（cudaDeviceSynchronizeと似ています）。
## コンパイルコマンド
IntelのDPC++/C++コンパイラicpxを使ってコンパイルする際は、syclcompatライブラリを含めます：

`icpx -fsycl -I$ONEAPI_ROOT/syclcompat/latest/include sample.cpp -o sample`
## プログラムの実行
コンパイル後、以下のように実行します：

`./sample`
実行が成功すると、次のように出力されます：

ベクトル加算が成功しました！
このサンプルコードは、CUDA風のAPIを提供するsyclcompatを利用して、CUDAコードをSYCLに移行する方法を示しています。メモリ管理、データ転送、カーネル起動のすべてをsyclcompatを使って行うことで、移行作業が簡単になります。

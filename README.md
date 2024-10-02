# sycl_101


## SYCLプログラムの基本構造
SYCLプログラムは、以下のような基本的な流れに従います。  

SYCLキュー（sycl::queue）を作成する 
デバイスに対してカーネル（並列に実行されるコード）をサブミットする  
メモリの管理と結果の取得    


## コードの解説
インクルードと名前空間の使用:   

#include <CL/sycl.hpp>：SYCLライブラリをインクルードします。    
using namespace sycl;：SYCLの名前空間を使用することで、コードを簡潔にします。   
データの準備:       

std::vector<int> A(N, 1)：要素が1のベクトルを作成。 
std::vector<int> B(N, 2)：要素が2のベクトルを作成。 
std::vector<int> C(N, 0)：結果を保存するためのベクトル。    
SYCLキューの作成:   

queue q;：デフォルトデバイスに対してSYCLキューを作成。デフォルトでは、SYCLは使用可能な最適なデバイス（例えば、GPUが優先される）を選びます。 
バッファの作成: 

buffer<int, 1> bufferA(A.data(), range<1>(N));：SYCLバッファは、ホストメモリからデバイスメモリへのデータ転送を抽象化します。範囲オブジェクトrange<1>(N)を指定して、サイズNの1次元バッファを作成します。 
カーネルの実行: 

q.submit([&](handler& h) {...})：この部分で、SYCLキューにカーネル（並列に実行される関数）をサブミットします。   
h.parallel_for(range<1>(N), [=](id<1> i) {...})：並列に実行されるループです。range<1>(N)は、ベクトルのサイズを指定し、id<1> iでインデックスを表します。 
結果の取得と表示:   

カーネルの実行後、bufferCのデータがCにコピーされるため、ホスト側で結果を使用できます。最後に、最初の10要素を表示しています。    

## 実行方法
SYCLプログラムを実行するには、SYCL対応のコンパイラ（例えばIntelのoneAPI DPC++またはCodeplayのComputeCpp）が必要です。   

IntelのDPC++を使用する場合は、以下のようにコンパイルできます。  

dpcpp -o vector_add vector_add.cpp  
./vector_add    

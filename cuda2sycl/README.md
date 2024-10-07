# CUDAからSYCL移植

## cudaコードのコンパイルと実行
```
nvcc cuda2sycl.cu   
./a.out 
デフォルトのCUDAデバイス: NVIDIA GeForce RTX 3060 Ti
計算完了
```

## dpctによる変換手順
### 1. dpctのインストール   
dpctはIntel® oneAPI DPC++ Compilerに含まれています。    
oneAPI Toolkitからダウンロードできます。    
### 2. CUDAコードの変換    
コマンドラインから以下のコマンドを実行して変換します。  
```
dpct --cuda-path=/usr/local/cuda/include --in-root=. --out-root=./dpct_output cuda2sycl.cu
```
--cuda-include-pathはCUDAのヘッダーファイルへのパスを指定します。   
-in-rootと-out-rootは入力ファイルと出力先ディレクトリを指定します。 
### 3. 変換後のコードの確認と修正  
dpctは自動変換を行いますが、完全ではない場合があります。    
変換後のコードを確認し、必要に応じて手動で修正します。  
### 4. ビルドと実行    
変換後のコードをビルドするには、DPC++コンパイラを使用します。   
`icpx -fsycl cuda2sycl.dp.cpp -o cuda2sycl_add`
プログラムを実行して、正しく動作することを確認します。  
```
./cuda2sycl
デフォルトのCUDAデバイス: 12th Gen Intel(R) Core(TM) i5-12400
計算完了
```


# 概要
SYCLはC++ベースの標準ですが、CからSYCLコードを呼び出すことも可能です。C言語では、SYCLのC++関数を利用するために外部Cインターフェース（extern "C"）を使うことが一般的です。       
これはSYCLの基本的なベクトル加算をCから呼び出すサンプルコードです。 

## コンパイルと実行
1. SYCL C++コードをコンパイル: SYCLコード（sycl_vector_add.cpp）をコンパイルします。icpxを使って共有ライブラリとしてコンパイルします。   
```icpx -fsycl -fPIC -shared -o libsycl_vector_add.so sycl_vector_add.cpp```      

2. Cコードをコンパイル: Cコード（main.c）をコンパイルします。   
```icx main.c -L. -lsycl_vector_add -o main```  
-L.: カレントディレクトリから共有ライブラリを検索。     
-lsycl_vector_add: libsycl_vector_add.soをリンク。  

3. 実行: 実行可能ファイルを実行します。   
```./main```

ex. Windows環境では以下のコマンドでコンパイル  
```
icx-cl -fsycl -c sycl_vector_add.cpp -o sycl_vector_add.obj    
lib /OUT:libsycl_vector_add.lib sycl_vector_add.obj    
icx-cl -fsycl main.c libsycl_vector_add.lib -o main.exe    
```  


## 実行結果  
``` Vector Addition Results:
c[0] = 0.000000
c[1] = 3.000000
c[2] = 6.000000
c[3] = 9.000000
c[4] = 12.000000
c[5] = 15.000000
c[6] = 18.000000
c[7] = 21.000000
c[8] = 24.000000
c[9] = 27.000000
```

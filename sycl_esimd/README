# コードの解説
## 1. キューの作成
`queue q{ gpu_selector{} };'
GPUデバイスを選択してキューを作成します。
USMを使用する際、メモリの割り当てにキューが必要となるため、キューの作成を先に行います。
## 2. USMメモリの割り当て
```
float* A = malloc_shared<float>(size, q);
float* B = malloc_shared<float>(size, q);
float* C = malloc_shared<float>(size, q);
```
malloc_sharedを使用して、ホストとデバイス間で共有されるメモリを割り当てます。
sizeは配列の要素数で、qは関連付けるキューです。
## 3. データの初期化
```
for (unsigned int i = 0; i < size; ++i) {
    A[i] = 1.0f;
    B[i] = 2.0f;
    C[i] = 0.0f;
}
```
ホスト側でUSMメモリにデータを初期化します。
## 4. カーネルの実行
```
q.submit([&](handler& cgh) {
    cgh.parallel_for<class esimd_vector_add>(
        range<1>(size / 16), [=](id<1> i) SYCL_ESIMD_KERNEL {
            // SIMDサイズを定義
            constexpr unsigned int VL = 16;

            // オフセットの計算
            unsigned int offset = i[0] * VL;

            // メモリからロード
            simd<float, VL> va;
            simd<float, VL> vb;
            va.copy_from(A + offset);
            vb.copy_from(B + offset);

            // 要素ごとの加算
            simd<float, VL> vc = va + vb;

            // メモリにストア
            vc.copy_to(C + offset);
        });
});
```
parallel_forを使用してカーネルを定義し、ESIMDカーネルであることをSYCL_ESIMD_KERNELマクロで指定します。
USMポインタA、B、Cを直接カーネル内で使用します。
simdオブジェクトを使用してSIMD演算を行います。
## 5. 結果の検証
```
bool passed = true;
for (size_t i = 0; i < size; i++) {
    if (C[i] != A[i] + B[i]) {
        passed = false;
        break;
    }
}

if (passed) {
    std::cout << "計算結果は正しいです。" << std::endl;
} else {
    std::cout << "計算結果に誤りがあります。" << std::endl;
}
```
ホスト側でUSMメモリCの内容を検証します。
A[i] + B[i]とC[i]を比較して、一致しない場合はエラーを報告します。
## 6. USMメモリの解放
```
sycl::free(A, q);
sycl::free(B, q);
sycl::free(C, q);
```
使用したUSMメモリを解放します。
sycl::free関数にメモリポインタと関連付けられたキューを渡します。

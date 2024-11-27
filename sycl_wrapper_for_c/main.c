#include <stdio.h>
#include "sycl_vector_add.h"

int main() {
    const int N = 10;
    float a[10], b[10], c[10];

    // ベクトルの初期化
    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0f;
        b[i] = i * 2.0f;
    }

    // SYCLを使ったベクトル加算を呼び出し
    vector_add(a, b, c, N);

    // 結果の表示
    printf("Vector Addition Results:\n");
    for (int i = 0; i < N; i++) {
        printf("c[%d] = %f\n", i, c[i]);
    }

    return 0;
}

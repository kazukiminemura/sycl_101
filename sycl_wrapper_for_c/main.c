#include <stdio.h>

// SYCLの関数を呼び出すための宣言
extern void vector_add(const float *a, const float *b, float *c, int n);

int main() {
    const int N = 10;
    float a[N], b[N], c[N];

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

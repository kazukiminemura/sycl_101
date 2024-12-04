#include <stdio.h>
#include "sycl_wrapper.h"
#include "sycl_vector_add.h"

int sycl_wrapper()
{
	const int N = SIZE;
	float a[SIZE], b[SIZE], c[SIZE];
	// ベクトルの初期化
	for (int i = 0; i < N; i++)
	{
		a[i] = i * 1.0f;
		b[i] = i * 2.0f;
	}

	// SYCLを使ったベクトル加算を呼び出し
	vector_add(a, b, c, N);
	// c_vector_add(a, b, c, N);

	// 結果の表示
	printf("Vector Addition Results:\n");
	for (int i = 0; i < N; i++)
	{
		printf("c[%d] = %f\n", i, c[i]);
	}

	return 0;
}

#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;

int main() {
    const int N = 4096;
    const int TILE_SIZE = 16;  // ワークグループのサイズ
    std::vector<int> A(N * N, 1);
    std::vector<int> B(N * N, 2);
    std::vector<int> C(N * N, 0);

    // デバイスキューを作成
    queue q;
    std::cout << q.get_device().get_info<info::device::name>() << std::endl;

    // USMを使って共有メモリを確保
    int* ptrA = malloc_shared<int>(N * N, q);
    int* ptrB = malloc_shared<int>(N * N, q);
    int* ptrC = malloc_shared<int>(N * N, q);

    // データをホストからコピー
    std::copy(A.begin(), A.end(), ptrA);
    std::copy(B.begin(), B.end(), ptrB);

    // カーネル実行
    q.submit([&](handler& h) {
        // ローカルメモリの使用
        local_accessor<int, 2> localA(range<2>(TILE_SIZE, TILE_SIZE), h);
        local_accessor<int, 2> localB(range<2>(TILE_SIZE, TILE_SIZE), h);

        // ワークグループサイズを設定してカーネルを最適化
        h.parallel_for(nd_range<2>(range<2>(N, N), range<2>(TILE_SIZE, TILE_SIZE)), [=](nd_item<2> item) {
            int row = item.get_global_id(0);
            int col = item.get_global_id(1);

            int local_row = item.get_local_id(0);
            int local_col = item.get_local_id(1);

            // ローカルメモリにデータをロード
            localA[local_row][local_col] = ptrA[row * N + col];
            localB[local_row][local_col] = ptrB[row * N + col];

            // ローカルメモリのバリア
            item.barrier(access::fence_space::local_space);

            // 結果をグローバルメモリに書き込む
            ptrC[row * N + col] = localA[local_row][local_col] + localB[local_row][local_col];
        });
    }).wait();  // カーネル完了まで待機

    // 結果の一部を表示
    std::cout << "C[0] = " << ptrC[0] << std::endl;
    std::cout << "C[N-1] = " << ptrC[N * N - 1] << std::endl;

    // USMメモリの解放
    free(ptrA, q);
    free(ptrB, q);
    free(ptrC, q);

    return 0;
}


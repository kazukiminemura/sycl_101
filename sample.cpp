#include <CL/sycl.hpp>
#include <iostream>

using namespace sycl;

int main() {
    const int N = 1024;
    std::vector<int> A(N, 1);  // サイズNのベクトルA、全ての要素を1で初期化
    std::vector<int> B(N, 2);  // サイズNのベクトルB、全ての要素を2で初期化
    std::vector<int> C(N, 0);  // 結果を格納するベクトルC

    // SYCLキューを作成（デフォルトデバイス、GPUまたはCPU）
    queue q;

    // デバイスバッファを作成
    buffer<int, 1> bufferA(A.data(), range<1>(N));
    buffer<int, 1> bufferB(B.data(), range<1>(N));
    buffer<int, 1> bufferC(C.data(), range<1>(N));

    // キューにコマンドをサブミット
    q.submit([&](handler& h) {
        // バッファからアクセス許可を得る
        auto accA = bufferA.get_access<access::mode::read>(h);
        auto accB = bufferB.get_access<access::mode::read>(h);
        auto accC = bufferC.get_access<access::mode::write>(h);

        // カーネルを定義
        h.parallel_for(range<1>(N), [=](id<1> i) {
            accC[i] = accA[i] + accB[i];  // ベクトルの加算を行う
        });
    }).wait();  // 実行完了まで待つ

    // 結果を表示
    for (int i = 0; i < 10; i++) {
        std::cout << "C[" << i << "] = " << C[i] << std::endl;
    }

    return 0;
}

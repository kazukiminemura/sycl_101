#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;

int main() {
    const int N = 4096;
    std::vector<int> A(N * N, 1);
    std::vector<int> B(N * N, 2);
    std::vector<int> C(N * N, 0);

    queue q;
    std::cout << q.get_device().get_info<info::device::name>() << std::endl;

    int* ptrA = malloc_shared<int>(N * N, q);
    int* ptrB = malloc_shared<int>(N * N, q);
    int* ptrC = malloc_shared<int>(N * N, q);

    // データをコピー
    std::copy(A.begin(), A.end(), ptrA);
    std::copy(B.begin(), B.end(), ptrB);

    // カーネルの実行
    q.parallel_for(range<2>(N, N), [=](id<2> i) {
        ptrC[i[0] * N + i[1]] = ptrA[i[0] * N + i[1]] + ptrB[i[0] * N + i[1]];
    }).wait();

    std::cout << "C[0] = " << ptrC[0] << std::endl;
    std::cout << "C[N-1] = " << ptrC[N * N - 1] << std::endl;

    free(ptrA, q);
    free(ptrB, q);
    free(ptrC, q);

    return 0;
}


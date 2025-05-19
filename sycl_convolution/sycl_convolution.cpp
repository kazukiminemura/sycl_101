#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

int main() {
    constexpr size_t width = 5;
    constexpr size_t height = 5;
    const std::vector<int> image = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25
    };

    const std::vector<int> kernel = {
        1, 0, -1,
        2, 0, -2,
        1, 0, -1
    };

    std::vector<int> result(width * height, 0);

    sycl::queue queue;

    // USM メモリの割り当て
    int* imageUSM = sycl::malloc_shared<int>(width * height, queue);
    int* kernelUSM = sycl::malloc_shared<int>(3 * 3, queue);
    int* resultUSM = sycl::malloc_shared<int>(width * height, queue);

    // ホストから USM メモリにデータをコピー
    std::copy(image.begin(), image.end(), imageUSM);
    std::copy(kernel.begin(), kernel.end(), kernelUSM);

    // カーネルの実行（queue.submit を使用）
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for<class image_convolution>(
            sycl::range<2>(width, height),
            [=] (sycl::item<2> item) {
                int sum = 0;
                for (int kx = -1; kx <= 1; ++kx) {
                    for (int ky = -1; ky <= 1; ++ky) {
                        int imageX = item[0] + kx;
                        int imageY = item[1] + ky;
                        if (imageX >= 0 && imageX < width && imageY >= 0 && imageY < height) {
                            sum += imageUSM[imageX * width + imageY] * kernelUSM[(kx + 1) * 3 + (ky + 1)];
                        }
                    }
                }
                resultUSM[item[0] * width + item[1]] = sum;
            });
    });

    queue.wait();

    // 結果をホストの result ベクターにコピー
    std::copy(resultUSM, resultUSM + width * height, result.begin());

    std::cout << "Convolution Result:" << std::endl;
    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            std::cout << result[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    // USM メモリの解放
    sycl::free(imageUSM, queue);
    sycl::free(kernelUSM, queue);
    sycl::free(resultUSM, queue);

    return 0;
}

#include <sycl/sycl.hpp>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <iostream>
#include <vector>

using namespace sycl;

int main() {
    // Image setup
    const int width = 512;
    const int height = 512;
    const int channels = 1;
    const size_t imageSize = width * height * channels;

    // SYCL queue
    queue q;
    std::cout << "Running on: "
              << q.get_device().get_info<info::device::name>()
              << std::endl;

    // DPLではBufferは想定されていないのでUSMを使う
    {
        // USM メモリを確保（shared memory）
        unsigned char* image = sycl::malloc_shared<unsigned char>(imageSize, q);

        // 初期化
        std::fill(image, image + imageSize, 128);

        // oneDPL iterator wrapper
        auto policy = oneapi::dpl::execution::make_device_policy(q);

        oneapi::dpl::for_each(
            policy,
            oneapi::dpl::counting_iterator<size_t>(0),
            oneapi::dpl::counting_iterator<size_t>(imageSize - 2),  // 最後の2要素はblurできない
            [=](size_t i) {
                image[i] = (image[i] + image[i + 1] + image[i + 2]) / 3;
            }
        );
    }

    // Save (mock)
    std::cout << "Saving image to blurred_onedpl.png" << std::endl;

    return 0;
}

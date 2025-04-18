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
    std::vector<unsigned char> imageData(width * height * channels, 128);
    size_t imageSize = imageData.size();

    // SYCL queue
    queue q;
    std::cout << "Running on: " 
              << q.get_device().get_info<info::device::name>() 
              << std::endl;

    {
        // oneDPL + buffer management
        buffer<unsigned char> imageBuffer(imageData.data(), range<1>(imageSize));

        // oneDPL iterator wrapper
        auto policy = oneapi::dpl::execution::make_device_policy(q);

	
        oneapi::dpl::for_each(
            policy,
            oneapi::dpl::counting_iterator<size_t>(0),
            oneapi::dpl::counting_iterator<size_t>(imageSize - 2),  // 最後の2要素はblurできない
            [=, acc = imageBuffer.get_access<sycl::access::mode::read_write>()](size_t i) {
                acc[i] = (acc[i] + acc[i + 1] + acc[i + 2]) / 3;
            }
        );


    }

    // Save (mock)
    std::cout << "Saving image to blurred_onedpl.png" << std::endl;

    return 0;
}


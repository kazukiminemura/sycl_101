#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

using namespace sycl;

int main(){

  // Dummy image data
  const int width = 512;
  const int height = 512;
  const int channels = 1;
  std::vector<unsigned char> imageData(width * height * channels, 128); // Initialize with dummy data
  size_t imageSize = imageData.size();

  // SYCL setup
  queue q;
  std::cout << q.get_device().get_info<info::device::name>() << std::endl;

  // Execute kernel
  buffer<unsigned char, 1> imageBuffer(imageData.data(), range<1>(imageSize));
  q.submit([&](handler& h) {
    auto image = imageBuffer.get_access<access::mode::read_write>(h);
    h.parallel_for(range<1>(imageSize), [=] (id<1> i) {
        // Simple blur logic
        image[i] = (image[i] + image[i+1] + image[i+2]) / 3;
    });
  }).wait();


  // Save image data
  // Dummy save function (replace with actual implementation)
  std::cout << "Saving image to " << "blurred_sycl.png" << std::endl;


}

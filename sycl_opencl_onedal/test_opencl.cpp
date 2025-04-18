#include <CL/cl2.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

int main() {
    // Dummy image data
    const int width     = 512;
    const int height    = 512;
    const int channels  = 1;
    size_t imageSize    = size_t(width) * height * channels;
    std::vector<unsigned char> imageData(imageSize, 128);

    // 1) Get all platforms and pick the first
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "No OpenCL platforms found.\n";
        return 1;
    }

    cl::Platform platform = platforms.front();
    std::cout << "Using platform: " 
              << platform.getInfo<CL_PLATFORM_NAME>() 
              << "\n";

    // 2) Get a GPU device and create a context + queue
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        std::cerr << "No GPU devices found; trying CPU...\n";
        platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
        if (devices.empty()) {
            std::cerr << "No OpenCL devices available.\n";
            return 1;
        }
    }
    cl::Device device = devices.front();
    std::cout << "Using device: " 
              << device.getInfo<CL_DEVICE_NAME>() 
              << "\n";

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    // 3) Load & build the kernel source
    std::ifstream file("blur.cl");
    if (!file.is_open()) {
        std::cerr << "Failed to load kernel file.\n";
        return 1;
    }
    std::ostringstream oss;
    oss << file.rdbuf();
    std::string src = oss.str();
    cl::Program program(context, src);
    if (program.build({device}) != CL_SUCCESS) {
        std::cerr << "Build log:\n"
                  << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device)
                  << "\n";
        return 1;
    }

    // 4) Create a buffer and copy image data
    cl::Buffer bufImage(
        context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        imageSize * sizeof(unsigned char),
        imageData.data()
    );

    // 5) Set up and launch the kernel
    cl::Kernel kernel(program, "blur");
    kernel.setArg(0, bufImage);
    kernel.setArg(1, static_cast<unsigned int>(imageSize));

    cl::NDRange global(imageSize);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, global);
    queue.finish();

    // 6) Read back results
    queue.enqueueReadBuffer(
        bufImage, CL_TRUE, 0,
        imageSize * sizeof(unsigned char),
        imageData.data()
    );

    // Dummy “save” placeholder
    std::cout << "Saving image to blurred_opencl.png\n";
    //…insert your image‐writing routine here…
    //
    //

    return 0;
}

#include <oneapi/dpl/algorithm> // oneapi::dpl::transform
#include <oneapi/dpl/execution> // oneapi::dpl::execution::make_device_policy

#include <sycl/sycl.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace sycl;
using namespace cv;

void grayscale_onedpl(queue& q, uchar4* input, uchar* output, int width, int height) {
    // 1. Create a oneDPL policy which executes on the provided SYCL queue.
    auto policy = oneapi::dpl::execution::make_device_policy(q);
    std::size_t num_pixels = width * height;
    // 2. Perform a transformation over each pixel.
    oneapi::dpl::transform(policy, input, input + num_pixels, output, [](uchar4 pixel) {
        uchar gray = static_cast<uchar>(
            0.299f * pixel.x() + 0.587f * pixel.y() + 0.114f * pixel.z()
        );
        return gray;
    });
    // There is an implicit wait, so no q.wait() is needed.
}

int main() {
    // 入力画像読み込み（カラー）
    Mat img = imread("input.png");
    if (img.empty()) {
        std::cerr << "Failed to load image.\n";
        return -1;
    }
    // RGBAに変換
    Mat rgba_img;
    cvtColor(img, rgba_img, COLOR_BGR2RGBA);

    int width = rgba_img.cols;
    int height = rgba_img.rows;

    queue q;
    // SYCL用ポインタ確保
    uchar4* input_data = static_cast<uchar4*>(malloc_shared(sizeof(uchar4) * width * height, q));
    uchar* output_data = static_cast<uchar*>(malloc_shared(sizeof(uchar) * width * height, q));

    // OpenCVからSYCLへコピー
    std::memcpy(input_data, rgba_img.data, sizeof(uchar4) * width * height);

    // SYCL実行
    grayscale_onedpl(q, input_data, output_data, width, height);

    // 結果をOpenCV形式に戻して保存
    Mat gray(height, width, CV_8UC1, output_data);
    imwrite("grayscale_onedpl.png", gray);

    // メモリ解放
    free(input_data, q.get_context());
    free(output_data, q.get_context());

    std::cout << "Grayscale conversion on oneDPL is done.\n";
    return 0;
}

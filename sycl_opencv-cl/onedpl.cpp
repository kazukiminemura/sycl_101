#include <sycl/sycl.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

using namespace sycl;
using namespace cv;

void grayscale_dpl(queue& q, uchar4* input, uchar* output, int width, int height) {
    auto policy = oneapi::dpl::execution::make_device_policy(q);

    oneapi::dpl::for_each(policy,
            oneapi::dpl::counting_iterator<size_t>(0),
            oneapi::dpl::counting_iterator<size_t>(width * height),
            [=] (size_t i) {
        uchar gray = static_cast<uchar>(
            0.299f * input[i].x() + 0.587f * input[i].y() + 0.114f * input[i].z()
        );
        output[i] = gray;
    });
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
    grayscale_dpl(q, input_data, output_data, width, height);
    q.wait();

    // 結果をOpenCV形式に戻して保存
    Mat gray(height, width, CV_8UC1, output_data);
    imwrite("grayscale_dpl.png", gray);

    // メモリ解放
    free(input_data, q.get_context());
    free(output_data, q.get_context());

    std::cout << "Grayscale conversion on oneDPL is done.\n";
    return 0;
}


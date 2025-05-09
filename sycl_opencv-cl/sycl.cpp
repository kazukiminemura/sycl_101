#include <sycl/sycl.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace sycl;
using namespace cv;

void grayscale_sycl(queue& q, uchar4* input, uchar* output, int width, int height) {
    q.submit([&](handler& h) {
        h.parallel_for(range<2>(height, width), [=](id<2> idx) {
            int y = idx[0];
            int x = idx[1];
            int i = y * width + x;

            uchar4 pixel = input[i];
            uchar gray = static_cast<uchar>(
                0.299f * pixel.x() + 0.587f * pixel.y() + 0.114f * pixel.z()
            );
            // uchar gray = 0.0f;
            output[i] = gray;
        });
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
    grayscale_sycl(q, input_data, output_data, width, height);
    q.wait();

    // 結果をOpenCV形式に戻して保存
    Mat gray(height, width, CV_8UC1, output_data);
    imwrite("grayscale_sycl.png", gray);

    // メモリ解放
    free(input_data, q.get_context());
    free(output_data, q.get_context());

    std::cout << "Grayscale conversion on SYCL is done.\n";
    return 0;
}

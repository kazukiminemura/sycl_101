#include <sycl/sycl.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace sycl;
using namespace cv;

// もらった関数の宣言（そのまま使う）
void grayscale_sycl(queue& q, uchar4* input, uchar* output, int width, int height) {
    buffer<uchar4, 1> input_buf(input, range<1>(width * height));
    buffer<uchar, 1> output_buf(output, range<1>(width * height));

    q.submit([&](handler& h) {
        auto in = input_buf.get_access<access::mode::read>(h);
        auto out = output_buf.get_access<access::mode::write>(h);

        h.parallel_for(range<2>(height, width), [=](id<2> idx) {
            int y = idx[0];
            int x = idx[1];
            int i = y * width + x;

            uchar4 pixel = in[i];
            uchar gray = static_cast<uchar>(
                0.299f * pixel.x() + 0.587f * pixel.y() + 0.114f * pixel.z()
            );
            out[i] = gray;
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

    // 結果をOpenCV形式に戻す
    Mat gray(height, width, CV_8UC1, output_data);
    imwrite("gray_output.png", gray);

    // メモリ解放
    free(input_data, q.get_context());
    free(output_data, q.get_context());

    std::cout << "Grayscale conversion done.\n";
    return 0;
}

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

// カーネルの読み込み関数
String loadKernelSource(const String& path) {
    std::ifstream file(path);
    if (!file.is_open()) throw runtime_error("Failed to open kernel file.");
    return String((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

int main() {
    // OpenCLが利用可能か確認
    if (!cv::ocl::haveOpenCL()) {
        cout << "OpenCL not available" << endl;
        return -1;
    }

    // OpenCLを有効に
    cv::ocl::setUseOpenCL(true);
    cout << "OpenCL is " << (cv::ocl::useOpenCL() ? "enabled" : "disabled") << endl;

    // 入力画像を読み
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

    // UMatに転送（OpenCLバックエンド利用）
    UMat uInput, uOutput;
    rgba_img.copyTo(uInput);
    uOutput.create(rgba_img.size(), CV_8UC1);

    // カーネルのソースを読み込む
    String kernelSrc = loadKernelSource("grayscale.cl");

    // OpenCLコンテキストとプログラム作成
    cv::ocl::ProgramSource ps(kernelSrc);
    cv::String errmsg;
    cv::ocl::Program program = cv::ocl::Context::getDefault().getProg(ps, "", errmsg);

    // カーネル作成
    cv::ocl::Kernel kernel("grayscale", program);

    // 引数をセット
    // KernelArg::ReadWrite(umat_dst)	uchar* ptr, int step, int offset, int rows, int cols
    // 
    kernel.args(
        cv::ocl::KernelArg::ReadOnlyNoSize(uInput),
        cv::ocl::KernelArg::WriteOnly(uOutput)
    );

    // 実行
    size_t globalSize[] = {(size_t)width, (size_t)height};
    if (!kernel.run(2, globalSize, NULL, true)) {
        cerr << "Kernel execution failed." << endl;
        return -1;
    }

    // 結果をOpenCV形式に戻して保存
    Mat gray(height, width, CV_8UC1, uOutput);
    imwrite("grayscale_cvcl.png", gray);

    std::cout << "Grayscale conversion on OpenCV-CL is done.\n";
    return 0;
}

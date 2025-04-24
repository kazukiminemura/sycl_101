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

    // 入力画像を読み込んでRGBAに変換
    Mat img = imread("input.png");
    if (img.empty()) return -1;
    cvtColor(img, img, COLOR_BGR2RGBA);

    // UMatに転送（OpenCLバックエンド利用）
    UMat uInput, uOutput;
    img.copyTo(uInput);
    uOutput.create(img.size(), CV_8UC1);

    // カーネルのソースを読み込む
    String kernelSrc = loadKernelSource("grayscale.cl");

    // OpenCLコンテキストとプログラム作成
    cv::ocl::ProgramSource ps(kernelSrc);
    cv::String errmsg;
    cv::ocl::Program program = cv::ocl::Context::getDefault().getProg(ps, "", errmsg);

    // カーネル作成
    cv::ocl::Kernel kernel("grayscale", program);

    // 引数をセット
    int width = img.cols;
    int height = img.rows;
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

    // 出力をMatに変換して保存
    Mat result;
    uOutput.copyTo(result);
    imwrite("grayscale_result.png", result);

    cout << "Done." << endl;
    return 0;
}

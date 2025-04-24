// OpenCL kernel: grayscale.cl
__kernel void grayscale(__global uchar4* input, __global uchar* output, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < width && y < height) {
        int idx = y * width + x;
        uchar4 pixel = input[idx];
        uchar gray = (uchar)(0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z);
        output[idx] = gray;
    }
}
__kernel void blur(
    __global unsigned char* image,
    const unsigned int imageSize)
{
    size_t i = get_global_id(0);
    // Prevent out‐of‐bounds at the very end
    if (i + 2 < imageSize) {
        image[i] = (image[i] + image[i+1] + image[i+2]) / 3;
    }
}


```
icpx -o ocl_gray opencv-cl.cpp `pkg-config --cflags --libs opencv4`

icpx -fsycl sycl.cpp -o sycl_gray `pkg-config --cflags --libs opencv4`

// ImageMagick
./ocl_gray; ./sycl_gray; compare -metric AE ./grayscale_cvcl.png ./grayscale_sycl.png ./difference.png

OpenCL is enabled
Done.
Grayscale conversion done.
0

```

```
icpx -o ocl_gray opencv-cl.cpp `pkg-config --cflags --libs opencv4`

icpx -fsycl sycl.cpp -o sycl_gray `pkg-config --cflags --libs opencv4`

icpx -fsycl dpl.cpp -o dpl_gray `pkg-config --cflags --libs opencv4`

// ImageMagick
./ocl_gray; ./sycl_gray; compare -metric AE ./grayscale_cvcl.png ./grayscale_sycl.png ./difference.png
./ocl_gray; ./dpl_gray; compare -metric AE ./grayscale_cvcl.png ./grayscale_onedpl.png ./difference.png

OpenCL is enabled
Grayscale conversion on OpenCV-CL is done.
Grayscale conversion on SYCL is done.
0

```

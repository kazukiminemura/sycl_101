```
icpx -o ocl_gray opencv-cl.cpp `pkg-config --cflags --libs opencv4`

icpx -fsycl sycl.cpp -o sycl_gray `pkg-config --cflags --libs opencv4`
```

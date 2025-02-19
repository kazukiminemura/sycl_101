mkdir build
cd build
cmake -G Ninja -DCMAKE_CXX_COMPILER=icpx ..
cmake --build .
./simple

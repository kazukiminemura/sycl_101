mkdir build
cd build
cmake -GNinja -DCMAKE_CXX_COMPILER=icpx ..
cmake --build .
./simple

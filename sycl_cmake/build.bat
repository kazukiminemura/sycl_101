mkdir build
cd build
cmake -GNinja -DCMAKE_CXX_COMPILER=icx ..
cmake --build .
.\simple.exe

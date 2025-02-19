# Linux
mkdir build  
cd build  
cmake -G Ninja -DCMAKE_CXX_COMPILER=icpx ..  
cmake --build .  
./simple  

# Windows
mkdir build  
cd build  
cmake -G Ninja -DCMAKE_CXX_COMPILER=icx ..  
cmake --build .  
.\simple.exe  

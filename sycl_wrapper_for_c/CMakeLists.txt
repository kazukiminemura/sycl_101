# Minimum CMake version required
cmake_minimum_required(VERSION 3.21)

# Project name and languages
project(SyclWrapperSample LANGUAGES C CXX)

# Set the SYCL compiler (Intel DPC++)
set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_C_COMPILER icx)

# Enable SYCL support for the compiler
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

# Add the SYCL shared library
add_library(sycl_vector_add SHARED sycl_vector_add.cpp)

# Add the main C program and link the SYCL library
add_executable(main_exec main.c)
target_link_libraries(main_exec PRIVATE sycl_vector_add)

# Specify runtime library path for the executable
set_target_properties(main_exec PROPERTIES
    INSTALL_RPATH "$ORIGIN"
)

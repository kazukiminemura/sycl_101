# Minimum CMake version required
cmake_minimum_required(VERSION 3.21)

# Project name and languages
project(SyclWrapperSample LANGUAGES C CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set the Intel oneAPI DPC++/C++ Compiler
set(CMAKE_CXX_COMPILER "icx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

# Add the SYCL source file
set(SYCL_SOURCE sycl_vector_add.cpp)

# Add the C source file
set(C_SOURCE main.c)

# Create the SYCL library
add_library(sycl_vector_add STATIC ${SYCL_SOURCE})

# Link the SYCL library with the SYCL runtime
target_link_libraries(sycl_vector_add PRIVATE sycl)

# Create the executable
add_executable(main ${C_SOURCE})

# Link the executable with the SYCL library
target_link_libraries(main PRIVATE sycl_vector_add)

# Include directories for SYCL
target_include_directories(sycl_vector_add PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(main PRIVATE ${CMAKE_SOURCE_DIR})

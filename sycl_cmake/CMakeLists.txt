if (CMAKE_HOST_WIN32)
    # need at least CMake 3.25 for IntelLLVM support of IntelSYCL package on Windows
    cmake_minimum_required(VERSION 3.25)
else()
    # CMake 3.23.5 is the minimum recommended for IntelLLVM on Linux
    cmake_minimum_required(VERSION 3.23.5)
endif()

project(simple-sycl LANGUAGES CXX)

find_package(IntelSYCL REQUIRED)

add_executable(simple simple.cpp)
add_sycl_to_target(TARGET simple  SOURCES simple.cpp )
cmake_minimum_required(VERSION 3.15)
project(knn_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(pybind11 REQUIRED)

pybind11_add_module(knn_cpp bindings.cpp)
target_compile_options(knn_cpp PRIVATE -O3)

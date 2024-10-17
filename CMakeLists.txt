cmake_minimum_required(VERSION 3.10)
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

#### CUDA related
enable_language(CUDA)
set(CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")

#### find_package PATH config
## pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})

execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")

find_package(pybind11 REQUIRED)

###################################################################################
project(litetorch LANGUAGES C CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

pybind11_add_module(ndarray_backend_cpu ./src/ndarray_backend_cpu.cc) 
pybind11_add_module(ndarray_backend_cuda ./src/ndarray_backend_cuda.cu) 

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUGMODE)
    set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
endif()

cmake_minimum_required(VERSION 3.14)
include(FetchContent)
project(PyTorch)
set(CMAKE_CXX_STANDARD 14)

FetchContent_Declare(
  Torch
  URL           https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcu101.zip
  URL_HASH      MD5=b18d6f2fdfe730c1fca3edd5ed2f3c8f
)
FetchContent_MakeAvailable(Torch)

set (CMAKE_MODULE_PATH "_deps;${CMAKE_MODULE_PATH}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(PyTorch_LIBRARIES PyTorch "${TORCH_LIBRARIES}")

if(NOT ITK_SOURCE_DIR)
  find_package(ITK REQUIRED)
  list(APPEND CMAKE_MODULE_PATH ${ITK_CMAKE_DIR})
  include(ITKModuleExternal)
else()
  set(ITK_DIR ${CMAKE_BINARY_DIR})
  itk_module_impl()
endif()

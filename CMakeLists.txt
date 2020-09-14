cmake_minimum_required(VERSION 3.14)

# The C++ version of PyTorch is also known as libtorch.
project(PyTorch)

# PyTorch requires C++-14.
# As of 2020-August-19, PyTorch requires gcc-6, gcc-7, or gcc-8.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED USE_CUDA AND DEFINED ENV{USE_CUDA})
  set(USE_CUDA $ENV{USE_CUDA})
endif()
file(GLOB_RECURSE where /*TorchConfig.cmake*)
message(STATUS "where=${where}")
message(STATUS "CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")
message(STATUS "Torch_DIR=${Torch_DIR}")
message(STATUS "Caffe2_DIR=${Caffe2_DIR}")
message(STATUS "ENV{CMAKE_PREFIX_PATH}=$ENV{CMAKE_PREFIX_PATH}")
message(STATUS "ENV{Torch_DIR}=$ENV{Torch_DIR}")
message(STATUS "ENV{Caffe2_DIR}=$ENV{Caffe2_DIR}")
message(STATUS "ITK_DIR=${ITK_DIR}")
find_package(Torch REQUIRED CONFIG)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(PyTorch_LIBRARIES "${TORCH_LIBRARIES}")
include_directories(
  ${Torch_DIR}/../../../include/torch/csrc/api/include
  ${Torch_DIR}/../../../include
  )

if(NOT ITK_SOURCE_DIR)
  find_package(ITK REQUIRED)
  list(APPEND CMAKE_MODULE_PATH ${ITK_CMAKE_DIR})
  include(ITKModuleExternal)
else()
  set(ITK_DIR ${CMAKE_BINARY_DIR})
  itk_module_impl()
endif()

if(0)
  get_cmake_property(_variableNames VARIABLES)
  list (SORT _variableNames)
  foreach (_variableName ${_variableNames})
      message(STATUS "${_variableName}=${${_variableName}}")
  endforeach()
endif()

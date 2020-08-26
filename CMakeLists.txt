cmake_minimum_required(VERSION 3.14)

# The C++ version of PyTorch is also known as libtorch.
project(PyTorch)

# PyTorch requires C++-14.
# As of 2020-August-19, PyTorch requires gcc-6, gcc-7, or gcc-8.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# We will need to use the location ${Python_EXECUTABLE}
find_package(Python REQUIRED)

# Get PyTorch and its dependencies, git recursively
set(torch_fetch_git_repository "https://github.com/pytorch/pytorch")
set(torch_fetch_git_tag "v1.5.1")
include(FetchContent)
FetchContent_Declare(
  torch_fetch
  GIT_REPOSITORY "${torch_fetch_git_repository}"
  GIT_TAG        "${torch_fetch_git_tag}"
)
message(STATUS "FetchContent_Populate ${torch_fetch_git_repository} ${torch_fetch_git_tag}")
FetchContent_Populate(torch_fetch)
message(STATUS "FetchContent_Populate ${torch_fetch_git_repository} ${torch_fetch_git_tag} - done")

# We can enable and disable functionality with the following options.
if(NOT DEFINED ENV{USE_CUDA})
  # If the user hasn't indicated either way, which would be the case
  # on GitHub, then we will not use CUDA.
  set(ENV{USE_CUDA} "0")
endif()
if($ENV{USE_CUDA})
  set(torch_fetch_additional_libraries "torch_cuda" "c10_cuda")
endif()
if(NOT DEFINED ENV{USE_CUDNN})
  # If the user hasn't indicated either way, which would be the case
  # on GitHub, then we will not use CUDNN.
  set(ENV{USE_CUDNN} "0")
endif()

# Use torch's setup.py to build and locally install it.
if(NOT DEFINED torch_fetch_install_directory)
  set(torch_fetch_install_directory "${torch_fetch_SOURCE_DIR}/../torch_fetch-install")
  set(torch_fetch_build_directory "${torch_fetch_SOURCE_DIR}/build")
  message(STATUS "Executing: cd ${torch_fetch_SOURCE_DIR}; ${Python_EXECUTABLE} setup.py install --prefix=${torch_fetch_install_directory}")
  add_custom_target(
    temp ALL
    ${Python_EXECUTABLE} setup.py install --prefix=${torch_fetch_install_directory}
    WORKING_DIRECTORY ${torch_fetch_SOURCE_DIR}
    )
  message(STATUS "Executing: cd ${torch_fetch_SOURCE_DIR}; ${Python_EXECUTABLE} setup.py install --prefix=${torch_fetch_install_directory} - done")
endif()

include_directories(
  "${torch_fetch_install_directory}/torch/include/torch/csrc/api/include"
  "${torch_fetch_install_directory}/torch/include"
  "${torch_fetch_install_directory}/caffe2/aten/src"
  "${torch_fetch_install_directory}/lib/python3.8/site-packages/torch/include"
  "${torch_fetch_install_directory}/lib/python3.8/site-packages/torch/include/torch/csrc/api/include"
  "${torch_fetch_build_directory}/aten/src"
  "${torch_fetch_build_directory}/nccl/include"
  "${torch_fetch_SOURCE_DIR}/aten/src"
  )

add_subdirectory(${torch_fetch_SOURCE_DIR} ${torch_fetch_build_directory})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-as-needed")
list(APPEND PyTorch_LIBRARIES "${torch_fetch_additional_libraries}")
list(APPEND PyTorch-Test_LIBRARIES "${torch_fetch_additional_libraries}")

if(NOT ITK_SOURCE_DIR)
  find_package(ITK REQUIRED)
  list(APPEND CMAKE_MODULE_PATH ${ITK_CMAKE_DIR})
  include(ITKModuleExternal)
else()
  set(ITK_DIR ${CMAKE_BINARY_DIR})
  itk_module_impl()
endif()

if(1)
  get_cmake_property(_variableNames VARIABLES)
  list (SORT _variableNames)
  foreach (_variableName ${_variableNames})
      message(STATUS "${_variableName}=${${_variableName}}")
  endforeach()
endif()

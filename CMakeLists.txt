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

if(NOT DEFINED Torch_DIR)
  if(USE_CUDA)
    if(APPLE)
      message(STATUS "FetchContent_Populate(torch_fetch) will be for APPLE without CUDA 10.1")
      set(torch_fetch_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.5.1.zip")
      set(torch_fetch_HASH "MD5=77ca8e871c1200f1e5864ae1d9251a48")
    elseif(WIN32)
      message(STATUS "FetchContent_Populate(torch_fetch) will be for WIN32 with CUDA 10.1")
      set(torch_fetch_URL "https://download.pytorch.org/libtorch/cu101/libtorch-win-shared-with-deps-1.5.1%2Bcu101.zip")
      set(torch_fetch_HASH "MD5=90eb7739fc3b2f4572a93ee6da1e2139")
    else()
      if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0")
        message(STATUS "FetchContent_Populate(torch_fetch) will be for Linux (not C++11 ABI) with CUDA 10.1")
        set(torch_fetch_URL "https://download.pytorch.org/libtorch/cu101/libtorch-shared-with-deps-1.5.1%2Bcu101.zip")
        set(torch_fetch_HASH "MD5=9c9fbdd9dfcc1abd1049e10342f3ba72")
      else()
        message(STATUS "FetchContent_Populate(torch_fetch) will be for Linux (C++11 ABI) with CUDA 10.1")
        set(torch_fetch_URL "https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcu101.zip")
        set(torch_fetch_HASH "MD5=b18d6f2fdfe730c1fca3edd5ed2f3c8f")
      endif()
    endif()
  else()
    if(APPLE)
      message(STATUS "FetchContent_Populate(torch_fetch) will be for APPLE without CUDA")
      set(torch_fetch_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.5.1.zip")
      set(torch_fetch_HASH "MD5=77ca8e871c1200f1e5864ae1d9251a48")
    elseif(WIN32)
      message(STATUS "FetchContent_Populate(torch_fetch) will be for WIN32 without CUDA")
      set(torch_fetch_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.5.1%2Bcpu.zip")
      set(torch_fetch_HASH "MD5=ff07328e322040662612c4a0f7706d3c")
    else()
      if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "6.0")
        message(STATUS "FetchContent_Populate(torch_fetch) will be for Linux (not C++11 ABI) without CUDA")
        set(torch_fetch_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.5.1%2Bcpu.zip")
        set(torch_fetch_HASH "MD5=712ffae2979e8bc6a9cc92d321c2befe")
      else()
        message(STATUS "FetchContent_Populate(torch_fetch) will be for Linux (C++11 ABI) without CUDA")
        set(torch_fetch_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcpu.zip")
        set(torch_fetch_HASH "MD5=088e4f38ae7a0fa2e6ee8fea254866cf")
      endif()
    endif()
  endif()

  include(FetchContent)
  FetchContent_Declare(
    torch_fetch
    URL           "${torch_fetch_URL}"
    URL_HASH      "${torch_fetch_HASH}"
    )
  message(STATUS "FetchContent_Populate ${torch_fetch_URL}")
  FetchContent_Populate(torch_fetch)
  message(STATUS "FetchContent_Populate ${torch_fetch_URL} - done")
  message(STATUS "torch_fetch_SOURCE_DIR=${torch_fetch_SOURCE_DIR}")
  message(STATUS "torch_fetch_BINARY_DIR=${torch_fetch_BINARY_DIR}")

  set(Torch_DIR "${torch_fetch_SOURCE_DIR}/share/cmake/Torch")
endif()
message(STATUS "Torch_DIR=${Torch_DIR}")
find_package(Torch REQUIRED CONFIG)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(PyTorch_LIBRARIES "${TORCH_LIBRARIES}")
include_directories(
  ${torch_fetch_SOURCE_DIR}/include/torch/csrc/api/include
  ${torch_fetch_SOURCE_DIR}/include
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

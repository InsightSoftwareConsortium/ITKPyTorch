cmake_minimum_required(VERSION 3.14)
project(PyTorch)
## libtorch requires C++-14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE)
  set(torch_fetch_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-1.5.1.zip")
  set(torch_fetch_HASH "MD5=77ca8e871c1200f1e5864ae1d9251a48")
elseif(WIN32)
  set(torch_fetch_URL "https://download.pytorch.org/libtorch/cu101/libtorch-win-shared-with-deps-1.5.1%2Bcu101.zip")
  set(torch_fetch_HASH "MD5=90eb7739fc3b2f4572a93ee6da1e2139")
else()
  set(torch_fetch_URL "https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.5.1%2Bcu101.zip")
  set(torch_fetch_HASH "MD5=b18d6f2fdfe730c1fca3edd5ed2f3c8f")
endif()

include(FetchContent)
FetchContent_Declare(
  torch_fetch
  URL           "${torch_fetch_URL}"
  URL_HASH      "${torch_fetch_HASH}"
)
FetchContent_Populate(torch_fetch)

# get_cmake_property(_variableNames VARIABLES)
# list (SORT _variableNames)
# foreach (_variableName ${_variableNames})
#     message(STATUS "${_variableName}=${${_variableName}}")
# endforeach()

set(Torch_DIR "${torch_fetch_SOURCE_DIR}/share/cmake/Torch")
find_package(Torch REQUIRED CONFIG)

# set(_Torch_lib Torch::Torch)
# set(Torch_DIR_INSTALL ${Torch_DIR})
# set(Torch_DIR_BUILD ${Torch_DIR})
# # It is only needed to EXPORT_CODE_BUILD when using external Torch
# set(${PROJECT_NAME}_EXPORT_CODE_BUILD "${${PROJECT_NAME}_EXPORT_CODE_BUILD}
# if(NOT ITK_BINARY_DIR)
#   set(Torch_DIR \"${Torch_DIR_BUILD}\")
#   find_package(Torch REQUIRED CONFIG)
# endif()
# ")
#
# include_directories(${TORCH_INCLUDE_DIRS})
# # itk_module_target(${_Torch_lib} NO_INSTALL)
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

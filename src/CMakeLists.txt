set(PyTorch_SRCS
  itkMinimalStandardRandomVariateGenerator.cxx
  )

itk_module_add_library(PyTorch ${PyTorch_SRCS})

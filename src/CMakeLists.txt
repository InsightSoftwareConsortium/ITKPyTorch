set(PyTorch_SRCS
  itkTorchDataManager.cxx
  )

itk_module_add_library(PyTorch ${PyTorch_SRCS})

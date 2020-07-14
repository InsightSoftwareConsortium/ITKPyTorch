set(PyTorch_SRCS
  itkPyTorchDataManager.cxx
  )

itk_module_add_library(PyTorch ${PyTorch_SRCS})

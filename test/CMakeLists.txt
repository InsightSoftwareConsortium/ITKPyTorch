itk_module_test()

set(PyTorchTests
  itkTorchImageTest.cxx
  itkTorchImageToImageFilterTest.cxx
  )

CreateTestDriver(PyTorch "${PyTorch-Test_LIBRARIES}" "${PyTorchTests}")

itk_add_test(NAME itkTorchImageTest
  COMMAND PyTorchTestDriver
  itkTorchImageTest
    ${ITK_TEST_OUTPUT_DIR}/itkTorchImageTestOutput.mha
  )

itk_add_test(NAME itkTorchImageToImageFilterTest
  COMMAND PyTorchTestDriver
  itkTorchImageToImageFilterTest
    ${ITK_TEST_OUTPUT_DIR}/itkTorchImageTestOutput.mha
  )

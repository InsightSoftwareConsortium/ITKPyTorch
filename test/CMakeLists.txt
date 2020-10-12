itk_module_test()

set(PyTorchTests
  itkTorchImageTest.cxx
  itkTorchImageToTorchImageFilterTest.cxx
  itkImageToTorchImageFilterTest.cxx
  itkTorchImageToImageFilterTest.cxx
  )

CreateTestDriver(PyTorch "${PyTorch-Test_LIBRARIES}" "${PyTorchTests}")

itk_add_test(NAME itkTorchImageTest
  COMMAND PyTorchTestDriver
  itkTorchImageTest
    ${ITK_TEST_OUTPUT_DIR}/itkTorchImageTestOutput.mha
  )

itk_add_test(NAME itkTorchImageToTorchImageFilterTest
  COMMAND PyTorchTestDriver
  itkTorchImageToTorchImageFilterTest
    ${ITK_TEST_OUTPUT_DIR}/itkTorchImageTestOutput.mha
  )

itk_add_test(NAME itkImageToTorchImageFilterTest
  COMMAND PyTorchTestDriver
    --compare
    ${ITK_TEST_OUTPUT_DIR}/itkImageToTorchImageFilterTestOutput.png
    DATA{Baseline/itkImageToTorchImageFilterTestOutput.png}
  itkImageToTorchImageFilterTest
    ${ITK_TEST_OUTPUT_DIR}/itkImageToTorchImageFilterTestOutput.png
  )

itk_add_test(NAME itkTorchImageToImageFilterTest
  COMMAND PyTorchTestDriver
  itkTorchImageToImageFilterTest
    ${ITK_TEST_OUTPUT_DIR}/itkTorchImageTestOutput.mha
  )

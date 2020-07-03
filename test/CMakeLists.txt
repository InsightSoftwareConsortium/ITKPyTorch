itk_module_test()

set(PyTorchTests
  itkMinimalStandardRandomVariateGeneratorTest.cxx
  itkPyTorchImageToImageFilterTest.cxx
  )

CreateTestDriver(PyTorch "${PyTorch-Test_LIBRARIES}" "${PyTorchTests}")

itk_add_test(NAME itkMinimalStandardRandomVariateGeneratorTest
  COMMAND PyTorchTestDriver itkMinimalStandardRandomVariateGeneratorTest
  )

itk_add_test(NAME itkPyTorchImageToImageFilterTest
  COMMAND PyTorchTestDriver
    --compare
    DATA{Baseline/itkPyTorchImageToImageFilterTestOutput.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkPyTorchImageToImageFilterTestOutput.mha
  itkPyTorchImageToImageFilterTest
    ${ITK_TEST_OUTPUT_DIR}/itkPyTorchImageToImageFilterTestOutput.mha
  )

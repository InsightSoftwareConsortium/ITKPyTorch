itk_module_test()

set(PyTorchTests
  itkPyTorchImageTest.cxx
  )

CreateTestDriver(PyTorch "${PyTorch-Test_LIBRARIES}" "${PyTorchTests}")

itk_add_test(NAME itkPyTorchImageTest
  COMMAND PyTorchTestDriver
    --compare
    DATA{Baseline/itkPyTorchImageTestOutput.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkPyTorchImageTestOutput.mha
  itkPyTorchImageTest
    ${ITK_TEST_OUTPUT_DIR}/itkPyTorchImageTestOutput.mha
  )

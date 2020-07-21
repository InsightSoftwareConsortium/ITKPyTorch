itk_module_test()

set(PyTorchTests
  itkTorchImageTest.cxx
  )

CreateTestDriver(PyTorch "${PyTorch-Test_LIBRARIES}" "${PyTorchTests}")

itk_add_test(NAME itkTorchImageTest
  COMMAND PyTorchTestDriver
    --compare
    DATA{Baseline/itkTorchImageTestOutput.mha}
    ${ITK_TEST_OUTPUT_DIR}/itkTorchImageTestOutput.mha
  itkTorchImageTest
    ${ITK_TEST_OUTPUT_DIR}/itkTorchImageTestOutput.mha
  )

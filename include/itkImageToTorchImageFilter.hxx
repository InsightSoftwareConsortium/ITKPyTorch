/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkImageToTorchImageFilter_hxx
#define itkImageToTorchImageFilter_hxx

#include <torch/torch.h>

#include "itkImageToTorchImageFilter.h"

namespace itk
{

template< typename TInputImage >
void
ImageToTorchImageFilter< TInputImage >
::PrintSelf( std::ostream &os, Indent indent ) ITKv5_CONST
{
  Superclass::PrintSelf( os, indent );
}


template< typename TInputImage >
void
ImageToTorchImageFilter< TInputImage >
::VerifyPreconditions() ITKv5_CONST
{
  // Call the superclass's implementation of this method
  Superclass::VerifyPreconditions();

  // Get pointers to the input image.
  typename Self::InputImageType *inputImage = const_cast< typename Self::InputImageType * >( this->GetInput( 0 ) );

  if( inputImage != nullptr )
    {
    inputImage->SetRequestedRegionToLargestPossibleRegion();
    }
}


template< typename TInputImage >
void
ImageToTorchImageFilter< TInputImage >
::GenerateData()
{
  this->AllocateOutputs();

  // Get input and output and check that their sizes match
  typename Self::InputImageType const * const inputImage = this->GetInput();
  const typename Self::InputImageSizeType inputBufferedRegionSize = inputImage->GetBufferedRegion().GetSize();
  typename Self::InputImagePixelType const * const inputBufferPointer = inputImage->GetBufferPointer();

  typename Self::OutputTorchImageType * const outputTorchImage = this->GetOutput();
  const typename Self::OutputTorchImageSizeType outputBufferedRegionSize = outputTorchImage->GetBufferedRegion().GetSize();

  itkAssertOrThrowMacro( inputBufferedRegionSize == outputBufferedRegionSize, "ImageToTorchImageFilter: input and output images' buffered regions must have the same size in each dimension" );

  // Do the memory copy of the whole image.
  const std::vector< int64_t > torchSize = outputTorchImage->ComputeTorchSize();
  const c10::TensorOptions commonTensorOptions = torch::dtype( Self::OutputTorchImageType::TorchValueType ).layout( torch::kStrided );
  const c10::TensorOptions inputImageTensorOptions = commonTensorOptions.requires_grad( false ).device( torch::kCPU );
  const c10::TensorOptions outputTorchImageTensorOptions =
    outputTorchImage->m_DeviceType == Self::OutputTorchImageType::itkCPU ?
    commonTensorOptions.device( torch::kCPU ) :
    commonTensorOptions.device( torch::kCUDA, outputTorchImage->m_CudaDeviceNumber );

  const torch::Tensor inputTensor =
    torch::from_blob( const_cast< typename Self::InputImagePixelType * >( inputBufferPointer ), torchSize, inputImageTensorOptions );

  outputTorchImage->m_Tensor = inputTensor.to( outputTorchImageTensorOptions );
}


} // end namespace itk

#endif // itkImageToTorchImageFilter_hxx

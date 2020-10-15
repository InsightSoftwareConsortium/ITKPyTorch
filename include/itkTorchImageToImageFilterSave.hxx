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
#ifndef itkTorchImageToImageFilter_hxx
#define itkTorchImageToImageFilter_hxx

#include <torch/torch.h>

#include "itkTorchImageToImageFilter.h"

namespace itk
{

template< typename TInputTorchImage >
void
TorchImageToImageFilter< TInputTorchImage >
::PrintSelf( std::ostream &os, Indent indent ) ITKv5_CONST
{
  Superclass::PrintSelf( os, indent );
}


template< typename TInputTorchImage >
void
TorchImageToImageFilter< TInputTorchImage >
::VerifyPreconditions() ITKv5_CONST
{
  // Call the superclass's implementation of this method
  Superclass::VerifyPreconditions();

  // Get pointers to the input image.
  typename Self::InputTorchImageType * const inputTorchImage = const_cast< typename Self::InputTorchImageType * >( this->GetInput( 0 ) );

  if( inputTorchImage != nullptr )
    {
    inputTorchImage->SetRequestedRegionToLargestPossibleRegion();
    }
}


template< typename TInputTorchImage >
void
TorchImageToImageFilter< TInputTorchImage >
::GenerateData()
{
  this->AllocateOutputs();

  typename Self::InputTorchImageType const * const inputTorchImage = this->GetInput();
  const typename Self::InputTorchImageSizeType inputBufferedRegionSize = inputTorchImage->GetBufferedRegion().GetSize();

  typename Self::OutputImageType * const outputImage = this->GetOutput();
  const typename Self::OutputImageSizeType outputBufferedRegionSize = outputImage->GetBufferedRegion().GetSize();
  typename Self::OutputImagePixelType * const outputBufferPointer = outputImage->GetBufferPointer();

  itkAssertOrThrowMacro( inputBufferedRegionSize == outputBufferedRegionSize, "TorchImageToImageFilter: input and output images' buffered regions must have the same size in each dimension" );

  // Do the memory copy of the whole image.

  const typename Self::OutputImageSizeType::SizeValueType numberOfPixels =
    std::accumulate( outputBufferedRegionSize.begin(), outputBufferedRegionSize.end(),
      static_cast< typename Self::OutputImageSizeType::SizeValueType >( 1 ),
      std::multiplies< typename Self::OutputImageSizeType::SizeValueType >() );

  const torch::Tensor cpuTensor = inputTorchImage->m_Tensor.to( torch::device( torch::kCPU ) );

  std::memcpy( outputBufferPointer, cpuTensor.data_ptr(), numberOfPixels * sizeof( OutputImagePixelType ) );
}


} // end namespace itk

#endif // itkTorchImageToImageFilter_hxx

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
#ifndef itkTorchImage_hxx
#define itkTorchImage_hxx

#include "itkTorchImage.h"
#include "itkTorchTensorAccessorHelper.h"

namespace itk
{

template< typename TPixel, unsigned int VImageDimension >
constexpr unsigned int
TorchImage< TPixel, VImageDimension >
::ImageDimension;

template< typename TPixel, unsigned int VImageDimension >
constexpr at::ScalarType
TorchImage< TPixel, VImageDimension >
::TorchValueType;

template< typename TPixel, unsigned int VImageDimension >
constexpr unsigned int
TorchImage< TPixel, VImageDimension >
::PixelDimension;

template< typename TPixel, unsigned int VImageDimension >
constexpr unsigned int
TorchImage< TPixel, VImageDimension >
::TorchDimension;

template< typename TPixel, unsigned int VImageDimension >
bool
TorchImage< TPixel, VImageDimension >
::ChangeDevice( DeviceType deviceType )
{
  switch( deviceType )
    {
    case itkCUDA:
      return ChangeDevice( deviceType, 0 );
      break;
    case itkCPU:
      if( m_DeviceType == torch::kCPU )
        {
        return true;            // no change
        }
      // Change from GPU to CPU
      m_Tensor = m_Tensor.to( torch::kCPU );
      m_DeviceType = deviceType;
    }

  return true;
}

template< typename TPixel, unsigned int VImageDimension >
bool
TorchImage< TPixel, VImageDimension >
::ChangeDevice( DeviceType deviceType, int64_t cudaDeviceNumber )
{
  switch( deviceType )
    {
    case itkCUDA:
      if( m_DeviceType == torch::kCUDA && m_CudaDeviceNumber == cudaDeviceNumber )
        {
        return true;            // no change
        }
      m_Tensor = m_Tensor.to( torch::kCUDA, cudaDeviceNumber );
      m_DeviceType = deviceType;
      m_CudaDeviceNumber = cudaDeviceNumber;
      break;
    case itkCPU:
      return false;     // cudaDeviceNumber not supported for torch::kCPU.
      break;
    }

  return true;
}

template< typename TPixel, unsigned int VImageDimension >
std::vector< int64_t >
TorchImage< TPixel, VImageDimension >
::ComputeTorchSize() const
{
  // Get index components of the pixel, reversing their order so that
  // the first one varies the slowest in the buffer.
  const SizeType &bufferSize = Superclass::GetBufferedRegion().GetSize();
  std::vector< int64_t > torchSize( Self::TorchDimension );
  for( SizeValueType i = 0; i < Self::ImageDimension; ++i )
    {
    torchSize.push_back( bufferSize[Self::ImageDimension-1-i] );
    }
  // Append 0 or more dimension sizes representing non-scalar pixels.
  Self::TorchPixelHelper::AppendSizes( torchSize );
  return torchSize;
}

template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::Allocate( bool initializePixels )
{
  Superclass::Allocate( initializePixels );
  // Non-scalar pixel types are represented as additional dimensions in the torch image.
  const std::vector< int64_t > torchSize = this->ComputeTorchSize();

  // Set up Tensor options
  c10::TensorOptions tensorOptions = torch::dtype( Self::TorchValueType ).layout( torch::kStrided ).requires_grad( false );
  switch( m_DeviceType )
    {
    case itkCUDA:
      tensorOptions = tensorOptions.device( torch::kCUDA, m_CudaDeviceNumber );
      break;
    case itkCPU:
      tensorOptions = tensorOptions.device( torch::kCPU );
      break;
    }

  if( initializePixels )
    {
    m_Tensor = torch::zeros( torchSize, tensorOptions );
    }
  else
    {
    m_Tensor = torch::empty( torchSize, tensorOptions );
    }

  // m_Allocated = true;
}

template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::Initialize()
{
  m_Tensor = torch::Tensor();
  Superclass::Initialize();
  // m_Allocated = false;
}

template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::FillBuffer( const TPixel &value )
{
  torch::TensorAccessor< DeepScalarType, TorchDimension > accessor
    = m_Tensor.accessor< DeepScalarType, TorchDimension >();
  const SizeType &bufferSize = Superclass::GetBufferedRegion().GetSize();
  FillBufferPart< TorchDimension, ImageDimension >( accessor, bufferSize, value );
}

template< typename TPixel, unsigned int VImageDimension >
template< int VCurrentAccessorLevel, int VNumberOfSteps >
void
TorchImage< TPixel, VImageDimension >
::FillBufferPart( torch::TensorAccessor< DeepScalarType, VCurrentAccessorLevel > &accessor,
  const SizeType &bufferSize,
  const TPixel &value )
{
  for( SizeValueType i = 0; i < bufferSize[VNumberOfSteps-1]; ++i )
    {
    FillBufferPart< TorchDimension-1, ImageDimension-1 >(accessor[i], bufferSize, value);
    }
}

template< typename TPixel, unsigned int VImageDimension >
template< int VCurrentAccessorLevel >
void
TorchImage< TPixel, VImageDimension >
::FillBufferPart< VCurrentAccessorLevel, 0 >( torch::TensorAccessor< DeepScalarType, VCurrentAccessorLevel > &accessor,
  const SizeType &bufferSize,
  const TPixel &value )
{
  accessor = value;
}

template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::SetPixel( const IndexType & index, const PixelType & value )
{
  torch::TensorAccessor< DeepScalarType, TorchDimension > accessor
    = m_Tensor.accessor< DeepScalarType, TorchDimension >();
  torch::TensorAccessor< DeepScalarType, ImageDimension > pixelAccessor
    = TorchTensorAccessorHelper< DeepScalarType, IndexType, TorchDimension, ImageDimension >::FindPixel( accessor, index );
  TorchPixelHelper {pixelAccessor} = value;
}

template< typename TPixel, unsigned int VImageDimension >
typename TorchImage< TPixel, VImageDimension >::TorchPixelHelper
TorchImage< TPixel, VImageDimension >
::GetPixel( const IndexType & index )
{
  torch::TensorAccessor< DeepScalarType, TorchDimension > accessor
    = m_Tensor.accessor< DeepScalarType, TorchDimension >();
  torch::TensorAccessor< DeepScalarType, ImageDimension > pixelAccessor
    = TorchTensorAccessorHelper< DeepScalarType, IndexType, TorchDimension, ImageDimension >::FindPixel( accessor, index );
  return TorchPixelHelper {pixelAccessor};
}

template< typename TPixel, unsigned int VImageDimension >
const typename TorchImage< TPixel, VImageDimension >::TorchPixelHelper
TorchImage< TPixel, VImageDimension >
::GetPixel( const IndexType & index ) const
{
  torch::TensorAccessor< DeepScalarType, TorchDimension > accessor
    = m_Tensor.accessor< DeepScalarType, TorchDimension >();
  torch::TensorAccessor< DeepScalarType, ImageDimension > pixelAccessor
    = TorchTensorAccessorHelper< DeepScalarType, IndexType, TorchDimension, ImageDimension >::FindPixel( accessor, index );
  return TorchPixelHelper {pixelAccessor};
}

/** The pointer might be to GPU memory and, if so, could not be
 * dereferenced */
template< typename TPixel, unsigned int VImageDimension >
TPixel *
TorchImage< TPixel, VImageDimension >
::GetBufferPointer()
{
  return reinterpret_cast< TPixel * >( m_Tensor.data_ptr< DeepScalarType >() );
}

/** The pointer might be to GPU memory and, if so, could not be
 * dereferenced */
template< typename TPixel, unsigned int VImageDimension >
const TPixel *
TorchImage< TPixel, VImageDimension >
::GetBufferPointer() const
{
  return reinterpret_cast< const TPixel * >( m_Tensor.data_ptr< DeepScalarType >() );
}

template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::Graft( const Self * data )
{
  Superclass::Graft( data );
  m_DeviceType = data->m_DeviceType;
  m_CudaDeviceNumber = data->m_CudaDeviceNumber;
  // m_Allocated = data->m_Allocated;
  m_Tensor = torch::from_blob( data->m_Tensor.data_ptr(), data->m_Tensor.sizes(), data->m_Tensor.options() );
  // m_Grafted = data->m_Grafted;
}

template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::Graft( const DataObject * data )
{
  if( data )
    {
    // Attempt to cast data to an Image
    const auto * const imgData = dynamic_cast< const Self * >( data );

    if ( imgData != nullptr )
      {
      this->Graft( imgData );
      }
    else
      {
      // pointer could not be cast back down
      itkExceptionMacro( << "itk::TorchImage::Graft() cannot cast " << typeid( data ).name() << " to "
        << typeid( const Self * ).name() );
      }
    }
}

template< typename TPixel, unsigned int VImageDimension >
TorchImage< TPixel, VImageDimension >
::TorchImage()
{
  m_DeviceType = itkCUDA;
  m_CudaDeviceNumber = 0;
  // m_Allocated = false;
  m_Tensor = torch::Tensor();
  // m_Grafted = false;
}

template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf(os, indent);
  os
    << indent << "m_DeviceType: " << m_DeviceType << std::endl
    << indent << "m_CudaDeviceNumber: " << m_CudaDeviceNumber << std::endl
    // << indent << "m_Allocated: " << m_Allocated << std::endl
    //!!! << indent << "m_Tensor: " << m_Tensor << std::endl
    // << indent << "m_Grafted: " << m_Grafted << std::endl
    ;
}

template <typename TPixel, unsigned int VImageDimension>
void
TorchImage<TPixel, VImageDimension>
::ComputeIndexToPhysicalPointMatrices()
{
  this->Superclass::ComputeIndexToPhysicalPointMatrices();
}

// Implement ComputeOffsetTable?!!!

// Do we need this->Modified calls()?!!!

} // end namespace itk

#endif

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

namespace itk
{

template< typename TPixel, unsigned int VImageDimension >
constexpr unsigned int
TorchImage< TPixel, VImageDimension >
::ImageDimension;

template< typename TPixel, unsigned int VImageDimension >
constexpr torch::ScalarType
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
::SetDevice( typename Self::DeviceType deviceType )
{
  switch( deviceType )
    {
    case Self::itkCUDA:
      return this->SetDevice( deviceType, 0 );
      break;
    case itkCPU:
      if( m_DeviceType == Self::itkCPU )
        {
        return true;            // no change
        }
      // Change from GPU to CPU
      if( m_Allocated )
        {
        m_Tensor = m_Tensor.to( torch::kCPU );
        }
      m_DeviceType = deviceType;
    }

  return true;
}

template< typename TPixel, unsigned int VImageDimension >
bool
TorchImage< TPixel, VImageDimension >
::SetDevice( typename Self::DeviceType deviceType, uint64_t cudaDeviceNumber )
{
  switch( deviceType )
    {
    case Self::itkCUDA:
      if( m_DeviceType == Self::itkCUDA && m_CudaDeviceNumber == cudaDeviceNumber )
        {
        return true;            // no change
        }
      if( !( torch::cuda::is_available() && cudaDeviceNumber< torch::cuda::device_count() ) )
        {
        return false;
        }
      if( m_Allocated )
        {
        m_Tensor = m_Tensor.to( torch::kCUDA, cudaDeviceNumber );
        }
      m_DeviceType = deviceType;
      m_CudaDeviceNumber = cudaDeviceNumber;
      return true;
      break;
    case itkCPU:
      return false;     // cudaDeviceNumber not supported for itkCPU.
      break;
    }

  return true;
}

template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::GetDevice( typename Self::DeviceType &deviceType, uint64_t &cudaDeviceNumber )
{
  deviceType = m_DeviceType;
  cudaDeviceNumber = m_CudaDeviceNumber;
}

template< typename TPixel, unsigned int VImageDimension >
std::vector< int64_t >
TorchImage< TPixel, VImageDimension >
::ComputeTorchSize() const
{
  // Get index components of the pixel, reversing their order so that
  // the first one varies the slowest in the buffer.
  const typename Self::SizeType &bufferSize = this->GetBufferedRegion().GetSize();
  std::vector< int64_t > torchSize;
  for( typename Self::SizeValueType i = 0; i < Self::ImageDimension; ++i )
    {
    torchSize.push_back( bufferSize[Self::ImageDimension-1-i] );
    }
  // Append 0 or more dimension sizes representing non-scalar pixels.
  Self::TorchImagePixelHelper::AppendSizes( torchSize );
  return torchSize;
}

template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::Allocate( typename Self::TensorInitializer tensorInitializer )
{
  // itkImage does not call Superclass::Allocate.  Should we?
  // Superclass::Allocate( initializePixels );

  // Non-scalar pixel types are represented as additional dimensions in the torch image.
  const std::vector< int64_t > torchSize = this->ComputeTorchSize();

  // Set up Tensor options
  c10::TensorOptions tensorOptions = torch::dtype( Self::TorchValueType ).layout( torch::kStrided ).requires_grad( false );
  switch( m_DeviceType )
    {
    case Self::itkCUDA:
      tensorOptions = tensorOptions.device( torch::kCUDA, m_CudaDeviceNumber );
      break;
    case Self::itkCPU:
      tensorOptions = tensorOptions.device( torch::kCPU );
      break;
    }

  switch( tensorInitializer )
    {
    case Self::itkEmpty:
      m_Tensor = torch::empty( torchSize, tensorOptions );
      break;
    case Self::itkZeros:
      m_Tensor = torch::zeros( torchSize, tensorOptions );
      break;
    case Self::itkOnes:
      m_Tensor = torch::ones( torchSize, tensorOptions );
      break;
    case Self::itkRand:
      m_Tensor = torch::rand( torchSize, tensorOptions );
      break;
    case Self::itkRandn:
      m_Tensor = torch::randn( torchSize, tensorOptions );
      break;
    }
  m_Allocated = true;
}

template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::Initialize()
{
  //
  // We don't modify ourselves because the "ReleaseData" methods depend upon
  // no modification when initialized.
  //

  // Call the superclass which should initialize the BufferedRegion ivar.
  Superclass::Initialize();

  // Replace the handle to the buffer. This is the safest thing to do,
  // since the same container can be shared by multiple images (e.g.
  // Grafted outputs and in place filters).
  m_Tensor = torch::Tensor();
  m_Allocated = false;
}

template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::FillBufferPart( int CurrentDimensions, const typename Self::SizeType &BufferSize, std::vector< torch::indexing::TensorIndex > &TorchIndex, const typename Self::PixelType &value )
{
  if( CurrentDimensions == 0 )
    {
    Self::TorchImagePixelHelper {m_Tensor, TorchIndex} = value;
    }
  else
    {
    // Slowest varying dimension in BufferSize is last
    for( typename Self::SizeValueType i = 0; i < BufferSize[CurrentDimensions-1]; ++i )
      {
      TorchIndex.push_back( static_cast< int64_t >( i ) );
      this->FillBufferPart( CurrentDimensions-1, BufferSize, TorchIndex, value );
      TorchIndex.pop_back();
      }
    }
}

template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::SetPixel( const typename Self::IndexType &index, const typename Self::PixelType &value )
{
  this->GetPixel( index ) = value;
}

template< typename TPixel, unsigned int VImageDimension >
typename TorchImage< TPixel, VImageDimension >::TorchImagePixelHelper
TorchImage< TPixel, VImageDimension >
::GetPixel( const typename Self::IndexType &index )
{
  std::vector< torch::indexing::TensorIndex > TorchIndex;
  for( unsigned int i = 0; i < Self::ImageDimension; ++i )
    {
    TorchIndex.push_back( static_cast< int64_t >( index[Self::ImageDimension-1-i] ) );
    }
  return Self::TorchImagePixelHelper { m_Tensor, TorchIndex };
}

template< typename TPixel, unsigned int VImageDimension >
const typename TorchImage< TPixel, VImageDimension >::TorchImagePixelHelper
TorchImage< TPixel, VImageDimension >
::GetPixel( const typename Self::IndexType &index ) const
{
  std::vector< torch::indexing::TensorIndex > TorchIndex;
  for( unsigned int i = 0; i < Self::ImageDimension; ++i )
    {
    TorchIndex.push_back( static_cast< int64_t >( index[Self::ImageDimension-1-i] ) );
    }
  return Self::TorchImagePixelHelper { m_Tensor, TorchIndex };
}

/** The pointer might be to GPU memory and, if so, cannot be directly
 * dereferenced */
template< typename TPixel, unsigned int VImageDimension >
typename TorchImage< TPixel, VImageDimension >::PixelType *
TorchImage< TPixel, VImageDimension >
::GetBufferPointer()
{
  return reinterpret_cast< typename Self::PixelType * >( m_Tensor.data_ptr< DeepScalarType >() );
}

/** The pointer might be to GPU memory and, if so, cannot be directly
 * dereferenced */
template< typename TPixel, unsigned int VImageDimension >
const typename TorchImage< TPixel, VImageDimension >::PixelType *
TorchImage< TPixel, VImageDimension >
::GetBufferPointer() const
{
  return reinterpret_cast< const typename Self::PixelType * >( m_Tensor.data_ptr< DeepScalarType >() );
}

template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::Graft( const Self *data )
{
  Superclass::Graft( data );
  m_DeviceType = data->m_DeviceType;
  m_CudaDeviceNumber = data->m_CudaDeviceNumber;
  m_Allocated = data->m_Allocated;
  if( m_Allocated )
    {
    m_Tensor = torch::from_blob( data->m_Tensor.data_ptr(), data->m_Tensor.sizes(), data->m_Tensor.options() );
    }
}

template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::Graft( const DataObject *data )
{
  if( data )
    {
    // Attempt to cast data to an Image
    const auto * const imgData = dynamic_cast< const Self * >( data );

    if( imgData != nullptr )
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
  m_DeviceType = Self::itkCPU;
  m_CudaDeviceNumber = 0;
  m_Allocated = false;
  m_Tensor = torch::Tensor();
  // SetDevice sets this TorchImage to Self::itkCUDA only if the GPU exists.
  this->SetDevice( Self::itkCUDA, 0 );
}

template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::PrintSelf( std::ostream &os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );
  os
    << indent << "m_DeviceType: " << m_DeviceType << std::endl
    << indent << "m_Allocated: " << m_Allocated << std::endl
    << indent << "m_CudaDeviceNumber: " << m_CudaDeviceNumber << std::endl
    // << indent << "m_Tensor: " << m_Tensor << std::endl
    ;
}

} // end namespace itk

#endif

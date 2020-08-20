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
      return this->ChangeDevice( deviceType, 0 );
      break;
    case itkCPU:
      if( m_DeviceType == itkCPU )
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
::ChangeDevice( DeviceType deviceType, int64_t cudaDeviceNumber )
{
  switch( deviceType )
    {
    case itkCUDA:
      if( m_DeviceType == itkCUDA && m_CudaDeviceNumber == cudaDeviceNumber )
        {
        return true;            // no change
        }
      if( m_Allocated )
        {
        m_Tensor = m_Tensor.to( torch::kCUDA, cudaDeviceNumber );
        }
      m_DeviceType = deviceType;
      m_CudaDeviceNumber = cudaDeviceNumber;
      break;
    case itkCPU:
      return false;     // cudaDeviceNumber not supported for itkCPU.
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
  const SizeType &bufferSize = Self::GetBufferedRegion().GetSize();
  std::vector< int64_t > torchSize;
  for( SizeValueType i = 0; i < Self::ImageDimension; ++i )
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
::Allocate( bool initializePixels )
{
  // itkImage does not call Superclass::Allocate.  Should we?
  // Superclass::Allocate( initializePixels );

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
::FillBuffer( const PixelType &value )
{
  if /* constexpr */ ( PixelDimension == 0 )
    {
    m_Tensor.fill_( value );
    }
  else
    {
    const SizeType &bufferSize = Self::GetBufferedRegion().GetSize();
    std::vector< int64_t > TorchIndex;
    FillBufferPart( Self::ImageDimension, bufferSize, TorchIndex, value );
    }
}

template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::FillBufferPart( int CurrentDimensions, const SizeType &BufferSize, std::vector< int64_t > &TorchIndex, const PixelType &value )
{
  if( CurrentDimensions == 0 )
    {
    TorchImagePixelHelper {m_Tensor, TorchIndex} = value;
    }
  else
    {
    // Slowest varying dimension in BufferSize is last
    for( SizeValueType i = 0; i < BufferSize[CurrentDimensions-1]; ++i )
      {
      TorchIndex.push_back( i );
      FillBufferPart( CurrentDimensions-1, BufferSize, TorchIndex, value );
      TorchIndex.pop_back();
      }
    }
}

template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::SetPixel( const IndexType & index, const PixelType & value )
{
  GetPixel( index ) = value;
}

template< typename TPixel, unsigned int VImageDimension >
typename TorchImage< TPixel, VImageDimension >::TorchImagePixelHelper
TorchImage< TPixel, VImageDimension >
::GetPixel( const IndexType & index )
{
  std::vector< int64_t > TorchIndex;
  for( SizeValueType i = 0; i < Self::ImageDimension; ++i )
    {
    TorchIndex.push_back( index[Self::ImageDimension-1-i] );
    }
  return TorchImagePixelHelper { m_Tensor, TorchIndex };
}

template< typename TPixel, unsigned int VImageDimension >
const typename TorchImage< TPixel, VImageDimension >::TorchImagePixelHelper
TorchImage< TPixel, VImageDimension >
::GetPixel( const IndexType & index ) const
{
  std::vector< int64_t > TorchIndex;
  for( SizeValueType i = 0; i < Self::ImageDimension; ++i )
    {
    TorchIndex.push_back( index[Self::ImageDimension-1-i] );
    }
  return TorchImagePixelHelper { m_Tensor, TorchIndex };
}

/** The pointer might be to GPU memory and, if so, cannot be directly
 * dereferenced */
template< typename TPixel, unsigned int VImageDimension >
TPixel *
TorchImage< TPixel, VImageDimension >
::GetBufferPointer()
{
  return reinterpret_cast< TPixel * >( m_Tensor.data_ptr< DeepScalarType >() );
}

/** The pointer might be to GPU memory and, if so, cannot be directly
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
  m_Allocated = data->m_Allocated;
  if( m_Allocated )
    {
    m_Tensor = torch::from_blob( data->m_Tensor.data_ptr(), data->m_Tensor.sizes(), data->m_Tensor.options() );
    }
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
  m_Allocated = false;
  m_CudaDeviceNumber = 0;
  m_Tensor = torch::Tensor();
}

template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::PrintSelf( std::ostream & os, Indent indent ) const
{
  Superclass::PrintSelf(os, indent);
  os
    << indent << "m_DeviceType: " << m_DeviceType << std::endl
    << indent << "m_Allocated: " << m_Allocated << std::endl
    << indent << "m_CudaDeviceNumber: " << m_CudaDeviceNumber << std::endl
    // << indent << "m_Tensor: " << m_Tensor << std::endl
    ;
}

} // end namespace itk

#endif

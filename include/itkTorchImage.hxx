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
TorchImage< TPixel, VImageDimension >
::TorchImage()
{
  m_DataManager = TorchImageDataManager< TorchImage< TPixel, VImageDimension > >::New();
  m_DataManager->SetTimeStamp( this->GetTimeStamp() );
  m_Graft = false;
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::Allocate( bool initialize )
{
  // allocate CPU memory - calling Allocate() in superclass
  Superclass::Allocate( initialize );

  if( !m_Graft )
    {
    AllocateGPU(); // allocate GPU memory
    }
}

//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::AllocateGPU()
{
  // Much of the same work is done in Initialize.  Where should it be done?!!!

  if( !m_Graft )
  {
    // allocate GPU memory
    const SizeType &bufferSize = this->GetBufferedRegion().GetSize();
    std::vector< int64_t > torchSize( TorchDimension );
    for (SizeValueType i = 0; i < ImageDimension; ++i)
      {
      torchSize.push_back( bufferSize[i] );
      }
    DimensionHelper< PixelType >::AppendSizes( torchSize );
    m_DataManager->SetTorchSize( torchSize );
    m_DataManager->SetImagePointer( this );
    m_DataManager->SetCPUBufferPointer( Superclass::GetBufferPointer() );
    m_DataManager->Allocate();

    /* prevent unnecessary copy from CPU to GPU at the beginning */
    m_DataManager->SetTimeStamp( this->GetTimeStamp() );
  }
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::Initialize()
{
  // CPU image initialize
  Superclass::Initialize();

  // GPU image initialize
  m_DataManager->Initialize();

  const SizeType &bufferSize = this->GetBufferedRegion().GetSize();
  std::vector< int64_t > torchSize ( TorchDimension );
  for (SizeValueType i = 0; i < ImageDimension; ++i)
    {
    torchSize.push_back( bufferSize[i] );
    }
  DimensionHelper< PixelType >::AppendSizes( torchSize );
  m_DataManager->SetTorchSize( torchSize );
  m_DataManager->SetImagePointer( this );
  m_DataManager->SetCPUBufferPointer( Superclass::GetBufferPointer() );
  m_DataManager->Allocate();

  /* prevent unnecessary copy from CPU to GPU at the beginning */
  m_DataManager->SetTimeStamp( this->GetTimeStamp() );
  m_Graft = false;
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::Modified() const
{
  Superclass::Modified();
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::FillBuffer( const TPixel &value )
{
  m_DataManager->SetGPUBufferStale();
  Superclass::FillBuffer( value );
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::SetPixel( const IndexType &index, const TPixel &value )
{
  m_DataManager->SetGPUBufferStale();
  Superclass::SetPixel( index, value );
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
const TPixel &
TorchImage< TPixel, VImageDimension >
::GetPixel( const IndexType &index ) const
{
  m_DataManager->UpdateCPUBuffer();
  return Superclass::GetPixel( index );
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
TPixel &
TorchImage< TPixel, VImageDimension >
::GetPixel( const IndexType &index )
{
  /* less conservative version - if you modify pixel value using
   * this pointer then you must set the image as modified manually */
  m_DataManager->UpdateCPUBuffer();
  return Superclass::GetPixel( index );
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
TPixel &
TorchImage< TPixel, VImageDimension >
::operator[]( const IndexType &index )
{
  /* less conservative version - if you modify pixel value using
   * this pointer then you must set the image as modified manually */
  m_DataManager->UpdateCPUBuffer();
  return Superclass::operator[]( index );
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
const TPixel &
TorchImage< TPixel, VImageDimension >
::operator[]( const IndexType &index ) const
{
  m_DataManager->UpdateCPUBuffer();
  return Superclass::operator[]( index );
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::SetPixelContainer( PixelContainer *container )
{
  Superclass::SetPixelContainer( container );

  m_DataManager->SetCPUStaleFlag( false );
  m_DataManager->SetGPUStaleFlag( true );
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::UpdateBuffers()
{
  m_DataManager->UpdateCPUBuffer();
  m_DataManager->UpdateGPUBuffer();
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::UpdateCPUBuffer()
{
  m_DataManager->UpdateCPUBuffer();
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::UpdateGPUBuffer()
{
  m_DataManager->UpdateGPUBuffer();
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
TPixel *
TorchImage< TPixel, VImageDimension >
::GetBufferPointer()
{
  /* less conservative version - if you modify pixel value using
   * this pointer then you must set the image as modified manually */
  m_DataManager->UpdateCPUBuffer();
  return Superclass::GetBufferPointer();
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
const TPixel *
TorchImage< TPixel, VImageDimension >
::GetBufferPointer() const
{
  m_DataManager->UpdateCPUBuffer();
  return Superclass::GetBufferPointer();
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::GraftITKImage( const DataObject *data )
{
  Superclass::Graft( data );
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::Graft( const DataObject *data )
{
  // call the superclass' implementation
  Superclass::Graft( data );

  if( data )
    {
    // Attempt to cast data to an TorchImageDataManagerType
    using TorchImageDataManagerType = TorchImageDataManager< TorchImage >;
    const TorchImageDataManagerType *ptr = nullptr;

    try
      {
      // Pass regular pointer to Graft() instead of smart pointer due to type casting problem
      ptr = dynamic_cast< const TorchImageDataManagerType * >( ( ( TorchImage * )data )->m_DataManager.GetPointer() );
      }
    catch( ... )
      {
      return;
      }

    if( ptr )
      {
      // Debug
      // std::cout << "GPU timestamp : " << m_DataManager->GetMTime() << ", CPU timestamp : " << this->GetMTime() << std::endl;

      // call GPU data graft function
      m_DataManager->SetImagePointer( this );
      m_DataManager->Graft( ptr );

      // Synchronize timestamp of TorchImage and TorchDataManager
      m_DataManager->SetTimeStamp( this->GetTimeStamp() );

      m_Graft = true;

      // Debug
      //std::cout << "GPU timestamp : " << m_DataManager->GetMTime() << ", CPU
      // timestamp : " << this->GetMTime() << std::endl;
      }
    else
      {
      // pointer could not be cast back down
      itkExceptionMacro(  << "itk::TorchImage::Graft() cannot cast "
        << typeid( data ).name() << " to "
        << typeid( const TorchImageDataManagerType * ).name() );
      }
    }
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
TorchImage< TPixel, VImageDimension >
::PrintSelf( std::ostream &os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  m_DataManager->PrintSelf( os, indent );
}


// Some compilers require that static constexpr members that are initialized when declared must nonetheless also be
// defined.
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
::TorchDimension;

} // namespace itk

#endif
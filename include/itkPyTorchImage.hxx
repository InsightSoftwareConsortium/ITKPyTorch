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

#ifndef itkPyTorchImage_hxx
#define itkPyTorchImage_hxx

#include "itkPyTorchImage.h"

namespace itk
{
//
// Constructor
//
template< typename TPixel, unsigned int VImageDimension >
PyTorchImage< TPixel, VImageDimension >
::PyTorchImage()
{
  m_DataManager = PyTorchImageDataManager< PyTorchImage< TPixel, VImageDimension > >::New();
  m_DataManager->SetTimeStamp( this->GetTimeStamp() );
  m_Graft = false;
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
PyTorchImage< TPixel, VImageDimension >
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
PyTorchImage< TPixel, VImageDimension >
::AllocateGPU()
{
  if( !m_Graft )
  {
    // allocate GPU memory
    const SizeType &bufferSize = this->GetBufferedRegion().GetSize();
    std::vector< SizeValueType > pyTorchSize;
    for (SizeValueType i = 0; i < ImageDimension; ++i)
      {
      pyTorchSize.push_back( bufferSize[i] );
      }
    DimensionHelper< PixelType >::AppendSizes( pyTorchSize );
    m_DataManager->SetPyTorchSize( pyTorchSize );
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
PyTorchImage< TPixel, VImageDimension >
::Initialize()
{
  // CPU image initialize
  Superclass::Initialize();

  // GPU image initialize
  m_DataManager->Initialize();

  const SizeType &bufferSize = this->GetBufferedRegion().GetSize();
  std::vector< SizeValueType > pyTorchSize;
  for (SizeValueType i = 0; i < ImageDimension; ++i)
    {
    pyTorchSize.push_back( bufferSize[i] );
    }
  DimensionHelper< PixelType >::AppendSizes( pyTorchSize );
  m_DataManager->SetPyTorchSize( pyTorchSize );
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
PyTorchImage< TPixel, VImageDimension >
::Modified() const
{
  Superclass::Modified();
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
PyTorchImage< TPixel, VImageDimension >
::FillBuffer( const TPixel &value )
{
  m_DataManager->SetGPUBufferStale();
  Superclass::FillBuffer( value );
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
PyTorchImage< TPixel, VImageDimension >
::SetPixel( const IndexType &index, const TPixel &value )
{
  m_DataManager->SetGPUBufferStale();
  Superclass::SetPixel( index, value );
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
const TPixel &
PyTorchImage< TPixel, VImageDimension >
::GetPixel( const IndexType &index ) const
{
  m_DataManager->UpdateCPUBuffer();
  return Superclass::GetPixel( index );
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
TPixel &
PyTorchImage< TPixel, VImageDimension >
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
PyTorchImage< TPixel, VImageDimension >
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
PyTorchImage< TPixel, VImageDimension >
::operator[]( const IndexType &index ) const
{
  m_DataManager->UpdateCPUBuffer();
  return Superclass::operator[]( index );
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
PyTorchImage< TPixel, VImageDimension >
::SetPixelContainer( PixelContainer *container )
{
  Superclass::SetPixelContainer( container );

  m_DataManager->SetCPUStaleFlag( false );
  m_DataManager->SetGPUStaleFlag( true );
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
PyTorchImage< TPixel, VImageDimension >
::UpdateBuffers()
{
  m_DataManager->UpdateCPUBuffer();
  m_DataManager->UpdateGPUBuffer();
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
PyTorchImage< TPixel, VImageDimension >
::UpdateCPUBuffer()
{
  m_DataManager->UpdateCPUBuffer();
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
PyTorchImage< TPixel, VImageDimension >
::UpdateGPUBuffer()
{
  m_DataManager->UpdateGPUBuffer();
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
TPixel *
PyTorchImage< TPixel, VImageDimension >
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
PyTorchImage< TPixel, VImageDimension >
::GetBufferPointer() const
{
  m_DataManager->UpdateCPUBuffer();
  return Superclass::GetBufferPointer();
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
PyTorchImage< TPixel, VImageDimension >
::GraftITKImage( const DataObject *data )
{
  Superclass::Graft( data );
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
PyTorchImage< TPixel, VImageDimension >
::Graft( const DataObject *data )
{
  // call the superclass' implementation
  Superclass::Graft( data );

  if( data )
  {
    // Attempt to cast data to an PyTorchImageDataManagerType
    using PyTorchImageDataManagerType = PyTorchImageDataManager< PyTorchImage >;
    const PyTorchImageDataManagerType *ptr;

    try
    {
      // Pass regular pointer to Graft() instead of smart pointer due to type casting problem
    ptr = dynamic_cast< const PyTorchImageDataManagerType * >( ( ( PyTorchImage * )data )->m_DataManager.GetPointer() );
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

      // Synchronize timestamp of PyTorchImage and PyTorchDataManager
      m_DataManager->SetTimeStamp( this->GetTimeStamp() );

      m_Graft = true;

      // Debug
      //std::cout << "GPU timestamp : " << m_DataManager->GetMTime() << ", CPU
      // timestamp : " << this->GetMTime() << std::endl;
    }
    else
    {
      // pointer could not be cast back down
      itkExceptionMacro(  << "itk::PyTorchImage::Graft() cannot cast "
        << typeid( data ).name() << " to "
        << typeid( const PyTorchImageDataManagerType * ).name() );
    }
  }
}


//------------------------------------------------------------------------------
template< typename TPixel, unsigned int VImageDimension >
void
PyTorchImage< TPixel, VImageDimension >
::PrintSelf( std::ostream &os, Indent indent ) const
{
  Superclass::PrintSelf( os, indent );

  m_DataManager->PrintSelf( os, indent );
}


// Some compilers want the static constexpr values to be defined, not merely declared.
template< typename TPixel, unsigned int VImageDimension >
template< typename TPixelType, typename TExtra >
constexpr typename PyTorchImage< TPixel, VImageDimension >::SizeValueType
PyTorchImage< TPixel, VImageDimension >
::PixelHelper< TPixelType, TExtra >
::NumberOfComponents;

template< typename TPixel, unsigned int VImageDimension >
constexpr unsigned int
PyTorchImage< TPixel, VImageDimension >
::ImageDimension;

template< typename TPixel, unsigned int VImageDimension >
constexpr at::ScalarType
PyTorchImage< TPixel, VImageDimension >
::PyTorchValueType;

} // namespace itk

#endif

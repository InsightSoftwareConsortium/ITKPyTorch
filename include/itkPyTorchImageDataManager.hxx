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

#ifndef itkPyTorchImageDataManager_hxx
#define itkPyTorchImageDataManager_hxx

#include "itkPyTorchImageDataManager.h"

//#define VERBOSE

namespace itk
{
template< typename TImage >
void
PyTorchImageDataManager< TImage >::SetImagePointer( typename ImageType::Pointer img )
{
  m_Image = img;
}


//------------------------------------------------------------------------------
template< typename TImage >
void
PyTorchImageDataManager< TImage >::Allocate()
{
  // Write me!!!
  m_IsCPUBufferAllocated = true;
  m_IsGPUBufferAllocated = true;
}


//------------------------------------------------------------------------------
template< typename TImage >
void
PyTorchImageDataManager< TImage >::Initialize()
{
  // Write me!!!  Release GPU memory if it exists!!!
  Superclass::Initialize();
}


//------------------------------------------------------------------------------
template< typename TImage >
void
PyTorchImageDataManager< TImage >::SetCPUBufferPointer( void *ptr )
{
  // Write me!!!
}


//------------------------------------------------------------------------------
template< typename TImage >
void
PyTorchImageDataManager< TImage >::UpdateCPUBuffer()
{
  if( this->m_CPUBufferLock )
    {
    return;
    }

  if( m_Image.IsNotNull() )
    {
    MutexHolderType holder( m_Mutex );

    unsigned long gpu_time       = this->GetMTime();
    TimeStamp     cpu_time_stamp = m_Image->GetTimeStamp();
    unsigned long cpu_time       = cpu_time_stamp.GetMTime();

    /* Why we check dirty flag and time stamp together?
    * Because existing CPU image filters do not use pixel/buffer
    * access function in PyTorchImage and therefore dirty flag is not
    * correctly managed. Therefore, we check the time stamp of
    * CPU and GPU data as well
    */
    if( ( m_IsCPUBufferDirty ||( gpu_time > cpu_time ) ) && m_IsGPUBufferAllocated && m_IsCPUBufferAllocated )
      {
      // Update CPU Buffer.  Does this change the location of the CPU data?!!!
      m_CPUTensor = m_GPUTensor.to(at::kCPU);

      m_Image->Modified();
      this->SetTimeStamp( m_Image->GetTimeStamp() );

      m_IsCPUBufferDirty = false;
      m_IsGPUBufferDirty = false;
      }
    }
}


//------------------------------------------------------------------------------
template< typename TImage >
void
PyTorchImageDataManager< TImage >::UpdateGPUBuffer()
{
  if( this->m_GPUBufferLock )
    {
    return;
    }

  if( m_Image.IsNotNull() )
    {
    MutexHolderType holder( m_Mutex );

    unsigned long gpu_time       = this->GetMTime();
    TimeStamp     cpu_time_stamp = m_Image->GetTimeStamp();
    unsigned long cpu_time       = m_Image->GetMTime();

    /* Why we check dirty flag and time stamp together?
    * Because existing CPU image filters do not use pixel/buffer
    * access function in PyTorchImage and therefore dirty flag is not
    * correctly managed. Therefore, we check the time stamp of
    * CPU and GPU data as well
    */
    if( ( m_IsGPUBufferDirty ||( gpu_time< cpu_time ) ) && m_IsCPUBufferAllocated && m_IsGPUBufferAllocated )
      {
      // Update GPU Buffer
      m_GPUTensor = m_CPUTensor.to(at::kCUDA);

      this->SetTimeStamp( cpu_time_stamp );

      m_IsCPUBufferDirty = false;
      m_IsGPUBufferDirty = false;
      }
    }
}


//------------------------------------------------------------------------------
template< typename TImage >
void
PyTorchImageDataManager< TImage >::Graft( const PyTorchImageDataManager *data )
{
  // std::cout << "GPU timestamp : " << this->GetMTime() << ", CPU timestamp : " << m_Image->GetMTime() << std::endl;

  // Write me!!!  Graft actual pixel values.
  Superclass::Graft( data );

  // std::cout << "GPU timestamp : " << this->GetMTime() << ", CPU timestamp : " << m_Image->GetMTime() << std::endl;
}


} // namespace itk

#endif

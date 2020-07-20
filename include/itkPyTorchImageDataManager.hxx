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
PyTorchImageDataManager< TImage >
::SetImagePointer( typename ImageType::Pointer img )
{
  m_Image = img;
}


//------------------------------------------------------------------------------
template< typename TImage >
void
PyTorchImageDataManager< TImage >
::Allocate()
{
  // Allocate memory for the GPU here.
  try
    {
    c10::TensorOptions options = torch::TensorOptions().dtype( ImageType::PyTorchValueType ).layout( torch::kStrided ).device( torch::kCUDA ).requires_grad( false );
    m_GPUTensor = torch::empty( m_Size, options );
    m_IsGPUBufferAllocated = true;
    }
  catch (...)
    {
    // What if there is no GPU?!!!
    }
}


//------------------------------------------------------------------------------
template< typename TImage >
void
PyTorchImageDataManager< TImage >
::Initialize()
{
  try
    {
    // Free up GPUTensor memory by replacing it with a very small tensor.  If there is no GPU this will fail.
    c10::TensorOptions options =
      torch::TensorOptions().dtype(torch::kFloat16).layout(torch::kStrided).device(torch::kCUDA).requires_grad(false);
    m_GPUTensor = torch::empty( { 1 }, options );
    m_IsGPUBufferAllocated = false;
    }
  catch (...) {}

  Superclass::Initialize();
}


//------------------------------------------------------------------------------
template< typename TImage >
void
PyTorchImageDataManager< TImage >
::SetPyTorchSize( const std::vector< int64_t > &pyTorchSize )
{
  m_Size = pyTorchSize;
}


//------------------------------------------------------------------------------
template< typename TImage >
void
PyTorchImageDataManager< TImage >
::SetCPUBufferPointer( void *ptr )
{
  // How do we know that m_Size is set correctly?!!!
  c10::TensorOptions options = torch::TensorOptions().dtype( ImageType::PyTorchValueType ).layout( torch::kStrided ).device( torch::kCPU ).requires_grad( false );
  m_CPUTensor = torch::from_blob( reinterpret_cast< DeepScalarType * >( ptr ), m_Size, options );
  m_IsCPUBufferAllocated = true;
}


//------------------------------------------------------------------------------
template< typename TImage >
void
PyTorchImageDataManager< TImage >
::UpdateCPUBuffer()
{
  if( m_IsCPUBufferLocked )
    {
    return;
    }

  if( m_Image.IsNotNull() )
    {
    MutexHolderType holder( m_Mutex );

    ModifiedTimeType gpu_time       = this->GetMTime();
    TimeStamp        cpu_time_stamp = m_Image->GetTimeStamp();
    ModifiedTimeType cpu_time       = cpu_time_stamp.GetMTime();

    /* Why we check stale flag and time stamp together?
    * Because existing CPU image filters do not use pixel/buffer
    * access function in PyTorchImage and therefore stale flag is not
    * correctly managed. Therefore, we check the time stamp of
    * CPU and GPU data as well
    */
    if( ( m_IsCPUBufferStale || gpu_time > cpu_time ) && m_IsGPUBufferAllocated && m_IsCPUBufferAllocated )
      {
      // Where is CPU buffer?
      const DeepScalarType *data_ptr = m_CPUTensor.data_ptr< DeepScalarType >();
      // Update CPU Buffer.
      m_CPUTensor = m_GPUTensor.to(at::kCPU);
      // If memory moves then objects with the old pointer will fail.
      itkAssertOrThrowMacro(data_ptr == m_CPUTensor.data_ptr< DeepScalarType >(), "Tensor moved within CPU memory");

      m_Image->Modified();
      this->SetTimeStamp( m_Image->GetTimeStamp() );

      m_IsCPUBufferStale = false;
      m_IsGPUBufferStale = false;
      }
    }
}


//------------------------------------------------------------------------------
template< typename TImage >
void
PyTorchImageDataManager< TImage >
::UpdateGPUBuffer()
{
  if( m_IsGPUBufferLocked )
    {
    return;
    }

  if( m_Image.IsNotNull() )
    {
    MutexHolderType holder( m_Mutex );

    ModifiedTimeType gpu_time       = this->GetMTime();
    TimeStamp        cpu_time_stamp = m_Image->GetTimeStamp();
    ModifiedTimeType cpu_time       = m_Image->GetMTime();

    /* Why we check stale flag and time stamp together?
    * Because existing CPU image filters do not use pixel/buffer
    * access function in PyTorchImage and therefore stale flag is not
    * correctly managed. Therefore, we check the time stamp of
    * CPU and GPU data as well
    */
    if( ( m_IsGPUBufferStale || gpu_time < cpu_time ) && m_IsCPUBufferAllocated && m_IsGPUBufferAllocated )
      {
      // Update GPU Buffer
      m_GPUTensor = m_CPUTensor.to(at::kCUDA);

      this->SetTimeStamp( cpu_time_stamp );

      m_IsCPUBufferStale = false;
      m_IsGPUBufferStale = false;
      }
    }
}


//------------------------------------------------------------------------------
template< typename TImage >
void
PyTorchImageDataManager< TImage >
::Graft( const PyTorchImageDataManager *data )
{
  // std::cout << "GPU timestamp : " << this->GetMTime() << ", CPU timestamp : " << m_Image->GetMTime() << std::endl;

  // Write me!!!  Graft actual pixel values.
  Superclass::Graft( data );

  // std::cout << "GPU timestamp : " << this->GetMTime() << ", CPU timestamp : " << m_Image->GetMTime() << std::endl;
}


} // namespace itk

#endif

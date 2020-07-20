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

#include "itkPyTorchDataManager.h"

namespace itk
{
// constructor
PyTorchDataManager::PyTorchDataManager()
{
  // Ask the PyTorchImageDataManager subclass to Initialize, which will also call Self::Initialize
  this->Initialize();
}


//------------------------------------------------------------------------------
PyTorchDataManager::~PyTorchDataManager()
{
}


//------------------------------------------------------------------------------
void
PyTorchDataManager::SetCPUStaleFlag( bool isStale )
{
  m_IsCPUBufferStale = isStale;
}


//------------------------------------------------------------------------------
void
PyTorchDataManager::SetGPUStaleFlag( bool isStale )
{
  m_IsGPUBufferStale = isStale;
}


//------------------------------------------------------------------------------
void
PyTorchDataManager::SetGPUBufferStale()
{
  this->UpdateCPUBuffer();
  m_IsGPUBufferStale = true;
}


//------------------------------------------------------------------------------
void
PyTorchDataManager::SetCPUBufferStale()
{
  this->UpdateGPUBuffer();
  m_IsCPUBufferStale = true;
}


//------------------------------------------------------------------------------
bool
PyTorchDataManager::Update()
{
  if( m_IsGPUBufferStale && m_IsCPUBufferStale )
    {
    itkExceptionMacro( "Cannot make up-to-date buffer because both CPU and GPU buffers are stale" );
    return false;
    }

  this->UpdateGPUBuffer();
  this->UpdateCPUBuffer();

  m_IsGPUBufferStale = false;
  m_IsCPUBufferStale = false;

  return true;
}


//------------------------------------------------------------------------------
void
PyTorchDataManager::Graft( const PyTorchDataManager *data )
{
  if( data )
    {
    m_IsCPUBufferStale = data->m_IsCPUBufferStale;
    m_IsGPUBufferStale = data->m_IsGPUBufferStale;
    }
}


//------------------------------------------------------------------------------
void
PyTorchDataManager::Initialize()
{
  m_IsCPUBufferAllocated = false;
  m_IsGPUBufferAllocated = false;
  m_IsGPUBufferStale = false;
  m_IsCPUBufferStale = false;
  m_IsCPUBufferLocked = false;
  m_IsGPUBufferLocked = false;
}


//------------------------------------------------------------------------------
void
PyTorchDataManager::PrintSelf( std::ostream &os, Indent indent ) const
{
  os << indent << "PyTorchDataManager( " << this << " )" << std::endl;
  os << indent << "m_IsGPUBufferAllocated: " << m_IsGPUBufferAllocated << std::endl;
  os << indent << "m_IsCPUBufferAllocated: " << m_IsCPUBufferAllocated << std::endl;
  os << indent << "m_IsGPUBufferStale: " << m_IsGPUBufferStale << std::endl;
  os << indent << "m_IsCPUBufferStale: " << m_IsCPUBufferStale << std::endl;
  os << indent << "m_IsCPUBufferLocked: " << m_IsCPUBufferLocked << std::endl;
  os << indent << "m_IsGPUBufferLocked: " << m_IsGPUBufferLocked << std::endl;
}


} // namespace itk

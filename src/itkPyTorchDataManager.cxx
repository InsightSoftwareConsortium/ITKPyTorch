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
PyTorchDataManager::SetCPUDirtyFlag( bool isDirty )
{
  m_IsCPUBufferDirty = isDirty;
}


//------------------------------------------------------------------------------
void
PyTorchDataManager::SetGPUDirtyFlag( bool isDirty )
{
  m_IsGPUBufferDirty = isDirty;
}


//------------------------------------------------------------------------------
void
PyTorchDataManager::SetGPUBufferDirty()
{
  // Why do we UpdateCPUBuffer before marking the GPUBuffer as dirty?!!!
  this->UpdateCPUBuffer();
  m_IsGPUBufferDirty = true;
}


//------------------------------------------------------------------------------
void
PyTorchDataManager::SetCPUBufferDirty()
{
  // Why do we UpdateGPUBuffer before marking the CPUBuffer as dirty?!!!
  this->UpdateGPUBuffer();
  m_IsCPUBufferDirty = true;
}


//------------------------------------------------------------------------------
bool
PyTorchDataManager::Update()
{
  if( m_IsGPUBufferDirty && m_IsCPUBufferDirty )
    {
    itkExceptionMacro( "Cannot make up-to-date buffer because both CPU and GPU buffers are dirty" );
    return false;
    }

  this->UpdateGPUBuffer();
  this->UpdateCPUBuffer();

  m_IsGPUBufferDirty = m_IsCPUBufferDirty = false;

  return true;
}


//------------------------------------------------------------------------------
void
PyTorchDataManager::Graft( const PyTorchDataManager *data )
{
  if( data )
    {
    m_IsCPUBufferDirty = data->m_IsCPUBufferDirty;
    m_IsGPUBufferDirty = data->m_IsGPUBufferDirty;
    }
}


//------------------------------------------------------------------------------
void
PyTorchDataManager::Initialize()
{
  m_IsCPUBufferAllocated = false;
  m_IsGPUBufferAllocated = false;
  m_IsGPUBufferDirty = false;
  m_IsCPUBufferDirty = false;
  m_CPUBufferLock = false;
  m_GPUBufferLock = false;
}


//------------------------------------------------------------------------------
void
PyTorchDataManager::PrintSelf( std::ostream &os, Indent indent ) const
{
  os << indent << "PyTorchDataManager( " << this << " )" << std::endl;
  os << indent << "m_IsGPUBufferAllocated: " << m_IsGPUBufferAllocated << std::endl;
  os << indent << "m_IsCPUBufferAllocated: " << m_IsCPUBufferAllocated << std::endl;
  os << indent << "m_IsGPUBufferDirty: " << m_IsGPUBufferDirty << std::endl;
  os << indent << "m_IsCPUBufferDirty: " << m_IsCPUBufferDirty << std::endl;
  os << indent << "m_CPUBufferLock: " << m_CPUBufferLock << std::endl;
  os << indent << "m_GPUBufferLock: " << m_GPUBufferLock << std::endl;
}


} // namespace itk

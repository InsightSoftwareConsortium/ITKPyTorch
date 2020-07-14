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
//#define VERBOSE

namespace itk
{
// constructor
PyTorchDataManager::PyTorchDataManager()
{
  m_ContextManager = PyTorchContextManager::GetInstance();
  m_GPUBuffer = nullptr;
  m_CPUBuffer = nullptr;

  this->Initialize();
}

PyTorchDataManager::~PyTorchDataManager()
{
  if (m_GPUBuffer)
  {
    clReleaseMemObject(m_GPUBuffer);
  }
}

void
PyTorchDataManager::SetBufferSize(unsigned int num)
{
  m_BufferSize = num;
}

void
PyTorchDataManager::SetBufferFlag(cl_mem_flags flags)
{
  m_MemFlags = flags;
}

void
PyTorchDataManager::Allocate()
{
  cl_int errid;

  if (m_BufferSize > 0)
  {
#ifdef VERBOSE
    std::cout << this << "::Allocate Create GPU buffer of size " << m_BufferSize << " Bytes" << std::endl;
#endif
    m_GPUBuffer = clCreateBuffer(m_ContextManager->GetCurrentContext(), m_MemFlags, m_BufferSize, nullptr, &errid);
    OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);
    m_IsGPUBufferDirty = true;
  }


  // this->UpdateGPUBuffer();
}

void
PyTorchDataManager::SetCPUBufferPointer(void * ptr)
{
  m_CPUBuffer = ptr;
}

void
PyTorchDataManager::SetCPUDirtyFlag(bool isDirty)
{
  m_IsCPUBufferDirty = isDirty;
}

void
PyTorchDataManager::SetGPUDirtyFlag(bool isDirty)
{
  m_IsGPUBufferDirty = isDirty;
}

void
PyTorchDataManager::SetGPUBufferDirty()
{
  this->UpdateCPUBuffer();
  m_IsGPUBufferDirty = true;
}

void
PyTorchDataManager::SetCPUBufferDirty()
{
  this->UpdateGPUBuffer();
  m_IsCPUBufferDirty = true;
}

void
PyTorchDataManager::UpdateCPUBuffer()
{
  MutexHolderType holder(m_Mutex);

  if (m_IsCPUBufferDirty && m_GPUBuffer != nullptr && m_CPUBuffer != nullptr)
  {
    cl_int errid;
#ifdef VERBOSE
    std::cout << this << "::UpdateCPUBuffer GPU->CPU data copy " << m_GPUBuffer << "->" << m_CPUBuffer << std::endl;
#endif
    errid = clEnqueueReadBuffer(m_ContextManager->GetCommandQueue(m_CommandQueueId),
                                m_GPUBuffer,
                                CL_TRUE,
                                0,
                                m_BufferSize,
                                m_CPUBuffer,
                                0,
                                nullptr,
                                nullptr);
    OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);

    m_IsCPUBufferDirty = false;
  }
}

void
PyTorchDataManager::UpdateGPUBuffer()
{
  MutexHolderType mutexHolder(m_Mutex);

  if (m_IsGPUBufferDirty && m_CPUBuffer != nullptr && m_GPUBuffer != nullptr)
  {
    cl_int errid;
#ifdef VERBOSE
    std::cout << this << "::UpdateGPUBuffer CPU->GPU data copy " << m_CPUBuffer << "->" << m_GPUBuffer << std::endl;
#endif
    errid = clEnqueueWriteBuffer(m_ContextManager->GetCommandQueue(m_CommandQueueId),
                                 m_GPUBuffer,
                                 CL_TRUE,
                                 0,
                                 m_BufferSize,
                                 m_CPUBuffer,
                                 0,
                                 nullptr,
                                 nullptr);
    OpenCLCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);

    m_IsGPUBufferDirty = false;
  }
}

cl_mem *
PyTorchDataManager::GetGPUBufferPointer()
{
  SetCPUBufferDirty();
  return &m_GPUBuffer;
}

void *
PyTorchDataManager::GetCPUBufferPointer()
{
  SetGPUBufferDirty();
  return m_CPUBuffer;
}

bool
PyTorchDataManager::Update()
{
  if (m_IsGPUBufferDirty && m_IsCPUBufferDirty)
  {
    itkExceptionMacro("Cannot make up-to-date buffer because both CPU and GPU buffers are dirty");
  }

  this->UpdateGPUBuffer();
  this->UpdateCPUBuffer();

  m_IsGPUBufferDirty = m_IsCPUBufferDirty = false;

  return true;
}

/**
 * NOTE: each device has a command queue. Therefore, changing command queue
 *       means change a compute device.
 */
void
PyTorchDataManager::SetCurrentCommandQueue(int queueid)
{
  if (queueid >= 0 && queueid < (int)m_ContextManager->GetNumberOfCommandQueues())
  {
    this->UpdateCPUBuffer();

    // Assumption: different command queue is assigned to different device
    m_CommandQueueId = queueid;

    m_IsGPUBufferDirty = true;
  }
  else
  {
    itkWarningMacro("Not a valid command queue id");
  }
}

int
PyTorchDataManager::GetCurrentCommandQueueID() const
{
  return m_CommandQueueId;
}

void
PyTorchDataManager::Graft(const PyTorchDataManager * data)
{
  if (data)
  {
    m_BufferSize = data->m_BufferSize;
    m_ContextManager = data->m_ContextManager;
    m_CommandQueueId = data->m_CommandQueueId;

    if (m_GPUBuffer) // Decrease reference count to GPU memory
    {
      clReleaseMemObject(m_GPUBuffer);
    }
    if (data->m_GPUBuffer) // Increase reference count to GPU memory
    {
      clRetainMemObject(data->m_GPUBuffer);
    }

    m_GPUBuffer = data->m_GPUBuffer;

    m_CPUBuffer = data->m_CPUBuffer;
    //    m_Platform  = data->m_Platform;
    //    m_Context   = data->m_Context;
    m_IsCPUBufferDirty = data->m_IsCPUBufferDirty;
    m_IsGPUBufferDirty = data->m_IsGPUBufferDirty;
  }
}

void
PyTorchDataManager::Initialize()
{
  if (m_ContextManager->GetNumberOfCommandQueues() > 0)
  {
    m_CommandQueueId = 0; // default command queue
  }

  if (m_GPUBuffer) // Release GPU memory if exists
  {
    clReleaseMemObject(m_GPUBuffer);
  }

  m_BufferSize = 0;
  m_GPUBuffer = nullptr;
  m_CPUBuffer = nullptr;
  m_MemFlags = CL_MEM_READ_WRITE; // default flag
  m_IsGPUBufferDirty = false;
  m_IsCPUBufferDirty = false;
}

void
PyTorchDataManager::PrintSelf(std::ostream & os, Indent indent) const
{
  os << indent << "PyTorchDataManager (" << this << ")" << std::endl;
  os << indent << "m_BufferSize: " << m_BufferSize << std::endl;
  os << indent << "m_IsGPUBufferDirty: " << m_IsGPUBufferDirty << std::endl;
  os << indent << "m_GPUBuffer: " << m_GPUBuffer << std::endl;
  os << indent << "m_IsCPUBufferDirty: " << m_IsCPUBufferDirty << std::endl;
  os << indent << "m_CPUBuffer: " << m_CPUBuffer << std::endl;
}

} // namespace itk

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

#ifndef itkPyTorchDataManager_h
#define itkPyTorchDataManager_h

#include "itkObject.h"
#include "itkDataObject.h"
#include "itkObjectFactory.h"
#include "itkOpenCLContext.h"
#include <mutex>

namespace itk
{
/** \class PyTorchDataManager
 * \brief GPU memory manager implemented using OpenCL. Required by PyTorchImage class.
 *
 * This class serves as a base class for GPU data container for PyTorchImage class,
 * which is similar to ImageBase class for Image class. However, all the image-related
 * meta data will be already stored in image class( parent of PyTorchImage ), therefore
 * we did not name it PyTorchImageBase. Rather, this class is a GPU-specific data manager
 * that provides functionalities for CPU-GPU data synchronization and grafting GPU data.
 *
 * \ingroup ITKPyTorchCommon
 */
class PyTorchDataManager : public Object   //DataObject//
{
  /** allow PyTorchKernelManager to access GPU buffer pointer */
  friend class OpenCLKernelManager;

public:
  ITK_DISALLOW_COPY_AND_ASSIGN( PyTorchDataManager );

  using Self = PyTorchDataManager;
  using Superclass = Object;
  using Pointer = SmartPointer< Self >;
  using ConstPointer = SmartPointer< const Self >;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information( and related methods ). */
  itkTypeMacro( PyTorchDataManager, Object );

  using MutexHolderType = std::lock_guard< std::mutex >;

  /** total buffer size in bytes */
  virtual void SetBufferSize( unsigned int num );

  virtual unsigned int GetBufferSize() const
  {
    return m_BufferSize;
  }

  virtual void SetBufferFlag( cl_mem_flags flags );

  virtual void SetCPUBufferPointer( void *ptr );

  virtual void SetCPUDirtyFlag( bool isDirty );

  virtual void SetGPUDirtyFlag( bool isDirty );

  /** Make GPU up-to-date and mark CPU as dirty.
   * Call this function when you want to modify CPU data */
  virtual void SetCPUBufferDirty();

  /** Make CPU up-to-date and mark GPU as dirty.
   * Call this function when you want to modify GPU data */
  virtual void SetGPUBufferDirty();

  virtual bool IsCPUBufferDirty() const
  {
    return m_IsCPUBufferDirty;
  }

  virtual bool IsGPUBufferDirty() const
  {
    return m_IsGPUBufferDirty;
  }

  /** actual GPU->CPU memory copy takes place here */
  virtual void UpdateCPUBuffer();

  /** actual CPU->GPU memory copy takes place here */
  virtual void UpdateGPUBuffer();

  virtual void Allocate();

  /** Synchronize CPU and GPU buffers( using dirty flags ) */
  virtual bool Update();

  /** Method for grafting the content of one PyTorchDataManager into another one */
  virtual void Graft( const PyTorchDataManager *data );

  /** Initialize PyTorchDataManager */
  virtual void Initialize();

  /** Get GPU buffer pointer */
  virtual cl_mem *GetGPUBufferPointer();

  /** Get CPU buffer pointer */
  virtual void *GetCPUBufferPointer();

  /** Make CPU buffer locked to avoid extra update from ITK pipeline. */
  // How does this interact with the mutex?!!!
  virtual void SetCPUBufferLock( const bool v ) { this->m_CPUBufferLock = v; }
  itkGetConstReferenceMacro( CPUBufferLock, bool );

  /** Make GPU buffer locked to avoid extra update from ITK pipeline. */
  // How does this interact with the mutex?!!!
  virtual void SetGPUBufferLock( const bool v ) { this->m_GPUBufferLock = v; }
  itkGetConstReferenceMacro( GPUBufferLock, bool );

protected:

  PyTorchDataManager();
  virtual ~PyTorchDataManager();
  virtual void PrintSelf( std::ostream &os, Indent indent ) const override;

protected:

  unsigned int m_BufferSize; // # of bytes

  OpenCLContext *m_Context;

  /** buffer type */
  cl_mem_flags m_MemFlags;

  /** buffer pointers */
  cl_mem m_GPUBuffer;
  void *m_CPUBuffer;

  /** checks if buffer needs to be updated */
  bool m_IsGPUBufferDirty;
  bool m_IsCPUBufferDirty;

  /** extra safety flags */
  bool m_CPUBufferLock;
  bool m_GPUBufferLock;

  /** Mutex lock to prevent r/w hazard for multithreaded code */
  std::mutex m_Mutex;
};

} // namespace itk

#endif

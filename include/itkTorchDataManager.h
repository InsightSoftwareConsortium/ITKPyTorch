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

#ifndef itkTorchDataManager_h
#define itkTorchDataManager_h

#include "itkObject.h"
#include "itkDataObject.h"
#include "itkObjectFactory.h"
#include <mutex>

namespace itk
{
/** \class TorchDataManager
 * \brief GPU memory manager implemented using libtorch. Required by TorchImage class.
 *
 * This class serves as a base class for GPU data container for TorchImage class,
 * which is similar to ImageBase class for Image class. However, all the image-related
 * meta data will be already stored in image class( parent of TorchImage ), therefore
 * we did not name it TorchImageBase. Rather, this class is a GPU-specific data manager
 * that provides functionalities for CPU-GPU data synchronization and grafting GPU data.
 *
 * \ingroup ITKTorchCommon
 */
class TorchDataManager : public Object   //DataObject//
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN( TorchDataManager );

  using Self = TorchDataManager;
  using Superclass = Object;
  using Pointer = SmartPointer< Self >;
  using ConstPointer = SmartPointer< const Self >;

  /** Method for creation through the object factory. */
  // itkNewMacro( Self );          // Cannot because class is abstract.  Is this a problem?!!!

  /** Run-time type information( and related methods ). */
  itkTypeMacro( TorchDataManager, Object );

  using MutexType = std::mutex;
  using MutexHolderType = std::lock_guard< MutexType >;

  virtual void SetCPUStaleFlag( bool isStale );

  virtual void SetGPUStaleFlag( bool isStale );

  /** Make GPU up-to-date and mark CPU as stale.
   * Call this function when you want to modify CPU data */
  virtual void SetCPUBufferStale();

  /** Make CPU up-to-date and mark GPU as stale.
   * Call this function when you want to modify GPU data */
  virtual void SetGPUBufferStale();

  virtual bool IsCPUBufferStale() const
  {
    return m_IsCPUBufferStale;
  }

  virtual bool IsGPUBufferStale() const
  {
    return m_IsGPUBufferStale;
  }

  /** actual GPU->CPU memory copy takes place here */
  virtual void UpdateCPUBuffer() = 0;

  /** actual CPU->GPU memory copy takes place here */
  virtual void UpdateGPUBuffer() = 0;

  /** Synchronize CPU and GPU buffers( using stale flags ) */
  virtual bool Update();

  /** Method for grafting the content of one TorchDataManager into another one */
  virtual void Graft( const TorchDataManager *data );

  /** Initialize TorchDataManager */
  virtual void Initialize();

  /** Make CPU buffer locked to avoid extra update from ITK pipeline. */
  virtual void SetIsCPUBufferLocked( const bool v ) { m_IsCPUBufferLocked = v; }
  itkGetConstReferenceMacro( IsCPUBufferLocked, bool );

  /** Make GPU buffer locked to avoid extra update from ITK pipeline. */
  virtual void SetIsGPUBufferLocked( const bool v ) { m_IsGPUBufferLocked = v; }
  itkGetConstReferenceMacro( IsGPUBufferLocked, bool );

protected:

  TorchDataManager();
  virtual ~TorchDataManager();
  virtual void PrintSelf( std::ostream &os, Indent indent ) const override;

protected:

  /** checks if buffer has been allocated */
  bool m_IsGPUBufferAllocated;  // Make sure I am updated appropriately!!!
  bool m_IsCPUBufferAllocated;  // Make sure I am updated appropriately!!!

  /** checks if buffer needs to be updated */
  bool m_IsGPUBufferStale;
  bool m_IsCPUBufferStale;

  /** extra safety flags */
  bool m_IsCPUBufferLocked;
  bool m_IsGPUBufferLocked;

  /** Mutex lock to prevent r/w hazard for multithreaded code */
  MutexType m_Mutex;
};

} // namespace itk

#endif

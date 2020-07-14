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

#ifndef itkPyTorchImageDataManager_h
#define itkPyTorchImageDataManager_h

#include <itkObject.h>
#include <itkTimeStamp.h>
#include <itkLightObject.h>
#include <itkObjectFactory.h>
#include "itkOpenCLUtil.h"
#include "itkPyTorchDataManager.h"
#include "itkPyTorchContextManager.h"
#include <mutex>

namespace itk
{
template <typename TPixel, unsigned int NDimension>
class PyTorchImage;

/**
 * \class PyTorchImageDataManager
 *
 * DataManager for PyTorchImage. This class will take care of data synchronization
 * between CPU Image and GPU Image.
 *
 * \ingroup ITKPYTORCHCommon
 */
template <typename ImageType>
class ITK_TEMPLATE_EXPORT PyTorchImageDataManager : public PyTorchDataManager
{
  // allow PyTorchKernelManager to access GPU buffer pointer
  friend class PyTorchKernelManager;
  friend class PyTorchImage<typename ImageType::PixelType, ImageType::ImageDimension>;

public:
  ITK_DISALLOW_COPY_AND_ASSIGN(PyTorchImageDataManager);

  using Self = PyTorchImageDataManager;
  using Superclass = PyTorchDataManager;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  itkTypeMacro(PyTorchImageDataManager, PyTorchDataManager);

  static constexpr unsigned int ImageDimension = ImageType::ImageDimension;

  itkGetModifiableObjectMacro(GPUBufferedRegionIndex, PyTorchDataManager);
  itkGetModifiableObjectMacro(GPUBufferedRegionSize, PyTorchDataManager);

  void
  SetImagePointer(ImageType * img);

  ImageType *
  GetImagePointer()
  {
    return this->m_Image.GetPointer();
  }

  /** actual GPU->CPU memory copy takes place here */
  virtual void
  MakeCPUBufferUpToDate();

  /** actual CPU->GPU memory copy takes place here */
  virtual void
  MakeGPUBufferUpToDate();

protected:
  PyTorchImageDataManager() = default;
  ~PyTorchImageDataManager() override = default;

private:
  WeakPointer<ImageType> m_Image; // WeakPointer has to be used here
                                  // to avoid SmartPointer loop
  int                              m_BufferedRegionIndex[ImageType::ImageDimension];
  int                              m_BufferedRegionSize[ImageType::ImageDimension];
  typename PyTorchDataManager::Pointer m_GPUBufferedRegionIndex;
  typename PyTorchDataManager::Pointer m_GPUBufferedRegionSize;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkPyTorchImageDataManager.hxx"
#endif

#endif

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
#include "itkPyTorchImage.h"
#include "itkPyTorchDataManager.h"

namespace itk
{
/**
 * \class PyTorchImage Data Management
 *
 * DataManager for PyTorchImage. This class will take care of data synchronization
 * between CPU Image and GPU Image.
 *
 * \ingroup ITKPyTorchCommon
 */
template< typename TPixel, unsigned int NDimension >
class PyTorchImage;

template< typename TImage >
class PyTorchImageDataManager : public PyTorchDataManager
{
public:

  using Self = PyTorchImageDataManager;
  using Superclass = PyTorchDataManager;
  using Pointer = SmartPointer< Self >;
  using ConstPointer = SmartPointer< const Self >;
  using ImageType = TImage;
  using PixelType = typename ImageType::PixelType;
  using ValueType = typename ImageType::ImageType;

  itkNewMacro( Self );
  itkTypeMacro( PyTorchImageDataManager, PyTorchDataManager );

  virtual void Allocate();

  virtual void Initialize();

  virtual void SetCPUBufferPointer( void *ptr );

  virtual void SetImagePointer( typename ImageType::Pointer img ) override;

  /** actual GPU->CPU memory copy takes place here */
  virtual void UpdateCPUBuffer() override;

  /** actual CPU->GPU memory copy takes place here */
  virtual void UpdateGPUBuffer() override;

  /** Grafting GPU Image Data */
  virtual void Graft( const PyTorchImageDataManager *data ) override;

protected:
  /** Storage for CPU and GPU tensors is type specific, so we have it here instead of in the base class PyTorchDataManager */
  torch::Tensor m_CPUTensor;
  torch::Tensor m_GPUTensor;
  virtual ValueType *GetCPUBufferPointer()
    {
    return m_CPUTensor.data< ValueType >();
    }
  virtual const ValueType *GetCPUBufferPointer() const
    {
    return m_CPUTensor.data< ValueType >();
    }

  PyTorchImageDataManager() { m_Image = nullptr; }
  virtual ~PyTorchImageDataManager() {}

private:

  PyTorchImageDataManager( const Self & );   // purposely not implemented
  Self &operator=( const Self & );           // purposely not implemented

  typename ImageType::Pointer m_Image;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkPyTorchImageDataManager.hxx"
#endif

#endif

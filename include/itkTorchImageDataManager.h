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

#ifndef itkTorchImageDataManager_h
#define itkTorchImageDataManager_h

#include <itkObject.h>
#include <itkTimeStamp.h>
#include <itkLightObject.h>
#include <itkObjectFactory.h>
#include "itkTorchImage.h"
#include "itkTorchDataManager.h"

namespace itk
{
/**
 * \class TorchImage Data Management
 *
 * DataManager for TorchImage. This class will take care of data synchronization
 * between CPU Image and GPU Image.
 *
 * \ingroup ITKTorchCommon
 */
template< typename TPixel, unsigned int NDimension >
class ITK_FORWARD_EXPORT TorchImage;

template< typename TImage >
class ITK_TEMPLATE_EXPORT TorchImageDataManager : public TorchDataManager
{
public:

  using Self = TorchImageDataManager;
  using Superclass = TorchDataManager;
  using Pointer = SmartPointer< Self >;
  using ConstPointer = SmartPointer< const Self >;
  using ImageType = TImage;
  friend ImageType;
  using PixelType = typename ImageType::PixelType;
  using ValueType = typename ImageType::ValueType;
  using DeepScalarType = typename ImageType::DeepScalarType;

  itkNewMacro( Self );
  itkTypeMacro( TorchImageDataManager, TorchDataManager );

  virtual void Allocate();

  virtual void Initialize();

  virtual void SetTorchSize( const std::vector< int64_t > &torchSize );

  virtual void SetCPUBufferPointer( void *ptr );

  virtual void SetImagePointer( typename ImageType::Pointer img );

  /** actual GPU->CPU memory copy takes place here */
  virtual void UpdateCPUBuffer() override;

  /** actual CPU->GPU memory copy takes place here */
  virtual void UpdateGPUBuffer() override;

  /** Grafting GPU Image Data */
  virtual void Graft( const TorchImageDataManager *data );

protected:
  /** Storage for CPU and GPU tensors is type specific, so we have it here instead of in the base class TorchDataManager */
  torch::Tensor m_CPUTensor;
  torch::Tensor m_GPUTensor;

  virtual DeepScalarType *GetCPUBufferPointer()
    {
    return m_CPUTensor.data_ptr< DeepScalarType >();
    }
  virtual const DeepScalarType *GetCPUBufferPointer() const
    {
    return m_CPUTensor.data_ptr< DeepScalarType >();
    }

  TorchImageDataManager() { m_Image = nullptr; }
  virtual ~TorchImageDataManager() {}

private:

  TorchImageDataManager( const Self & );   // purposely not implemented
  Self &operator=( const Self & );           // purposely not implemented

  typename ImageType::Pointer m_Image;

  std::vector< int64_t > m_Size;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkTorchImageDataManager.hxx"
#endif

#endif

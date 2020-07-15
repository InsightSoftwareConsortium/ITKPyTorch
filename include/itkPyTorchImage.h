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

#ifndef itkPyTorchImage_h
#define itkPyTorchImage_h

#include "itkImage.h"
#include "itkPyTorchImageDataManager.h"
#include "itkVersion.h"
#include "itkObjectFactoryBase.h"

namespace itk
{
/** \class PyTorchImage
 *  \brief Templated n-dimensional image class for the GPU.
 *
 * Derived from itk Image class to use with GPU image filters.
 * This class manages both CPU and GPU memory implicitly, and
 * can be used with non-GPU itk filters as well. Memory transfer
 * between CPU and GPU is done automatically and implicitly.
 *
 * \ingroup ITKPyTorchCommon
 */
template< typename TPixel, unsigned int VImageDimension = 2 >
class PyTorchImage : public Image< TPixel, VImageDimension >
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN( PyTorchImage );

  using Self = PyTorchImage;
  using Superclass = Image< TPixel, VImageDimension >;
  using Pointer = SmartPointer< Self >;
  using ConstPointer = SmartPointer< const Self >;
  using ConstWeakPointer = WeakPointer< const Self >;

  itkNewMacro( Self );

  itkTypeMacro( PyTorchImage, Image );

  static constexpr unsigned int ImageDimension = VImageDimension;

  using PixelType = typename Superclass::PixelType;
  using ValueType = typename Superclass::ValueType;
  using InternalPixelType = typename Superclass::InternalPixelType;
  using IOPixelType = typename Superclass::IOPixelType;
  using DirectionType = typename Superclass::DirectionType;
  using SpacingType = typename Superclass::SpacingType;
  using PixelContainer = typename Superclass::PixelContainer;
  using SizeType = typename Superclass::SizeType;
  using IndexType = typename Superclass::IndexType;
  using OffsetType = typename Superclass::OffsetType;
  using RegionType = typename Superclass::RegionType;
  using PixelContainerPointer = typename PixelContainer::Pointer;
  using PixelContainerConstPointer = typename PixelContainer::ConstPointer;
  using AccessorType = typename Superclass::AccessorType;

  using AccessorFunctorType = DefaultPixelAccessorFunctor< Self >;

  using NeighborhoodAccessorFunctorType = NeighborhoodAccessorFunctor< Self >;
  // NeighborhoodAccessorFunctorType;

  /** Allocate CPU and GPU memory space */
  virtual void Allocate( bool initialize = false ) override;

  virtual void AllocateGPU() override;

  virtual void Initialize() override;

  virtual void FillBuffer( const TPixel &value ) override;

  virtual void SetPixel( const IndexType &index, const TPixel &value ) override;

  virtual const TPixel &GetPixel( const IndexType &index ) const override;

  virtual TPixel &GetPixel( const IndexType &index ) override;

  virtual const TPixel &operator[]( const IndexType &index ) const override;

  virtual TPixel &operator[]( const IndexType &index ) override;

  /** Explicit synchronize CPU/GPU buffers */
  virtual void UpdateBuffers() override;

  /** Explicit synchronize CPU/GPU buffers */
  virtual void UpdateCPUBuffer() override;

  virtual void UpdateGPUBuffer() override;

  /** Get CPU buffer pointer */
  virtual TPixel *GetBufferPointer() override;

  virtual const TPixel *GetBufferPointer() const override;

  /** Return the Pixel Accessor object */
  virtual AccessorType GetPixelAccessor() override
  {
    m_DataManager->SetGPUBufferDirty();
    return Superclass::GetPixelAccessor();
  }


  /** Return the Pixel Accesor object */
  virtual const AccessorType GetPixelAccessor() const override
  {
    m_DataManager->UpdateCPUBuffer();
    return Superclass::GetPixelAccessor();
  }


  /** Return the NeighborhoodAccessor functor */
  virtual NeighborhoodAccessorFunctorType GetNeighborhoodAccessor() override
  {
    m_DataManager->SetGPUBufferDirty();
    return NeighborhoodAccessorFunctorType();
  }


  /** Return the NeighborhoodAccessor functor */
  virtual const NeighborhoodAccessorFunctorType GetNeighborhoodAccessor() const override
  {
    m_DataManager->UpdateCPUBuffer();
    return NeighborhoodAccessorFunctorType();
  }


  virtual void SetPixelContainer( PixelContainer *container ) override;

  /** Return a pointer to the container. */
  virtual PixelContainer *GetPixelContainer() override
  {
    m_DataManager->SetGPUBufferDirty();
    return Superclass::GetPixelContainer();
  }


  virtual const PixelContainer *GetPixelContainer() const override
  {
    m_DataManager->UpdateCPUBuffer();
    return Superclass::GetPixelContainer();
  }


  virtual void SetCurrentCommandQueue( int queueid ) override
  {
    m_DataManager->SetCurrentCommandQueue( queueid );
  }


  virtual int GetCurrentCommandQueueId() override
  {
    return m_DataManager->GetCurrentCommandQueueId();
  }


  // Returns base class?!!!
  virtual PyTorchDataManager::Pointer GetPyTorchDataManager() const override;

  /** Override DataHasBeenGenerated() in DataObject class.
   * We need this because CPU time stamp is always bigger
   * than GPU's. That is because Modified() is called at
   * the end of each filter in the pipeline so although we
   * increment GPU's time stamp in GPUGenerateData() the
   * CPU's time stamp will be increased after that.
   */
  virtual void DataHasBeenGenerated() override
  {
    Superclass::DataHasBeenGenerated();

    if( m_DataManager->IsCPUBufferDirty() )
    {
      m_DataManager->Modified();
    }

  }


  /** Graft the data and information from one PyTorchImage to another. */
  virtual void Graft( const DataObject *data ) override;

  virtual void GraftITKImage( const DataObject *data ) override;

  /** Whenever the image has been modified, set the GPU Buffer to dirty */
  virtual void Modified() const;

  /** Get matrices intended to help with the conversion of Index coordinates
   *  to PhysicalPoint coordinates */
  itkGetConstReferenceMacro( IndexToPhysicalPoint, DirectionType );
  itkGetConstReferenceMacro( PhysicalPointToIndex, DirectionType );

protected:

  PyTorchImage();
  virtual ~PyTorchImage() {}

  virtual void PrintSelf( std::ostream &os, Indent indent ) const override;

private:
  bool m_Graft;

  typename PyTorchImageDataManager< PyTorchImage >::Pointer m_DataManager;
};

//------------------------------------------------------------------------------
template< typename T >
class PyTorchTraits
{
public:
  using Type = T;
};

template< typename TPixelType, unsigned int NDimension >
class PyTorchTraits< Image< TPixelType, NDimension > >
{
public:
  using Type = PyTorchImage< TPixelType, NDimension >;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkPyTorchImage.hxx"
#endif

#endif

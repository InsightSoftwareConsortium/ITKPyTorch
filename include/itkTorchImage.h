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

#ifndef itkTorchImage_h
#define itkTorchImage_h

#include <torch/torch.h>
#include "itkImage.h"
#include "itkTorchImageDataManager.h"
#include "itkVersion.h"
#include "itkObjectFactoryBase.h"
#include "itkRGBPixel.h"
#include "itkRGBAPixel.h"
#include "itkVector.h"
#include "itkCovariantVector.h"

namespace itk
{
/** \class TorchImage
 *  \brief Templated n-dimensional image class for the GPU.
 *
 * Derived from itk Image class to use with GPU image filters.
 * This class manages both CPU and GPU memory implicitly, and
 * can be used with non-GPU itk filters as well. Memory transfer
 * between CPU and GPU is done automatically and implicitly.
 *
 * \ingroup ITKTorchCommon
 */
template< typename TPixel, unsigned int VImageDimension = 2 >
class ITK_TEMPLATE_EXPORT TorchImage : public Image< TPixel, VImageDimension >
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN( TorchImage );

  using Self = TorchImage;
  using Superclass = Image< TPixel, VImageDimension >;
  using Pointer = SmartPointer< Self >;
  using ConstPointer = SmartPointer< const Self >;
  using ConstWeakPointer = WeakPointer< const Self >;

  itkNewMacro( Self );

  itkTypeMacro( TorchImage, Image );

  using PixelType = typename Superclass::PixelType;
  using InternalPixelType = typename Superclass::InternalPixelType;
  using IOPixelType = typename Superclass::IOPixelType;
  using DirectionType = typename Superclass::DirectionType;
  using SpacingType = typename Superclass::SpacingType;
  using PixelContainer = typename Superclass::PixelContainer;
  using SizeType = typename Superclass::SizeType;
  using SizeValueType = typename SizeType::SizeValueType;
  using IndexType = typename Superclass::IndexType;
  using OffsetType = typename Superclass::OffsetType;
  using RegionType = typename Superclass::RegionType;
  using PixelContainerPointer = typename PixelContainer::Pointer;
  using PixelContainerConstPointer = typename PixelContainer::ConstPointer;

  using AccessorType = typename Superclass::AccessorType;
  using AccessorFunctorType = DefaultPixelAccessorFunctor< Self >;
  using NeighborhoodAccessorFunctorType = NeighborhoodAccessorFunctor< Self >;
  // NeighborhoodAccessorFunctorType;

  // For the default case that the number of vector components is not determined at compile time:
  // We define our own void_t because it is C++-17; we do it in a way that works even for C++14.
  template< typename... Ts > struct myMake_void { typedef void type; };
  template< typename... Ts > using myVoid_t = typename myMake_void< Ts... >::type;

  template< typename TPixelType, typename = void >
  struct PixelHelper
    {
    using PixelType = TPixelType;
    using ValueType = typename PixelType::ValueType;
    static constexpr SizeValueType NumberOfComponents = -1;
    using PixelTypeIsVectorType = int;
    static PixelType pixelInstance( unsigned numberOfComponents ) { return PixelType {numberOfComponents}; }
    };
  // For the case that the number of colors is implicitly set to 1 at compile time:
  template< typename TPixelType >
  struct PixelHelper< TPixelType, myVoid_t< typename std::enable_if< std::is_arithmetic< TPixelType >::value >::type > >
    {
    using PixelType = TPixelType;
    using ValueType = PixelType;
    static constexpr SizeValueType NumberOfComponents = 1;
    using PixelTypeIsScalarType = int;
    static PixelType pixelInstance( unsigned numberOfComponents ) { return PixelType {}; }
    };
  // For the case that the pixel type is RGBPixel:
  template< typename TScalar>
  struct PixelHelper< RGBPixel< TScalar >, void >
    {
    using PixelType = RGBPixel< TScalar >;
    using ValueType = typename PixelType::ValueType;
    static constexpr SizeValueType NumberOfComponents = 3;
    using PixelTypeIsVectorType = int;
    static PixelType pixelInstance( unsigned numberOfComponents ) { return PixelType {}; }
    };
  // For the case that the pixel type is RGBAPixel:
  template< typename TScalar>
  struct PixelHelper< RGBAPixel< TScalar >, void >
    {
    using PixelType = RGBAPixel< TScalar >;
    using ValueType = typename PixelType::ValueType;
    static constexpr SizeValueType NumberOfComponents = 4;
    using PixelTypeIsVectorType = int;
    static PixelType pixelInstance( unsigned numberOfComponents ) { return PixelType {}; }
    };
  // For the cases that the pixel type is Vector or CovariantVector:
  template< typename TScalar, unsigned int NVectorComponent >
  struct PixelHelper< Vector< TScalar, NVectorComponent >, void >
    {
    using PixelType = Vector< TScalar, NVectorComponent >;
    using ValueType = typename PixelType::ValueType;
    static constexpr SizeValueType NumberOfComponents = NVectorComponent;
    using PixelTypeIsVectorType = int;
    static PixelType pixelInstance( unsigned numberOfComponents ) { return PixelType {}; }
    };
  template< typename TScalar, unsigned int NVectorComponent >
  struct PixelHelper< CovariantVector< TScalar, NVectorComponent >, void >
    {
    using PixelType = CovariantVector< TScalar, NVectorComponent >;
    using ValueType = typename PixelType::ValueType;
    static constexpr SizeValueType NumberOfComponents = NVectorComponent;
    using PixelTypeIsVectorType = int;
    static PixelType pixelInstance( unsigned numberOfComponents ) { return PixelType {}; }
    };

  using ValueType = typename PixelHelper< PixelType >::ValueType;
  static constexpr unsigned int ImageDimension = VImageDimension;
  static constexpr auto pixelInstance = PixelHelper< PixelType >::pixelInstance;

  // A pixel value may be a basic type (integer, real), or a class (RGB, RGBA, Vector, CovariantVector) whose value
  // types are, recursively pixel types.  DimensionHelper can handle all cases that PixelHelper knows about.
  // For the case that the TPixelType is not known by PixelHelper:
  template< typename TPixelType, typename = void >
  struct DimensionHelper
    {
    };
  // For the case that the TPixelType is known by PixelHelper to be a scalar type:
  template< typename TPixelType >
  struct DimensionHelper< TPixelType, myVoid_t< typename PixelHelper< TPixelType >::PixelTypeIsScalarType > >
    {
    using PixelType = TPixelType;
    using DeepScalarType = PixelType;
    static constexpr unsigned int TorchDimension = 0;
    static void AppendSizes( std::vector< int64_t > &size ) {/* Nothing to append */}
    };
  // For the case that the TPixelType is known by PixelHelper to be a vector type:
  template< typename TPixelType >
  struct DimensionHelper< TPixelType, myVoid_t< typename PixelHelper< TPixelType >::PixelTypeIsVectorType > >
    {
    using PixelType = TPixelType;
    using DeepScalarType = typename DimensionHelper< typename PixelHelper< PixelType >::ValueType >::DeepScalarType;
    static constexpr unsigned int TorchDimension = 1 + DimensionHelper< typename PixelHelper< PixelType >::ValueType >::TorchDimension;
    static void AppendSizes( std::vector< int64_t > &size )
      {
      size.push_back( PixelHelper< PixelType >::NumberOfComponents );
      // Recurse
      DimensionHelper< typename PixelHelper< PixelType >::ValueType >::AppendSizes( size );
      }
    };
  using DeepScalarType = typename DimensionHelper< PixelType >::DeepScalarType;
  static constexpr at::ScalarType TorchValueType = c10::impl::CPPTypeToScalarType< DeepScalarType >::value;
  static constexpr unsigned int TorchDimension = ImageDimension + DimensionHelper< PixelType >::TorchDimension;

  /** Allocate CPU and GPU memory space */
  virtual void Allocate( bool initialize = false ) override;

  virtual void AllocateGPU();

  virtual void Initialize() override;

  void FillBuffer( const TPixel &value );

  void SetPixel( const IndexType &index, const TPixel &value );

  const TPixel &GetPixel( const IndexType &index ) const;

  TPixel &GetPixel( const IndexType &index );

  const TPixel &operator[]( const IndexType &index ) const;

  TPixel &operator[]( const IndexType &index );

  /** Explicit synchronize CPU/GPU buffers */
  virtual void UpdateBuffers();

  /** Explicit synchronize CPU/GPU buffers */
  virtual void UpdateCPUBuffer();

  virtual void UpdateGPUBuffer();

  /** Get CPU buffer pointer */
  virtual TPixel *GetBufferPointer() override;

  virtual const TPixel *GetBufferPointer() const override;

  /** Return the Pixel Accessor object */
  AccessorType GetPixelAccessor()
  {
    m_DataManager->SetGPUBufferStale();
    return Superclass::GetPixelAccessor();
  }


  /** Return the Pixel Accesor object */
  const AccessorType GetPixelAccessor() const
  {
    m_DataManager->UpdateCPUBuffer();
    return Superclass::GetPixelAccessor();
  }


  /** Return the NeighborhoodAccessor functor */
  NeighborhoodAccessorFunctorType GetNeighborhoodAccessor()
  {
    m_DataManager->SetGPUBufferStale();
    return NeighborhoodAccessorFunctorType();
  }


  /** Return the NeighborhoodAccessor functor */
  const NeighborhoodAccessorFunctorType GetNeighborhoodAccessor() const
  {
    m_DataManager->UpdateCPUBuffer();
    return NeighborhoodAccessorFunctorType();
  }


  void SetPixelContainer( PixelContainer *container );

  /** Return a pointer to the container. */
  PixelContainer *GetPixelContainer()
  {
    m_DataManager->SetGPUBufferStale();
    return Superclass::GetPixelContainer();
  }


  const PixelContainer *GetPixelContainer() const
  {
    m_DataManager->UpdateCPUBuffer();
    return Superclass::GetPixelContainer();
  }


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

    if( m_DataManager->IsCPUBufferStale() )
    {
      m_DataManager->Modified();
    }

  }


  /** Graft the data and information from one TorchImage to another. */
  virtual void Graft( const DataObject *data ) override;

  virtual void GraftITKImage( const DataObject *data );

  /** Whenever the image has been modified, set the GPU Buffer to stale */
  virtual void Modified() const;

  /** Get matrices intended to help with the conversion of Index coordinates
   *  to PhysicalPoint coordinates */
  // Should these be in the Superclass only?!!!
  itkGetConstReferenceMacro( IndexToPhysicalPoint, DirectionType );
  itkGetConstReferenceMacro( PhysicalPointToIndex, DirectionType );

protected:

  TorchImage();
  virtual ~TorchImage() {}

  virtual void PrintSelf( std::ostream &os, Indent indent ) const override;

private:

  typename TorchImageDataManager< TorchImage >::Pointer m_DataManager;

  bool m_Graft;
};

//------------------------------------------------------------------------------
template< typename T >
class ITK_TEMPLATE_EXPORT TorchTraits
{
public:
  using Type = T;
};

template< typename TPixelType, unsigned int NDimension >
class ITK_TEMPLATE_EXPORT TorchTraits< Image< TPixelType, NDimension > >
{
public:
  using Type = TorchImage< TPixelType, NDimension >;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkTorchImage.hxx"
#endif

#endif

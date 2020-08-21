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
#ifndef itkTorchPixelHelper_h
#define itkTorchPixelHelper_h

namespace itk
{
// We define our own void_t so that we do not have to require C++-17;
// we do it in a way that works even for C++14 peculiarities.
template< typename... Ts >
struct TorchMakeVoid { using type = void; };
template< typename... Ts >
using TorchVoid_t = typename TorchMakeVoid< Ts... >::type;
} // end anonymous namespace

#include <torch/torch.h>

namespace itk
{
/** \class TorchPixelHelper
 *  \brief Converts between ITK vector pixel types and torch scalar pixel types.
 *
 * ITK supports pixels of type RGBPixel, Vector, CovaraiantVector,
 * etc., which are "vectors" of an underlying scalar type such as
 * unsigned char, int, and double.  Torch supports only scalars.
 * TorchPixelHelper is a class to help with this difference.
 *
 * The GetPixel and operator[] methods for a non-Torch Image provide
 * rvalues and lvalues for accessing pixel data.  However, especially
 * when the pixel data of a TorchImage reside in GPU memory, an
 * ordinary C++ reference will not work as an lvalue.  This class
 * allows templated code that is syntactically correct for the
 * ordinary Image case to also work with the TorchImage case.
 *
 * This is the non-specialization of TorchPixelHelper, which is not
 * used.
 *
 * \ingroup PyTorch
 */
template< typename TPixelType, typename = void >
class TorchPixelHelper
{
public:
  using Self = TorchPixelHelper;
  using PixelType = TPixelType;
  static constexpr unsigned int NumberOfComponents = 0;
  static constexpr unsigned int SizeOf = 0;
  static constexpr unsigned int PixelDimension = 0;
};

/** \class TorchPixelHelper
 *  \brief Converts between ITK vector pixel types and torch scalar pixel types.
 *
 * ITK supports pixels of type RGBPixel, Vector, CovaraiantVector,
 * etc., which are "vectors" of an underlying scalar type such as
 * unsigned char, int, and double.  Torch supports only scalars.
 * TorchPixelHelper is a class to help with this difference.
 *
 * The GetPixel and operator[] methods for a non-Torch Image provide
 * rvalues and lvalues for accessing pixel data.  However, especially
 * when the pixel data of a TorchImage reside in GPU memory, an
 * ordinary C++ reference will not work as an lvalue.  This class
 * allows templated code that is syntactically correct for the
 * ordinary Image case to also work with the TorchImage case.
 *
 * This is the specialization of TorchPixelHelper for pixel types that
 * are already scalars.
 *
 * \ingroup PyTorch
 */
template< typename TPixelType >
class TorchPixelHelper< TPixelType, TorchVoid_t< typename std::enable_if< std::is_arithmetic< TPixelType >::value >::type > >
{
public:
  using Self = TorchPixelHelper;
  using PixelType = TPixelType;
  using DeepScalarType = PixelType;
  static constexpr unsigned int NumberOfComponents = 1;
  static constexpr unsigned int SizeOf = NumberOfComponents;
  static constexpr unsigned int PixelDimension = 0; // a zero-dimensional array

  TorchPixelHelper &operator=( const PixelType &value )
    {
    m_Tensor.index( m_TorchIndex ).fill_( value );
    return *this;
    }

  operator PixelType() const
    {
    return m_Tensor.index( m_TorchIndex ).item< DeepScalarType >();
    }

protected:
  template< typename NTPixelType, typename Void > friend class TorchPixelHelper;
  template< typename NTPixelType, unsigned int NVImageDimension > friend class TorchImage;

  static void AppendSizes( std::vector< int64_t > & itkNotUsed( size ) )
    {
    // Nothing to append
    }

  TorchPixelHelper( torch::Tensor Tensor, std::vector< at::indexing::TensorIndex > &TorchIndex ) : m_Tensor( Tensor ), m_TorchIndex( TorchIndex )
    {
    }

  torch::Tensor m_Tensor;
  std::vector< at::indexing::TensorIndex > m_TorchIndex;
};

/** \class TorchPixelHelper
 *  \brief Converts between ITK vector pixel types and torch scalar pixel types.
 *
 * ITK supports pixels of type RGBPixel, Vector, CovaraiantVector,
 * etc., which are "vectors" of an underlying scalar type such as
 * unsigned char, int, and double.  Torch supports only scalars.
 * TorchPixelHelper is a class to help with this difference.
 *
 * The GetPixel and operator[] methods for a non-Torch Image provide
 * rvalues and lvalues for accessing pixel data.  However, especially
 * when the pixel data of a TorchImage reside in GPU memory, an
 * ordinary C++ reference will not work as an lvalue.  This class
 * allows templated code that is syntactically correct for the
 * ordinary Image case to also work with the TorchImage case.
 *
 * This is the specialization of TorchPixelHelper for pixel types that
 * are known "vector" types.
 *
 * \ingroup PyTorch
 */
template< typename TPixelType >
class TorchPixelHelper< TPixelType, TorchVoid_t< typename std::enable_if< !std::is_arithmetic< TPixelType >::value >::type > >
{
public:
  using Self = TorchPixelHelper;
  using PixelType = TPixelType;
  using ValueType = typename PixelType::ValueType;
  using NextTorchPixelHelper = TorchPixelHelper< ValueType >;
  using DeepScalarType = typename NextTorchPixelHelper::DeepScalarType;
  static constexpr unsigned int NumberOfComponents = PixelType::Dimension;
  static constexpr unsigned int SizeOf = NumberOfComponents * NextTorchPixelHelper::SizeOf;
  static constexpr unsigned int PixelDimension = 1 + NextTorchPixelHelper::PixelDimension;

  TorchPixelHelper &operator=( const PixelType &value )
    {
    for( unsigned int i = 0; i < Self::NumberOfComponents; ++i )
      {
      m_TorchIndex.push_back( static_cast< int64_t >( i ) );
      NextTorchPixelHelper { m_Tensor, m_TorchIndex } = value[i];
      m_TorchIndex.pop_back();
      }
    return *this;
    }

  operator PixelType() const
    {
    PixelType response;
    for( unsigned int i = 0; i < Self::NumberOfComponents; ++i )
      {
      m_TorchIndex.push_back( static_cast< int64_t >( i ) );
      response[i] = NextTorchPixelHelper { m_Tensor, m_TorchIndex };
      m_TorchIndex.pop_back();
      }
    return response;
    }

protected:
  template< typename NTPixelType, typename Void > friend class TorchPixelHelper;
  template< typename NTPixelType, unsigned int NVImageDimension > friend class TorchImage;

  static void AppendSizes( std::vector< int64_t > &size )
    {
    size.push_back( PixelType::Dimension );
    NextTorchPixelHelper::AppendSizes( size );
    }

  TorchPixelHelper( torch::Tensor Tensor, std::vector< at::indexing::TensorIndex > &TorchIndex ) : m_Tensor( Tensor ), m_TorchIndex( TorchIndex )
    {
    }

  torch::Tensor m_Tensor;
  mutable std::vector< at::indexing::TensorIndex > m_TorchIndex;
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkTorchPixelHelper.hxx"
#endif

#endif

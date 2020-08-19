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
#ifndef itkUniformTorchTensorAccessorHelper_h
#define itkUniformTorchTensorAccessorHelper_h

#include <torch/torch.h>
#include "itkImageBase.h"
#include "itkUniformTorchTensorAccessor.h"
#include "itkTorchPixelHelper.h"

namespace itk
{
/** \class UniformTorchTensorAccessorHelper
 *  \brief Efficiently uses underlying torch::TensorAccessor object.
 *
 * Access to pixel information via an `IndexType index` is achieved
 * with an underlying torch::TensorAccessor via multiple applications
 * of operator[].  First the last component of `index` is used, which
 * is the slowest varying component in the underlying pixel memory.
 * Then the second-to-last component, etc.  This class automates that
 * repeated application of the operator[].

 * This is the non-specialization, which can be used for an
 * arbitrarily large number of dimensions in `IndexType index`.
 *
 * \ingroup PyTorch
 */
template< typename PixelType, typename IndexType, typename SizeType, int VCurrentAccessorLevel, int VNumberOfSteps >
class UniformTorchTensorAccessorHelper
{
public:
  using DeepScalarType = typename TorchPixelHelper< PixelType >::DeepScalarType;
  /** The `index` variable has the dimension that varies fastest in
   * the underlying buffer at index[0], but `accessor` uses indices
   * starting with the slowest varying dimension first.
   */
  static typename UniformTorchTensorAccessor< DeepScalarType, VCurrentAccessorLevel-VNumberOfSteps >::type
  FindPixel( typename UniformTorchTensorAccessor< DeepScalarType, VCurrentAccessorLevel >::type accessor, const IndexType &index )
    {
    return UniformTorchTensorAccessorHelper< DeepScalarType, IndexType, SizeType, VCurrentAccessorLevel-1, VNumberOfSteps-1 >::FindPixel( accessor[index[VNumberOfSteps-1]], index );
    }
  /** Set every part of the tensor accessible from `accessor` to the
   * pixel value `value` */
  static void
  SetAllPixels( typename UniformTorchTensorAccessor< DeepScalarType, VCurrentAccessorLevel >::type accessor, const SizeType &bufferSize, const PixelType &value )
  {
  for( SizeValueType i = 0; i < bufferSize[VNumberOfSteps-1]; ++i )
    {
    typename UniformTorchTensorAccessor< DeepScalarType, VCurrentAccessorLevel-1 >::type nextAccessor = accessor[i];
    UniformTorchTensorAccessorHelper< PixelType, IndexType, SizeType, VCurrentAccessorLevel-1, VNumberOfSteps-1 >::SetAllPixels( nextAccessor, bufferSize, value );
    }
  }
};

/** \class UniformTorchTensorAccessorHelper
 *  \brief Efficiently uses underlying torch::TensorAccessor object.
 *
 * Access to pixel information via an `IndexType index` is achieved
 * with an underlying torch::TensorAccessor via multiple applications
 * of operator[].  First the last component of `index` is used, which
 * is the slowest varying component in the underlying pixel memory.
 * Then the second-to-last component, etc.  This class automates that
 * repeated application of the operator[].

 * This is the specialization for which 0 additional applications of
 * operator[] are required.
 *
 * \ingroup PyTorch
 */
template< typename PixelType, typename IndexType, typename SizeType, int VCurrentAccessorLevel >
class UniformTorchTensorAccessorHelper< PixelType, IndexType, SizeType, VCurrentAccessorLevel, 0 >
{
public:
  using DeepScalarType = typename TorchPixelHelper< PixelType >::DeepScalarType;
  static typename UniformTorchTensorAccessor< DeepScalarType, VCurrentAccessorLevel >::type
  FindPixel( typename UniformTorchTensorAccessor< DeepScalarType, VCurrentAccessorLevel >::type accessor, const IndexType & itkNotUsed( index ) )
    {
    return accessor;
    }
  static void
  SetAllPixels( typename UniformTorchTensorAccessor< DeepScalarType, VCurrentAccessorLevel >::type accessor, const SizeType &bufferSize, const PixelType &value )
  {
    TorchPixelHelper< PixelType > {accessor} = value;
  }
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkTorchImage.hxx"
#endif

#endif

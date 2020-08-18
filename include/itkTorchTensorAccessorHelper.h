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
#ifndef itkTorchTensorAccessorHelper_h
#define itkTorchTensorAccessorHelper_h

#include <torch/torch.h>
#include "itkImageBase.h"
#include "itkTorchPixelHelper.h"

namespace itk
{
/** \class TorchTensorAccessorHelper
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
template< typename DeepScalarType, typename IndexType, int VCurrentAccessorLevel, int VNumberOfSteps >
class TorchTensorAccessorHelper
{
public:
  static torch::TensorAccessor< DeepScalarType, VCurrentAccessorLevel-VNumberOfSteps > &
  FindPixel( torch::TensorAccessor< DeepScalarType, VCurrentAccessorLevel > &accessor, const IndexType &index )
    {
    // The `index` variable has the dimension that varies fastest in
    // the underlying buffer at index[0], but `accessor` uses
    // indices starting with the slowest varying dimension first.
    return TorchTensorAccessorHelper< DeepScalarType, IndexType, VCurrentAccessorLevel-1, VNumberOfSteps-1 >::FindPixel( accessor[index[VNumberOfSteps-1]], index );
    }
};

/** \class TorchTensorAccessorHelper
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
template< typename DeepScalarType, typename IndexType, int VCurrentAccessorLevel >
class TorchTensorAccessorHelper< DeepScalarType, IndexType, VCurrentAccessorLevel, 0 >
{
public:
  static torch::TensorAccessor< DeepScalarType, VCurrentAccessorLevel > &
  FindPixel( torch::TensorAccessor< DeepScalarType, VCurrentAccessorLevel > &accessor, const IndexType & /* index */ )
    {
    return accessor;
    }
};

/** Specializations for VNumberOfSteps > 0 are not strictly necessary
 * but are implemented for speed. */

/** \class TorchTensorAccessorHelper
 *  \brief Efficiently uses underlying torch::TensorAccessor object.
 *
 * Access to pixel information via an `IndexType index` is achieved
 * with an underlying torch::TensorAccessor via multiple applications
 * of operator[].  First the last component of `index` is used, which
 * is the slowest varying component in the underlying pixel memory.
 * Then the second-to-last component, etc.  This class automates that
 * repeated application of the operator[].

 * This is the specialization for which 1 additional applications of
 * operator[] are required.
 *
 * \ingroup PyTorch
 */
template< typename DeepScalarType, typename IndexType, int VCurrentAccessorLevel >
class TorchTensorAccessorHelper< DeepScalarType, IndexType, VCurrentAccessorLevel, 1 >
{
public:
  static torch::TensorAccessor< DeepScalarType, VCurrentAccessorLevel-1 > &
  FindPixel( torch::TensorAccessor< DeepScalarType, VCurrentAccessorLevel > &accessor, const IndexType &index )
    {
    return accessor[index[0]];
    }
};

/** \class TorchTensorAccessorHelper
 *  \brief Efficiently uses underlying torch::TensorAccessor object.
 *
 * Access to pixel information via an `IndexType index` is achieved
 * with an underlying torch::TensorAccessor via multiple applications
 * of operator[].  First the last component of `index` is used, which
 * is the slowest varying component in the underlying pixel memory.
 * Then the second-to-last component, etc.  This class automates that
 * repeated application of the operator[].

 * This is the specialization for which 2 additional applications of
 * operator[] are required.
 *
 * \ingroup PyTorch
 */
template< typename DeepScalarType, typename IndexType, int VCurrentAccessorLevel >
class TorchTensorAccessorHelper< DeepScalarType, IndexType, VCurrentAccessorLevel, 2 >
{
public:
  static torch::TensorAccessor< DeepScalarType, VCurrentAccessorLevel-2 > &
  FindPixel( torch::TensorAccessor< DeepScalarType, VCurrentAccessorLevel > &accessor, const IndexType &index )
    {
    return accessor[index[1]][index[0]];
    }
};

/** \class TorchTensorAccessorHelper
 *  \brief Efficiently uses underlying torch::TensorAccessor object.
 *
 * Access to pixel information via an `IndexType index` is achieved
 * with an underlying torch::TensorAccessor via multiple applications
 * of operator[].  First the last component of `index` is used, which
 * is the slowest varying component in the underlying pixel memory.
 * Then the second-to-last component, etc.  This class automates that
 * repeated application of the operator[].

 * This is the specialization for which 3 additional applications of
 * operator[] are required.
 *
 * \ingroup PyTorch
 */
template< typename DeepScalarType, typename IndexType, int VCurrentAccessorLevel >
class TorchTensorAccessorHelper< DeepScalarType, IndexType, VCurrentAccessorLevel, 3 >
{
public:
  static torch::TensorAccessor< DeepScalarType, VCurrentAccessorLevel-3 > &
  FindPixel( torch::TensorAccessor< DeepScalarType, VCurrentAccessorLevel > &accessor, const IndexType &index )
    {
    return accessor[index[2]][index[1]][index[0]];
    }
};

/** \class TorchTensorAccessorHelper
 *  \brief Efficiently uses underlying torch::TensorAccessor object.
 *
 * Access to pixel information via an `IndexType index` is achieved
 * with an underlying torch::TensorAccessor via multiple applications
 * of operator[].  First the last component of `index` is used, which
 * is the slowest varying component in the underlying pixel memory.
 * Then the second-to-last component, etc.  This class automates that
 * repeated application of the operator[].

 * This is the specialization for which 4 additional applications of
 * operator[] are required.
 *
 * \ingroup PyTorch
 */
template< typename DeepScalarType, typename IndexType, int VCurrentAccessorLevel >
class TorchTensorAccessorHelper< DeepScalarType, IndexType, VCurrentAccessorLevel, 4 >
{
public:
  static torch::TensorAccessor< DeepScalarType, VCurrentAccessorLevel-4 > &
  FindPixel( torch::TensorAccessor< DeepScalarType, VCurrentAccessorLevel > &accessor, const IndexType &index )
    {
    return accessor[index[3]][index[2]][index[1]][index[0]];
    }
};

/** \class TorchTensorAccessorHelper
 *  \brief Efficiently uses underlying torch::TensorAccessor object.
 *
 * Access to pixel information via an `IndexType index` is achieved
 * with an underlying torch::TensorAccessor via multiple applications
 * of operator[].  First the last component of `index` is used, which
 * is the slowest varying component in the underlying pixel memory.
 * Then the second-to-last component, etc.  This class automates that
 * repeated application of the operator[].

 * This is the specialization for which 5 additional applications of
 * operator[] are required.
 *
 * \ingroup PyTorch
 */
template< typename DeepScalarType, typename IndexType, int VCurrentAccessorLevel >
class TorchTensorAccessorHelper< DeepScalarType, IndexType, VCurrentAccessorLevel, 5 >
{
public:
  static torch::TensorAccessor< DeepScalarType, VCurrentAccessorLevel-5 > &
  FindPixel( torch::TensorAccessor< DeepScalarType, VCurrentAccessorLevel > &accessor, const IndexType &index )
    {
    return accessor[index[4]][index[3]][index[2]][index[1]][index[0]];
    }
};

/** \class TorchTensorAccessorHelper
 *  \brief Efficiently uses underlying torch::TensorAccessor object.
 *
 * Access to pixel information via an `IndexType index` is achieved
 * with an underlying torch::TensorAccessor via multiple applications
 * of operator[].  First the last component of `index` is used, which
 * is the slowest varying component in the underlying pixel memory.
 * Then the second-to-last component, etc.  This class automates that
 * repeated application of the operator[].

 * This is the specialization for which 6 additional applications of
 * operator[] are required.
 *
 * \ingroup PyTorch
 */
template< typename DeepScalarType, typename IndexType, int VCurrentAccessorLevel >
class TorchTensorAccessorHelper< DeepScalarType, IndexType, VCurrentAccessorLevel, 6 >
{
public:
  static torch::TensorAccessor< DeepScalarType, VCurrentAccessorLevel-6 > &
  FindPixel( torch::TensorAccessor< DeepScalarType, VCurrentAccessorLevel > &accessor, const IndexType &index )
    {
    return accessor[index[5]][index[4]][index[3]][index[2]][index[1]][index[0]];
    }
};

/** \class TorchTensorAccessorHelper
 *  \brief Efficiently uses underlying torch::TensorAccessor object.
 *
 * Access to pixel information via an `IndexType index` is achieved
 * with an underlying torch::TensorAccessor via multiple applications
 * of operator[].  First the last component of `index` is used, which
 * is the slowest varying component in the underlying pixel memory.
 * Then the second-to-last component, etc.  This class automates that
 * repeated application of the operator[].

 * This is the specialization for which 7 additional applications of
 * operator[] are required.
 *
 * \ingroup PyTorch
 */
template< typename DeepScalarType, typename IndexType, int VCurrentAccessorLevel >
class TorchTensorAccessorHelper< DeepScalarType, IndexType, VCurrentAccessorLevel, 7 >
{
public:
  static torch::TensorAccessor< DeepScalarType, VCurrentAccessorLevel-7 > &
  FindPixel( torch::TensorAccessor< DeepScalarType, VCurrentAccessorLevel > &accessor, const IndexType &index )
    {
    return accessor[index[6]][index[5]][index[4]][index[3]][index[2]][index[1]][index[0]];
    }
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkTorchImage.hxx"
#endif

#endif

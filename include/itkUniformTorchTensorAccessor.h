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
#ifndef itkUniformTorchTensorAccessor_h
#define itkUniformTorchTensorAccessor_h

#include <torch/torch.h>

namespace itk
{
/** \class UniformTorchTensorAccessor
 *  \brief Makes the torch::TensorAccessor class more uniform
 *
 * torch functions that would return `torch::TensorAccessor< PixelType, 0 >` instead
 * return `PixelType` (or, likewise, as references) so we create
 * itk::UniformTorchTensorAccessor that understands this pattern.
 *
 * \ingroup PyTorch
 */
template< typename DeepScalarType, int VCurrentAccessorLevel >
class UniformTorchTensorAccessor
{
public:
  using type = torch::TensorAccessor< DeepScalarType, VCurrentAccessorLevel >;
};

template< typename DeepScalarType >
class UniformTorchTensorAccessor< DeepScalarType, 0 >
{
public:
  using type = DeepScalarType &;
};

} // end namespace itk

#endif

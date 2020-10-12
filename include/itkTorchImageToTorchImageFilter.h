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
#ifndef itkTorchImageToTorchImageFilter_h
#define itkTorchImageToTorchImageFilter_h

#include <type_traits>
#include "itkTorchImage.h"
#include "itkImageToImageFilter.h"

namespace itk
{
/** \class TorchImageToTorchImageFilter
 *
 * \brief Base class for filters that take a TorchImage as input and
 * produce a TorchImage as output.
 *
 * This is simply the ImageToImageFilter base class but with its input
 * and output images restricted to TorchImage< TPixel, VImageDimension >
 * types.
 *
 * \ingroup PyTorch
 */

template< typename TInputTorchImage, typename TOutputTorchImage >
class ITK_TEMPLATE_EXPORT TorchImageToTorchImageFilter
  : public ImageToImageFilter< TInputTorchImage, TOutputTorchImage >
{
public:
  static constexpr bool ConceptCheck =
    std::is_same< TInputTorchImage, TorchImage< typename TInputTorchImage::PixelType, TInputTorchImage::ImageDimension > >::value &&
      std::is_same< TOutputTorchImage, TorchImage< typename TOutputTorchImage::PixelType, TOutputTorchImage::ImageDimension > >::value;
  static_assert( ConceptCheck, "itk::TorchImageToTorchImageFilter: each template parameter must be an itk::TorchImage." );

  ITK_DISALLOW_COPY_AND_ASSIGN( TorchImageToTorchImageFilter );

  /** Standard class type aliases. */
  using Self = TorchImageToTorchImageFilter;
  using Superclass = ImageToImageFilter< TInputTorchImage, TOutputTorchImage >;
  using Pointer = SmartPointer< Self >;
  using ConstPointer = SmartPointer< const Self >;

  /** Some convenient type aliases. */
  using InputTorchImageType = TInputTorchImage;
  using InputTorchImagePointer = typename Self::InputTorchImageType::Pointer;
  using InputTorchImageConstPointer = typename Self::InputTorchImageType::ConstPointer;
  using InputTorchImageRegionType = typename Self::InputTorchImageType::RegionType;
  using InputTorchImagePixelType = typename Self::InputTorchImageType::PixelType;
  using OutputTorchImageType = TOutputTorchImage;
  using OutputTorchImagePointer = typename Self::OutputTorchImageType::Pointer;
  using OutputTorchImageConstPointer = typename Self::OutputTorchImageType::ConstPointer;
  using OutputTorchImageRegionType = typename Self::OutputTorchImageType::RegionType;
  using OutputTorchImagePixelType = typename Self::OutputTorchImageType::PixelType;

  /** ImageDimension constants */
  static constexpr unsigned int InputTorchImageDimension = Self::TInputTorchImage::ImageDimension;
  static constexpr unsigned int OutputTorchImageDimension = Self::TOutputTorchImage::ImageDimension;

  /** Method for creation through the object factory. */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( TorchImageToTorchImageFilter, ImageToImageFilter );

protected:
  TorchImageToTorchImageFilter() = default;
  ~TorchImageToTorchImageFilter() override = default;
};


} // end namespace itk

#endif

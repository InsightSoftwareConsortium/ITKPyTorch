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
#ifndef itkTorchImageToImageFilter_h
#define itkTorchImageToImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

#include "itkTorchImage.h"

namespace itk
{

/** \class TorchImageToImageFilter
 *
 * \brief Converts an itk::TorchImage to an itk::Image.
 *
 * Conversions between TorchImage and Image permit pipelines with
 * steps for either kind of image.
 *
 * \ingroup PyTorch
 *
 */
template< typename TInputTorchImage >
class ITK_TEMPLATE_EXPORT TorchImageToImageFilter
  : public ImageToImageFilter< TInputTorchImage, Image< typename TInputTorchImage::PixelType, TInputTorchImage::ImageDimension > >
{
public:
#ifdef ITK_USE_CONCEPT_CHECKING
  static constexpr bool ConceptCheck =
    std::is_same< TInputTorchImage, TorchImage< typename TInputTorchImage::PixelType, TInputTorchImage::ImageDimension > >::value;
  static_assert( ConceptCheck, "itk::TorchImageToImageFilter template parameter must be an itk::TorchImage." );
#endif

  ITK_DISALLOW_COPY_AND_ASSIGN( TorchImageToImageFilter );

  using TOutputImage = Image< typename TInputTorchImage::PixelType, TInputTorchImage::ImageDimension >;
  static constexpr unsigned int InputTorchImageDimension = TInputTorchImage::ImageDimension;
  static constexpr unsigned int OutputImageDimension = TOutputImage::ImageDimension;

  /** Standard class typedefs. */
  using Self = TorchImageToImageFilter< TInputTorchImage >;
  using Superclass = ImageToImageFilter< TInputTorchImage, TOutputImage >;
  using Pointer = SmartPointer< Self >;
  using ConstPointer = SmartPointer< const Self >;

  /** Some convenient type aliases. */
  using InputTorchImageType = TInputTorchImage;
  using InputTorchImagePointer = typename Self::InputTorchImageType::Pointer;
  using InputTorchImageConstPointer = typename Self::InputTorchImageType::ConstPointer;
  using InputTorchImageRegionType = typename Self::InputTorchImageType::RegionType;
  using InputTorchImageRegionIterator = ImageRegionIterator< InputTorchImageType >;
  using InputTorchImageRegionConstIterator = ImageRegionConstIterator< InputTorchImageType >;
  using InputTorchImagePixelType = typename Self::InputTorchImageType::PixelType;
  using InputTorchImageSizeType = Size< InputTorchImageDimension >;

  using OutputImageType = TOutputImage;
  using OutputImagePointer = typename Self::OutputImageType::Pointer;
  using OutputImageConstPointer = typename Self::OutputImageType::ConstPointer;
  using OutputImageRegionType = typename Self::OutputImageType::RegionType;
  using OutputImageRegionIterator = ImageRegionIterator< OutputImageType >;
  using OutputImageRegionConstIterator = ImageRegionConstIterator< OutputImageType >;
  using OutputImagePixelType = typename Self::OutputImageType::PixelType;
  using OutputImageSizeType = Size< OutputImageDimension >;

  /** Run-time type information. */
  itkTypeMacro( TorchImageToImageFilter, ImageToImageFilter );

  /** Standard New macro. */
  itkNewMacro( Self );

protected:
  TorchImageToImageFilter() = default;
  ~TorchImageToImageFilter() override = default;

  void PrintSelf( std::ostream &os, Indent indent ) ITKv5_CONST override;

  void VerifyPreconditions() ITKv5_CONST override;

  void GenerateData() override;

private:
};
} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkTorchImageToImageFilter.hxx"
#endif

#endif // itkTorchImageToImageFilter

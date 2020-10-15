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
#ifndef itkImageToTorchImageFilter_h
#define itkImageToTorchImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkImageRegionIterator.h"
#include "itkImageRegionConstIterator.h"

#include "itkTorchImage.h"

namespace itk
{

/** \class ImageToTorchImageFilter
 *
 * \brief Converts an itk::Image to an itk::TorchImage.
 *
 * Conversions between Image and TorchImage permit pipelines with
 * steps for either kind of image.
 *
 * \ingroup PyTorch
 *
 */
template< typename TInputImage >
class ITK_TEMPLATE_EXPORT ImageToTorchImageFilter
  : public ImageToImageFilter< TInputImage, TorchImage< typename TInputImage::PixelType, TInputImage::ImageDimension > >
{
public:
#ifdef ITK_USE_CONCEPT_CHECKING
  static constexpr bool ConceptCheck =
    !std::is_same< TInputImage, TorchImage< typename TInputImage::PixelType, TInputImage::ImageDimension > >::value;
  static_assert( ConceptCheck, "itk::ImageToTorchImageFilter template parameter must be an itk::Image." );
#endif

  ITK_DISALLOW_COPY_AND_ASSIGN( ImageToTorchImageFilter );

  using TOutputTorchImage = TorchImage< typename TInputImage::PixelType, TInputImage::ImageDimension >;
  static constexpr unsigned int InputImageDimension = TInputImage::ImageDimension;
  static constexpr unsigned int OutputTorchImageDimension = TOutputTorchImage::ImageDimension;

  /** Standard class typedefs. */
  using Self = ImageToTorchImageFilter< TInputImage >;
  using Superclass = ImageToImageFilter< TInputImage, TOutputTorchImage >;
  using Pointer = SmartPointer< Self >;
  using ConstPointer = SmartPointer< const Self >;

  /** Some convenient type aliases. */
  using InputImageType = TInputImage;
  using InputImagePointer = typename Self::InputImageType::Pointer;
  using InputImageConstPointer = typename Self::InputImageType::ConstPointer;
  using InputImageRegionType = typename Self::InputImageType::RegionType;
  using InputImageRegionIterator = ImageRegionIterator< InputImageType >;
  using InputImageRegionConstIterator = ImageRegionConstIterator< InputImageType >;
  using InputImagePixelType = typename Self::InputImageType::PixelType;
  using InputImageSizeType = Size< InputImageDimension >;

  using OutputTorchImageType = TOutputTorchImage;
  using OutputTorchImagePointer = typename Self::OutputTorchImageType::Pointer;
  using OutputTorchImageConstPointer = typename Self::OutputTorchImageType::ConstPointer;
  using OutputTorchImageRegionType = typename Self::OutputTorchImageType::RegionType;
  using OutputTorchImageRegionIterator = ImageRegionIterator< OutputTorchImageType >;
  using OutputTorchImageRegionConstIterator = ImageRegionConstIterator< OutputTorchImageType >;
  using OutputTorchImagePixelType = typename Self::OutputTorchImageType::PixelType;
  using OutputTorchImageSizeType = Size< OutputTorchImageDimension >;

  /** Run-time type information. */
  itkTypeMacro( ImageToTorchImageFilter, ImageToImageFilter );

  /** Standard New macro. */
  itkNewMacro( Self );

protected:
  ImageToTorchImageFilter() = default;
  ~ImageToTorchImageFilter() override = default;

  void PrintSelf( std::ostream &os, Indent indent ) ITKv5_CONST override;

  void VerifyPreconditions() ITKv5_CONST override;

  void GenerateData() override;

private:
};
} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkImageToTorchImageFilter.hxx"
#endif

#endif // itkImageToTorchImageFilter

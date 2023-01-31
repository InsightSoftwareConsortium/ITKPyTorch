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
#include "itkTorchImage.h"
#include <type_traits>

namespace itk {
/** \class TorchImageToImageFilter
 *
 * \brief class to abstract the behaviour of the Torch filters.
 *
 * TorchImageToImageFilter is the Torch version of ImageToImageFilter.
 * This class can accept both CPU and Torch image as input and output,
 * and apply filter accordingly. If Torch is available for use, then
 * TorchGenerateData() is called. Otherwise, GenerateData() in the
 * parent class (i.e., ImageToImageFilter) will be called.
 *
 * \ingroup PyTorch
 */

template <typename TInputImage, typename TOutputImage,
          typename TParentImageFilter =
              ImageToImageFilter<TInputImage, TOutputImage>>
class ITK_TEMPLATE_EXPORT TorchImageToImageFilter : public TParentImageFilter {
public:
  static constexpr bool ConceptCheck =
      std::is_same<TInputImage,
                   TorchImage<typename TInputImage::PixelType,
                              TInputImage::ImageDimension>>::value ||
      std::is_same<TOutputImage,
                   TorchImage<typename TOutputImage::PixelType,
                              TOutputImage::ImageDimension>>::value;
  static_assert(ConceptCheck,
                "itk::TorchImageToImageFilter: at least one of the first two "
                "template parameters must be an itk::TorchImage.");
  ITK_DISALLOW_COPY_AND_ASSIGN(TorchImageToImageFilter);

  /** Standard class type aliases. */
  using Self = TorchImageToImageFilter;
  using Superclass = TParentImageFilter;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Some convenient type aliases. */
  using InputImageType = TInputImage;
  using InputImagePointer = typename Self::InputImageType::Pointer;
  using InputImageConstPointer = typename Self::InputImageType::ConstPointer;
  using InputImageRegionType = typename Self::InputImageType::RegionType;
  using InputImagePixelType = typename Self::InputImageType::PixelType;
  using OutputImageType = TOutputImage;
  using OutputImagePointer = typename Self::OutputImageType::Pointer;
  using OutputImageConstPointer = typename Self::OutputImageType::ConstPointer;
  using OutputImageRegionType = typename Self::OutputImageType::RegionType;
  using OutputImagePixelType = typename Self::OutputImageType::PixelType;

  /** ImageDimension constants */
  static constexpr unsigned int InputImageDimension =
      Self::TInputImage::ImageDimension;
  static constexpr unsigned int OutputImageDimension =
      Self::TOutputImage::ImageDimension;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(TorchImageToImageFilter, TParentImageFilter);

protected:
  TorchImageToImageFilter() = default;
  ~TorchImageToImageFilter() override = default;
};

} // end namespace itk

#endif

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

#include "itkTorchImageToImageFilter.h"

#include "itkCommand.h"
#include "itkTestingMacros.h"
#include "itkRGBPixel.h"
#include "itkRGBAPixel.h"
#include "itkVector.h"
#include "itkCovariantVector.h"

namespace
{
class ShowProgress : public itk::Command
{
public:
  itkNewMacro( ShowProgress );

  void
  Execute( itk::Object *caller, const itk::EventObject &event ) override
  {
    Execute( ( const itk::Object * )caller, event );
  }

  void
  Execute( const itk::Object *caller, const itk::EventObject &event ) override
  {
    if( !itk::ProgressEvent().CheckEvent( &event ) )
    {
      return;
    }
    const auto *processObject = dynamic_cast< const itk::ProcessObject * >( caller );
    if( !processObject )
    {
      return;
    }
    std::cout << " " << processObject->GetProgress();
  }
};
} // namespace

int itkTorchImageToImageFilterTest( int argc, char *argv[] )
{
  std::cout << "Test compiled " << __DATE__ << " " << __TIME__ << std::endl;

  if( argc< 2 )
    {
    std::cerr << "Missing parameters." << std::endl;
    std::cerr << "Usage: " << itkNameOfTestExecutableMacro( argv );
    std::cerr << " outputImage";
    std::cerr << std::endl;
    return EXIT_FAILURE;
    }
  // const char * const outputImageFileName = argv[1];

  // Torch supports:
  //   Unsigned integer types: 1, 8 bits.
  //   Signed integer types: 8, 16, 32, 64 bits.
  //   Floating point types: 16, 32, 64 bits
  // though we do not support 16-bit floats.
#if 0
  {
    // This should fail to compile because the itk::TorchImageToImageFilter template parameter must be an
    // itk::TorchImage.
    using PixelType = bool;
    constexpr int ImageDimension = 6;
    using ImageType = itk::Image< PixelType, ImageDimension >;
    using FilterType = itk::TorchImageToImageFilter< ImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
#endif
  {
    using PixelType = bool;
    constexpr int ImageDimension = 6;
    using InputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using OutputImageType = itk::Image< PixelType, ImageDimension >;
    using FilterType = itk::TorchImageToImageFilter< InputTorchImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
  {
    using PixelType = uint8_t;
    constexpr int ImageDimension = 6;
    using InputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using OutputImageType = itk::Image< PixelType, ImageDimension >;
    using FilterType = itk::TorchImageToImageFilter< InputTorchImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
  {
    using PixelType = int8_t;
    constexpr int ImageDimension = 4;
    using InputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using OutputImageType = itk::Image< PixelType, ImageDimension >;
    using FilterType = itk::TorchImageToImageFilter< InputTorchImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
  {
    using PixelType = int16_t;
    constexpr int ImageDimension = 3;
    using InputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using OutputImageType = itk::Image< PixelType, ImageDimension >;
    using FilterType = itk::TorchImageToImageFilter< InputTorchImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
  {
    using PixelType = int32_t;
    constexpr int ImageDimension = 2;
    using InputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using OutputImageType = itk::Image< PixelType, ImageDimension >;
    using FilterType = itk::TorchImageToImageFilter< InputTorchImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
  {
    using PixelType = int64_t;
    constexpr int ImageDimension = 1;
    using InputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using OutputImageType = itk::Image< PixelType, ImageDimension >;
    using FilterType = itk::TorchImageToImageFilter< InputTorchImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
  {
    using PixelType = float;
    constexpr int ImageDimension = 2;
    using InputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using OutputImageType = itk::Image< PixelType, ImageDimension >;
    using FilterType = itk::TorchImageToImageFilter< InputTorchImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
  {
    using PixelType = double;
    constexpr int ImageDimension = 1;
    using InputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using OutputImageType = itk::Image< PixelType, ImageDimension >;
    using FilterType = itk::TorchImageToImageFilter< InputTorchImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
  {
    using PixelType = itk::RGBPixel< uint8_t >;
    constexpr int ImageDimension = 2;
    using InputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using OutputImageType = itk::Image< PixelType, ImageDimension >;
    using FilterType = itk::TorchImageToImageFilter< InputTorchImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
  {
    using PixelType = itk::RGBAPixel< int16_t >;
    constexpr int ImageDimension = 2;
    using InputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using OutputImageType = itk::Image< PixelType, ImageDimension >;
    using FilterType = itk::TorchImageToImageFilter< InputTorchImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
  {
    using PixelType = itk::Vector< float, 2 >;
    constexpr int ImageDimension = 3;
    using InputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using OutputImageType = itk::Image< PixelType, ImageDimension >;
    using FilterType = itk::TorchImageToImageFilter< InputTorchImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
  {
    using PixelType = itk::CovariantVector< double, 3 >;
    constexpr int ImageDimension = 2;
    using InputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using OutputImageType = itk::Image< PixelType, ImageDimension >;
    using FilterType = itk::TorchImageToImageFilter< InputTorchImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }

  std::cout << "Test finished." << std::endl;
  return EXIT_SUCCESS;
}

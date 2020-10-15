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

// ITK
#include "itkCommand.h"
#include "itkCovariantVector.h"
#include "itkImageFileWriter.h"
#include "itkRGBAPixel.h"
#include "itkRGBPixel.h"
#include "itkVector.h"

// ITK testing
#include "itkMersenneTwisterRandomVariateGenerator.h"
#include "itkTestingMacros.h"

// ITKPyTorch
#include "itkImageToTorchImageFilter.h"
#include "itkTorchImageToImageFilter.h"

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

int itkImageToTorchImageFilterTest( int argc, char *argv[] )
{
  std::cout << "Test compiled " << __DATE__ << " " << __TIME__ << std::endl;

  if( argc< 2 )
    {
    std::cerr << "Missing parameters." << std::endl;
    std::cerr << "Usage: " << itkNameOfTestExecutableMacro( argv );
    std::cerr << " outputTorchImage";
    std::cerr << std::endl;
    return EXIT_FAILURE;
    }
  const char * const outputImageFileName = argv[1];

  // Torch supports:
  //   Unsigned integer types: 1, 8 bits.
  //   Signed integer types: 8, 16, 32, 64 bits.
  //   Floating point types: 16, 32, 64 bits
  // though we do not support 16-bit floats.
#if 0
  {
    // This should fail to compile because the itk::TorchImageToImageFilter template parameter must not be an
    // itk::TorchImage.
    using PixelType = bool;
    constexpr int ImageDimension = 6;
    using TorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using FilterType = itk::ImageToTorchImageFilter< TorchImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
#endif
  {
    using PixelType = bool;
    constexpr int ImageDimension = 6;
    using InputImageType = itk::Image< PixelType, ImageDimension >;
    using OutputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using FilterType = itk::ImageToTorchImageFilter< InputImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
  {
    using PixelType = uint8_t;
    constexpr int ImageDimension = 6;
    using InputImageType = itk::Image< PixelType, ImageDimension >;
    using OutputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using FilterType = itk::ImageToTorchImageFilter< InputImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
  {
    using PixelType = int8_t;
    constexpr int ImageDimension = 4;
    using InputImageType = itk::Image< PixelType, ImageDimension >;
    using OutputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using FilterType = itk::ImageToTorchImageFilter< InputImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
  {
    using PixelType = int16_t;
    constexpr int ImageDimension = 3;
    using InputImageType = itk::Image< PixelType, ImageDimension >;
    using OutputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using FilterType = itk::ImageToTorchImageFilter< InputImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
  {
    using PixelType = int32_t;
    constexpr int ImageDimension = 2;
    using InputImageType = itk::Image< PixelType, ImageDimension >;
    using OutputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using FilterType = itk::ImageToTorchImageFilter< InputImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
  {
    using PixelType = int64_t;
    constexpr int ImageDimension = 1;
    using InputImageType = itk::Image< PixelType, ImageDimension >;
    using OutputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using FilterType = itk::ImageToTorchImageFilter< InputImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
  {
    using PixelType = float;
    constexpr int ImageDimension = 2;
    using InputImageType = itk::Image< PixelType, ImageDimension >;
    using OutputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using FilterType = itk::ImageToTorchImageFilter< InputImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
  {
    using PixelType = double;
    constexpr int ImageDimension = 1;
    using InputImageType = itk::Image< PixelType, ImageDimension >;
    using OutputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using FilterType = itk::ImageToTorchImageFilter< InputImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
  {
    using PixelType = itk::RGBPixel< uint8_t >;
    constexpr int ImageDimension = 2;
    using InputImageType = itk::Image< PixelType, ImageDimension >;
    using OutputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using FilterType = itk::ImageToTorchImageFilter< InputImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
  {
    using PixelType = itk::RGBAPixel< int16_t >;
    constexpr int ImageDimension = 2;
    using InputImageType = itk::Image< PixelType, ImageDimension >;
    using OutputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using FilterType = itk::ImageToTorchImageFilter< InputImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
  {
    using PixelType = itk::Vector< float, 2 >;
    constexpr int ImageDimension = 3;
    using InputImageType = itk::Image< PixelType, ImageDimension >;
    using OutputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using FilterType = itk::ImageToTorchImageFilter< InputImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }
  {
    using PixelType = itk::CovariantVector< double, 3 >;
    constexpr int ImageDimension = 2;
    using InputImageType = itk::Image< PixelType, ImageDimension >;
    using OutputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;
    using FilterType = itk::ImageToTorchImageFilter< InputImageType >;
    typename FilterType::Pointer filter = FilterType::New();
  }

  // Copy an image from an itkImage to an itkTorchImage and back.
  //

  // We will need a random number generator.
  using UniformGeneratorType = itk::Statistics::MersenneTwisterRandomVariateGenerator;
  UniformGeneratorType::Pointer uniformGenerator = UniformGeneratorType::New();
  uniformGenerator->Initialize( 20200925 );

  // The itkImage and itkTorchImage types we are testing.
  using PixelType = uint8_t;
  constexpr int ImageDimension = 2;
  using InputImageType = itk::Image< PixelType, ImageDimension >;
  using OutputTorchImageType = itk::TorchImage< PixelType, ImageDimension >;

  // Build an itkImage
  const InputImageType::SizeValueType testSize = 200;
  InputImageType::SizeType size;
  size.Fill( testSize );
  InputImageType::Pointer image = InputImageType::New();
  image->SetRegions( size );
  image->Allocate();
  using InputRegionIterator = itk::ImageRegionIterator< InputImageType >;
  InputRegionIterator iter {image, size};
  PixelType tmp;
  for( iter.GoToBegin(); !iter.IsAtEnd(); ++iter )
    {
    tmp = std::floor( uniformGenerator->GetVariate() * 256 );
    iter.Set( tmp );
    }

  using WriterType = itk::ImageFileWriter< InputImageType >;
  WriterType::Pointer writer = WriterType::New();

#if 0
  // Save the baseline image
  writer->SetInput( image );
#else
  // Check that we get the same as the baseline
  using FirstFilterType = itk::ImageToTorchImageFilter< InputImageType >;
  typename FirstFilterType::Pointer firstFilter = FirstFilterType::New();
  firstFilter->SetInput( image );

  using SecondFilterType = itk::TorchImageToImageFilter< OutputTorchImageType >;
  typename SecondFilterType::Pointer secondFilter = SecondFilterType::New();
  secondFilter->SetInput( firstFilter->GetOutput() );
  writer->SetInput( secondFilter->GetOutput() );
#endif

  writer->SetFileName( outputImageFileName );
  writer->SetUseCompression( true );

  TRY_EXPECT_NO_EXCEPTION( writer->Update() );

  std::cout << "Test finished." << std::endl;
  return EXIT_SUCCESS;
}

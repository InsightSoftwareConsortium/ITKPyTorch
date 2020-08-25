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

#include "itkTorchImage.h"

#include "itkCommand.h"
#include "itkImageFileWriter.h"
#include "itkTestingMacros.h"
#include "itkRGBPixel.h"

namespace
{
class ShowProgress : public itk::Command
{
public:
  itkNewMacro(ShowProgress);

  void
  Execute(itk::Object *caller, const itk::EventObject &event) override
  {
    Execute((const itk::Object *)caller, event);
  }

  void
  Execute(const itk::Object *caller, const itk::EventObject &event) override
  {
    if (!itk::ProgressEvent().CheckEvent(&event))
    {
      return;
    }
    const auto *processObject = dynamic_cast<const itk::ProcessObject *>(caller);
    if (!processObject)
    {
      return;
    }
    std::cout << " " << processObject->GetProgress();
  }
};
} // namespace

int itkTorchImageTest(int argc, char *argv[])
{
  if (argc < 2)
    {
    std::cerr << "Missing parameters." << std::endl;
    std::cerr << "Usage: " << itkNameOfTestExecutableMacro(argv);
    std::cerr << " outputImage";
    std::cerr << std::endl;
    return EXIT_FAILURE;
    }
  // const char * const outputImageFileName = argv[1];

  // Torch supports:
  // Unsigned integer types: 8 bits.
  // Signed integer types: 1, 8, 16, 32, 64 bits.
  // Floating point types: 16, 32, 64 bits
  {
    using ImageType = itk::TorchImage< unsigned char, 3 >;
    ImageType::Pointer image = ImageType::New();
  }
  {
    using ImageType = itk::TorchImage< bool, 2 >;
    ImageType::Pointer image = ImageType::New();
  }
  {
    using ImageType = itk::TorchImage< signed char, 2 >;
    ImageType::Pointer image = ImageType::New();
  }
  {
    using ImageType = itk::TorchImage< short, 1 >;
    ImageType::Pointer image = ImageType::New();
  }
  {
    using ImageType = itk::TorchImage< int, 1 >;
    ImageType::Pointer image = ImageType::New();
  }
  {
    using ImageType = itk::TorchImage< long, 1 >;
    ImageType::Pointer image = ImageType::New();
  }
  {
    using ImageType = itk::TorchImage< long, 1 >;
    ImageType::Pointer image = ImageType::New();
  }
  {
    using ImageType = itk::TorchImage< double, 1 >;
    ImageType::Pointer image = ImageType::New();
  }
  {
    using ImageType = itk::TorchImage< itk::RGBPixel< short >, 3 >;
    ImageType::Pointer image = ImageType::New();
  }
  {
    using ImageType = itk::TorchImage< itk::RGBAPixel< short >, 1 >;
    ImageType::Pointer image = ImageType::New();
  }
  {
    using ImageType = itk::TorchImage< itk::Vector< short, 3 >, 4 >;
    ImageType::Pointer image = ImageType::New();
  }
  {
    using ImageType = itk::TorchImage< itk::CovariantVector< short, 4 >, 5 >;
    ImageType::Pointer image = ImageType::New();
  }
  {
    using ImageType = itk::TorchImage< itk::Vector< itk::Vector< unsigned char, 2 >, 3 >, 4 >;
    ImageType::Pointer image = ImageType::New();
  }
  {
    using ImageType = itk::TorchImage< itk::CovariantVector< itk::Vector< itk::RGBAPixel< unsigned char >, 2 >, 3 >, 4 >;
    ImageType::Pointer image = ImageType::New();
  }

  {
    using PixelType = float;
    constexpr int ImageDimension = 2;
    const int SizePerDimension = 128;
    const std::string StructName = "TorchImage<float, 2>";
    const PixelType firstValue = 1.1f;
    const PixelType secondValue = 1.2f;
    const PixelType thirdValue = 1.3f;

    // Duplicated testing code
    //
    using ImageType = itk::TorchImage< PixelType, ImageDimension >;
    ImageType::Pointer image = ImageType::New();

    // Create input image
    ImageType::SizeType size;
    const bool response = image->SetDevice( ImageType::itkCUDA );
    itkAssertOrThrowMacro(response, StructName + "::SetDevice failed");
    ImageType::DeviceType deviceType;
    int64_t cudaDeviceNumber;
    image->GetDevice( deviceType, cudaDeviceNumber );
    itkAssertOrThrowMacro( deviceType == ImageType::itkCUDA, StructName + "::GetDevice failed for deviceType");
    itkAssertOrThrowMacro( cudaDeviceNumber == 0, StructName + "::GetDevice failed for cudaDeviceNumber");

    size.Fill( SizePerDimension );
    image->SetRegions( size );
    image->Allocate();

    ImageType::IndexType location0;
    location0.Fill( 0 );
    ImageType::IndexType location1;
    location1.Fill( 1 );
    PixelType pixelValue;

    // TRY_EXPECT_NO_EXCEPTION();

    image->FillBuffer( firstValue );
    pixelValue = image->GetPixel( location0 );
    itkAssertOrThrowMacro( pixelValue == firstValue, StructName + "::FillBuffer failed" );
    pixelValue = image->GetPixel( location1 );
    itkAssertOrThrowMacro( pixelValue == firstValue, StructName + "::FillBuffer failed" );

    image->GetPixel( location0 ) = secondValue;
    pixelValue = image->GetPixel( location0 );
    itkAssertOrThrowMacro( pixelValue == secondValue, StructName + "::GetPixel as lvalue failed" );

    pixelValue = image->GetPixel( location1 );
    itkAssertOrThrowMacro( pixelValue == firstValue, StructName + "::GetPixel has side effect" );

    image->SetPixel( location0, thirdValue );
    pixelValue = image->GetPixel( location0 );
    itkAssertOrThrowMacro( pixelValue == thirdValue, StructName + "::SetPixel failed" );

    ImageType::Pointer image2 = ImageType::New();
    image2->SetRegions( size );
    image2->Allocate();
    image2->Graft( image );
  }

  {
    using PixelType = itk::RGBPixel< unsigned char >;
    constexpr int ImageDimension = 3;
    const int SizePerDimension = 20;
    const std::string StructName = "TorchImage<RGBPixel<unsigned char>, 3>";
    const PixelType firstValue = []() -> PixelType
      {
      PixelType temp;
      temp[0] = 1;
      temp[1] = 2;
      temp[2] = 3;
      return temp;
      }();
    const PixelType secondValue = []() -> PixelType
      {
      PixelType temp;
      temp[0] = 5;
      temp[1] = 5;
      temp[2] = 5;
      return temp;
      }();
    const PixelType thirdValue = []() -> PixelType
      {
      PixelType temp;
      temp[0] = 10;
      temp[1] = 8;
      temp[2] = 5;
      return temp;
      }();

    // Duplicated testing code
    //
    using ImageType = itk::TorchImage< PixelType, ImageDimension >;
    ImageType::Pointer image = ImageType::New();

    // Create input image
    ImageType::SizeType size;
    const bool response = image->SetDevice( ImageType::itkCUDA );
    itkAssertOrThrowMacro(response, StructName + "::SetDevice failed");
    ImageType::DeviceType deviceType;
    int64_t cudaDeviceNumber;
    image->GetDevice( deviceType, cudaDeviceNumber );
    itkAssertOrThrowMacro( deviceType == ImageType::itkCUDA, StructName + "::GetDevice failed for deviceType");
    itkAssertOrThrowMacro( cudaDeviceNumber == 0, StructName + "::GetDevice failed for cudaDeviceNumber");

    size.Fill( SizePerDimension );
    image->SetRegions( size );
    image->Allocate();

    ImageType::IndexType location0;
    location0.Fill( 0 );
    ImageType::IndexType location1;
    location1.Fill( 1 );
    PixelType pixelValue;

    // TRY_EXPECT_NO_EXCEPTION();

    image->FillBuffer( firstValue );
    pixelValue = image->GetPixel( location0 );
    itkAssertOrThrowMacro( pixelValue == firstValue, StructName + "::FillBuffer failed" );
    pixelValue = image->GetPixel( location1 );
    itkAssertOrThrowMacro( pixelValue == firstValue, StructName + "::FillBuffer failed" );

    image->GetPixel( location0 ) = secondValue;
    pixelValue = image->GetPixel( location0 );
    itkAssertOrThrowMacro( pixelValue == secondValue, StructName + "::GetPixel as lvalue failed" );

    pixelValue = image->GetPixel( location1 );
    itkAssertOrThrowMacro( pixelValue == firstValue, StructName + "::GetPixel has side effect" );

    image->SetPixel( location0, thirdValue );
    pixelValue = image->GetPixel( location0 );
    itkAssertOrThrowMacro( pixelValue == thirdValue, StructName + "::SetPixel failed" );

    ImageType::Pointer image2 = ImageType::New();
    image2->SetRegions( size );
    image2->Allocate();
    image2->Graft( image );
  }

  std::cout << "Test finished." << std::endl;
  return EXIT_SUCCESS;
}

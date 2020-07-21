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
  // !!! const char *outputImageFileName = argv[1];

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

  using ImageType = itk::TorchImage< float, 2 >;
  ImageType::Pointer image = ImageType::New();

  // Create input image to avoid test dependencies.
  ImageType::SizeType size;
  size.Fill(128);
  image->SetRegions(size);
  image->Allocate();
  image->FillBuffer(1.1f);

  std::cout << "Test finished." << std::endl;
  return EXIT_SUCCESS;
}

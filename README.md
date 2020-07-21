# ITKPyTorch

[ ![ Apache 2.0 License ](https://img.shields.io/badge/License-Apache%202.0-blue.svg) ](https://github.com/InsightSoftwareConsortium/ITKPyTorch/blob/master/LICENSE) [ ![ Build, test, package status ](https://github.com/InsightSoftwareConsortium/ITKPyTorch/workflows/Build,%20test,%20package/badge.svg) ](https://github.com/InsightSoftwareConsortium/ITKPyTorch/actions?query=workflow%3A%22Build%2C+test%2C+package%22)

## Overview

This [ Insight Toolkit (ITK) ](https://itk.org/) module provides support for PyTorch Tensors as ITK Images.

### Try it online

We will set up a demonstration of this using [ Binder ](www.mybinder.org) that anyone can try.  Click here: [ ![ Binder ](https://mybinder.org/badge_logo.svg) ](https://mybinder.org/v2/gh/InsightSoftwareConsortium/ITKPyTorch/master?filepath=examples%2FITKPyTorch.ipynb)

### Technical description

The ITK `TorchImage< TPixel, VImageDimension >` templated class in C++ is a subclass of the corresponding ITK `Image< TPixel, VImageDimension >` templated class and supports the same pixel types (i.e., integers, real numbers, `RGBPixel`, `RGBAPixel`, `Vector`, and `CovariantVector`) and the same numbers of dimensions.  A `TorchImage` can be used in lieu of an `Image` in ITK pipelines but also can be used with filters and transformations that are specialized for `TorchImage` objects.  These specialized pipeline steps are executed on a CUDA GPU using the C++ `libTorch` interface.

It is intended that this module enable MONAI intergration.

## Installation for Python
[ ![ PyPI Version ](https://img.shields.io/pypi/v/itk-pytorch.svg) ](https://pypi.python.org/pypi/itk-pytorch)

## Usage in Python

### Functional interface to ITK

### ITK pipeline interface

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

#ifndef __PyTorchExport_h
#define __PyTorchExport_h

#include "itkConfigure.h"
#include "itkMacro.h"

// Setup symbol export
#define PyTorch_HIDDEN ITK_ABI_HIDDEN

#if !defined( ITKSTATIC )
#ifdef PyTorch_EXPORTS
#define PyTorch_EXPORT ITK_ABI_EXPORT
#else
#define PyTorch_EXPORT ITK_ABI_IMPORT
#endif  /* PyTorch_EXPORTS */
#else
/* PyTorch is built as a static lib */
#if __GNUC__ >= 4
// Do not hide symbols in the static PyTorch library in case
// -fvisibility=hidden is used
#define PyTorch_EXPORT ITK_ABI_EXPORT
#else
#define PyTorch_EXPORT
#endif
#endif

#endif /* __PyTorchExport_h */

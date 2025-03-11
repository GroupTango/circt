//===- JSONGraphTraits.h - JSON graph traits for Model Explorer ------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the JSONGraphTraits
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWJSONGRAPHTRAITS_H
#define CIRCT_DIALECT_HW_HWJSONGRAPHTRAITS_H

#include "llvm/Support/DOTGraphTraits.h"

namespace circt {
namespace hw {

// Generic templated JSONGraphTraits for any graph type.
// Represents the DOTGraphTraits in JSON format.
template <typename GraphT>
struct JSONGraphTraits : public llvm::DOTGraphTraits<GraphT> {};
} // end namespace hw
} // end namespace circt

#endif // CIRCT_DIALECT_HW_HWJSONGRAPHTRAITS_H

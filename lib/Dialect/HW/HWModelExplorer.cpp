//===- HWModelExplorer.cpp - Model graph JSON generation ------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the implementation of the Model Explorer JSON generation
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWModelExplorer.h"
#include "mlir/Support/LLVM.h"

// Option used to write output of a Json graphing pass to.
llvm::cl::opt<std::string> outFile("outfile",
                                   llvm::cl::desc("Specify output file"),
                                   llvm::cl::value_desc("filename"));
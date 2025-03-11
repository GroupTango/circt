//===- HWJsonInstanceDiffGraph.cpp - Model graph JSON generation *- C++ -*-===//
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
#include "circt/Dialect/HW/HWModuleGraph.h"
#include "circt/Dialect/HW/HWOps.h"

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <stack>

using namespace circt;
using namespace circt::hw;
using namespace circt::hw::detail;

namespace {

// Graph generator for Instance Diff Graphs.
class InstanceDiffGraphGenerator : GraphGenerator {
public:
  InstanceDiffGraphGenerator(HWOperationRef baseOriginalModule,
                             HWOperationRef baseNewModule,
                             llvm::raw_ostream *os)
      : GraphGenerator(os), baseOriginalModule(baseOriginalModule),
        baseNewModule(baseNewModule) {}

  std::string generateGraphJson() override {
    // TO BE IMPLEMENTED
    return wrapJson(outputJsonObjects);
  }

protected:
  HWOperationRef baseOriginalModule;
  HWOperationRef baseNewModule;
  std::stack<std::pair<HWOperationRef, int64_t>> modulesToProcess;
};

} // end anonymous namespace

namespace circt {
namespace hw {

// Public API functions instantiates the corresponding generator.
std::string MlirInstanceDiffGraphJson(HWOperationRef baseOriginalModule,
                                      HWOperationRef baseNewModule,
                                      llvm::raw_ostream *os) {
  InstanceDiffGraphGenerator generator(baseOriginalModule, baseNewModule, os);
  return generator.generateGraphJson();
}

} // end namespace hw
} // end namespace circt
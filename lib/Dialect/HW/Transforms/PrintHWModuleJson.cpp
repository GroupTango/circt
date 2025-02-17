//===- PrintHWModuleJson.cpp - Print the model graph --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Prints a JSON representation of all modules in the MLIR file, in a format
// that can be consumed by the Google Model Explorer.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWModelExplorerInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"

#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_PRINTHWMODULEJSON
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;

namespace {
struct PrintHWModuleJsonPass
    : public circt::hw::impl::PrintHWModuleJsonBase<PrintHWModuleJsonPass> {

  PrintHWModuleJsonPass(raw_ostream &os) : os(os) {}

  void runOnOperation() override {
    mlir::Operation *baseOp = getOperation();
    const std::string Json = circt::hw::MlirToOperationGraphJson(baseOp, &os);

    os << "JSON:\n\n";
    os << Json;

    // New lines to space out raw MLIR in case we print it.
    os << "\n\n";
  } // namespace

  raw_ostream &os;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::hw::createPrintHWModuleJsonPass() {
  return std::make_unique<PrintHWModuleJsonPass>(llvm::errs());
}

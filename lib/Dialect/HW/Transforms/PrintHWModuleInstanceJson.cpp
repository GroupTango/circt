//===- PrintHWModuleInstanceJson.cpp - Print the model graph --------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Prints a JSON representation of all modules (based on instancing) in the MLIR
// file, in a format that can be consumed by the Google Model Explorer.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWModelExplorer.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"

#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_PRINTHWMODULEINSTANCEJSON
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;

namespace {
struct PrintHWModuleInstanceJsonPass
    : public circt::hw::impl::PrintHWModuleInstanceJsonBase<
          PrintHWModuleInstanceJsonPass> {

  PrintHWModuleInstanceJsonPass(raw_ostream &os) : os(os) {}

  void runOnOperation() override {
    mlir::Operation *baseOp = getOperation();
    const std::string json = circt::hw::MlirToInstanceGraphJson(baseOp, &os);

    if (!outFile.empty()) {
      std::error_code error;
      llvm::raw_fd_ostream file(outFile, error);
      if (error) {
        os << "Error opening file: " << error.message() << "\n";
        return;
      }
      file << json;
    } else {
      os << json;
    }
  } // namespace

  raw_ostream &os;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::hw::createPrintHWModuleInstanceJsonPass() {
  return std::make_unique<PrintHWModuleInstanceJsonPass>(llvm::errs());
}

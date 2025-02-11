//===- PrintHWModuleJson.cpp - Print the instance graph --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Prints an HW module as a JSON graph compatible with Google's Model Explorer.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWModuleGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/JSON.h"
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
    getOperation().walk([&](hw::HWModuleOp module) {
      llvm::json::Object json = llvm::json::Object();

      auto entryNode =
          llvm::GraphTraits<circt::hw::HWModuleOp>::getEntryNode(module);

      for (auto it = llvm::GraphTraits<circt::hw::HWModuleOp>::child_begin(
                    entryNode),
                end = llvm::GraphTraits<circt::hw::HWModuleOp>::child_end(
                    entryNode);
           it != end; ++it) {
        auto node = *it;
        // Process each node as needed
        llvm::outs() << "Visiting node: " << node->getName() << "\n";
      }
    });
  }
  raw_ostream &os;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::hw::createPrintHWModuleJsonPass() {
  return std::make_unique<PrintHWModuleJsonPass>(llvm::errs());
}

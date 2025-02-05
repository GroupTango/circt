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

      json["label"] = module.getName().data();

      // Populate the module attributes
      llvm::json::Object attributes = llvm::json::Object();
      if (!module->getAttrs().empty()) {
        for (NamedAttribute attr : module->getAttrs()) {
          std::string value;
          llvm::raw_string_ostream sstream(value);
          attr.getValue().print(sstream);
          attributes[attr.getName().str()] = sstream.str();
        }
      }

      json["attributes"] = std::move(attributes);

      // Recurse into each of the regions attached to the operation.
      llvm::json::Array regions;
      for (Region &region : module->getRegions())
        regions.push_back(parseRegion(region));
      json["subgraphs"] = std::move(regions);

      // Output json
      os << "json: " << llvm::json::Value(std::move(json)) << "\n";
    });
  }

  llvm::json::Object parseRegion(Region &region) {
    // A region does not hold anything by itself other than a list of blocks.
    llvm::json::Object json = llvm::json::Object();
    json["label"] = "region" + std::to_string(region.getRegionNumber());

    llvm::json::Array blocks;
    for (Block &block : region.getBlocks()) {
      blocks.push_back(parseBlock(block));
    }
    json["subgraphs"] = std::move(blocks);
    return json;
  }

  llvm::json::Object parseBlock(Block &block) {
    // Print the block intrinsics properties (basically: argument list)
    // os << "Block with " << block.getNumArguments() << " arguments, "
    //    << block.getNumSuccessors()
    //    << " successors, and "
    //    // Note, this `.size()` is traversing a linked-list and is O(n).
    //    << block.getOperations().size() << " operations\n";

    llvm::json::Object json = llvm::json::Object();

    // A block main role is to hold a list of Operations: let's recurse into
    // printing each operation.
    llvm::json::Array blocks;
    for (Operation &op : block.getOperations()) {
      blocks.push_back(parseOp(op));
    }
    json["subgraphs"] = std::move(blocks);
    return json;
  }

  llvm::json::Object parseOp(Operation &op) {
    llvm::json::Object json = llvm::json::Object();
    json["label"] = op.getName().getStringRef().str();

    llvm::json::Array operands;
    for (Value operand : op.getOperands()) {
      std::string value;
      llvm::raw_string_ostream sstream(value);
      operand.printAsOperand(sstream, mlir::OpPrintingFlags());
      operands.push_back(llvm::json::Object{{"label", sstream.str()}});
    }
    json["subgraphs"] = std::move(operands);
    return json;
  }
  raw_ostream &os;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::hw::createPrintHWModuleJsonPass() {
  return std::make_unique<PrintHWModuleJsonPass>(llvm::errs());
}

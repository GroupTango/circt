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

// Use the GraphTraits specialized for circt::hw::HWModuleOp to traverse the
// graph.
using NodeType = circt::hw::detail::HWOperation;
using NodeRef = NodeType *;
using HWModuleOpGraphTraits = llvm::GraphTraits<HWModuleOp>;
using HWModuleOpJSONGraphTraits =
    circt::hw::JSONGraphTraits<circt::hw::HWModuleOp>;

namespace {
struct PrintHWModuleJsonPass
    : public circt::hw::impl::PrintHWModuleJsonBase<PrintHWModuleJsonPass> {
  PrintHWModuleJsonPass(raw_ostream &os) : os(os), jsonGraphTraits(false) {}
  void runOnOperation() override {
    getOperation().walk([&](HWModuleOp module) {
      llvm::SmallPtrSet<NodeRef, 16> visited;

      llvm::json::Object moduleJson;
      llvm::json::Array moduleNodes;
      moduleJson["name"] = module.getNameAttr().getValue();
      moduleJson["label"] = jsonGraphTraits.getNodeLabel(module, module);
      moduleJson["attributes"] =
          jsonGraphTraits.getNodeAttributes(module, module);

      // Iterate over all top-level nodes in the module.
      for (auto it = HWModuleOpGraphTraits::nodes_begin(module),
                end = HWModuleOpGraphTraits::nodes_end(module);
           it != end; ++it) {
        NodeRef node = *it;
        if (visited.count(node) == 0)
          moduleNodes.push_back(visitNode(node, module, visited));
      }
      moduleJson["children"] = std::move(moduleNodes);

      // Output the JSON representation of the module's graph.
      os << llvm::json::Value(std::move(moduleJson)) << "\n";
    });
  }

  llvm::json::Object visitNode(NodeRef node, const HWModuleOp &module,
                               llvm::SmallPtrSetImpl<NodeRef> &visited) {
    if (visited.count(node) > 0)
      // TODO: should copy the node's JSON representation that we already
      // created
      return llvm::json::Object();

    visited.insert(node);

    llvm::json::Object json;
    json["name"] = node->getName().getStringRef();
    json["label"] = jsonGraphTraits.getNodeLabel(node, module);
    json["attributes"] = jsonGraphTraits.getNodeAttributes(node, module);

    llvm::json::Array children;
    for (auto it = HWModuleOpGraphTraits::child_begin(node),
              end = HWModuleOpGraphTraits::child_end(node);
         it != end; ++it) {
      NodeRef child = *it;
      llvm::json::Object childJson = visitNode(child, module, visited);
      children.push_back(std::move(childJson));
    }
    json["children"] = std::move(children);
    return json;
  }

  raw_ostream &os;
  HWModuleOpJSONGraphTraits jsonGraphTraits;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::hw::createPrintHWModuleJsonPass() {
  return std::make_unique<PrintHWModuleJsonPass>(llvm::errs());
}

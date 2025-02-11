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

using NodeType = circt::hw::detail::HWOperation;
using NodeRef = NodeType *;

namespace {
struct PrintHWModuleJsonPass
    : public circt::hw::impl::PrintHWModuleJsonBase<PrintHWModuleJsonPass> {
  PrintHWModuleJsonPass(raw_ostream &os) : os(os) {}
  void runOnOperation() override {
    getOperation().walk([&](hw::HWModuleOp module) {
      // Retrieve the entry node via your GraphTraits specialization.
      NodeRef entryNode =
          llvm::GraphTraits<hw::HWModuleOp>::getEntryNode(module);

      llvm::SmallPtrSet<NodeRef, 16> visited;
      llvm::json::Object moduleJson = visitNode(entryNode, module, visited);

      // Output the JSON representation of the module's graph.
      os << llvm::json::Value(std::move(moduleJson)) << "\n";
    });
  }
  llvm::json::Object visitNode(NodeRef node, const hw::HWModuleOp &module,
                               llvm::SmallPtrSetImpl<NodeRef> &visited) {
    // If we've seen this node already, return an empty JSON object.
    if (!visited.insert(node).second)
      return llvm::json::Object();

    llvm::DOTGraphTraits<circt::hw::HWModuleOp> traits(false);

    llvm::json::Object json;
    json["name"] = node->getName().getStringRef();
    json["label"] =
        llvm::DOTGraphTraits<circt::hw::HWModuleOp>::getNodeLabel(node, module);
    json["attributes"] = traits.getNodeAttributes(node, module);

    llvm::json::Array children;
    // Use the GraphTraits specialized for circt::hw::HWModuleOp.
    // Note that our specialization inherited the child iterators
    // from GraphTraits<HWOperation*>, which use op->user_begin()/user_end().
    for (auto it = llvm::GraphTraits<circt::hw::HWModuleOp>::child_begin(node),
              end = llvm::GraphTraits<circt::hw::HWModuleOp>::child_end(node);
         it != end; ++it) {
      NodeRef child = *it;
      llvm::json::Object childJson = visitNode(child, module, visited);
      // Optionally, check if childJson is empty to skip uninteresting nodes.
      children.push_back(std::move(childJson));
    }
    json["children"] = std::move(children);
    return json;
  }

  raw_ostream &os;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::hw::createPrintHWModuleJsonPass() {
  return std::make_unique<PrintHWModuleJsonPass>(llvm::errs());
}

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

#include "circt/Dialect/HW/HWModuleGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <mlir/Support/LLVM.h>
#include <stack>
#include <vector>

namespace circt {
namespace hw {
#define GEN_PASS_DEF_PRINTHWMODULEJSON
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;

// Use the GraphTraits specialized for circt::hw::HWModuleOp to traverse the
// module graph.
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

    // NEED TO DO SOME CHANGES TO DISPLAY MORE INFO MISSING MODULES BETTER
    // (Currently passing nullptr so can't print any info such as missing module
    // names)

    // Preprocessing:
    // 1. Assign unique IDs to all operations
    int64_t counter = 0;
    mlir::Operation *baseOp = getOperation();

    baseOp->walk([&](mlir::Operation *op) {
      auto id = mlir::IntegerAttr::get(
          mlir::IntegerType::get(op->getContext(), 64), counter++);
      op->setAttr("hw.unique_id", id);
    });

    // 2. Find all top level modules, populate moduleMap and incomingEdges
    for (mlir::Region &region : baseOp->getRegions()) {
      for (mlir::Block &block : region.getBlocks()) {
        for (mlir::Operation &op : block.getOperations()) {
          llvm::TypeSwitch<mlir::Operation *>(&op)
              .Case<circt::hw::HWModuleOp>([&](auto module) {
                os << "Found HWModuleOp: " << module.getName() << "\n";
                moduleMap[module.getName()] = &op;
                populateIncomingEdges(module);
              })
              .Case<circt::hw::HWModuleExternOp>([&](auto module) {
                os << "Found HWModuleExternOp: " << module.getName()
                   << " SKIPPING\n";
              })
              .Case<circt::hw::HWModuleGeneratedOp>([&](auto module) {
                os << "Found HWModuleGeneratedOp: " << module.getName()
                   << " SKIPPING\n";
              })
              .Default([&](auto) {
                os << "Found unknown top level module type: " << op.getName()
                   << " SKIPPING\n";
              });
        }
      }
    }

    // Start processing Modules to JSON
    std::vector<std::pair<mlir::Operation *, std::string>> modulesToProcess;
    llvm::json::Array outputJsonObjects;

    // Note: Iterator only returns keys, not values
    for (auto const &x : moduleMap) {
      mlir::Operation *op = moduleMap[x.getKey()];
      modulesToProcess.push_back(std::make_pair(op, x.getKey().str()));

      os << "Adding top level Module for processing - Name: " << x.getKey()
         << " Type: " << op->getName() << "\n";
    }

    while (modulesToProcess.size() > 0) {
      std::pair<mlir::Operation *, std::string> nextPair =
          modulesToProcess.back();
      modulesToProcess.pop_back();
      mlir::Operation *module = nextPair.first;

      if (module == nullptr) {
        llvm::json::Object moduleJson;
        moduleJson["id"] = getUniqueId(module, nextPair.second);
        // Change to display said module name in future?
        moduleJson["label"] = "Unknown Module";
        moduleJson["namespace"] = nextPair.second;

        outputJsonObjects.push_back(std::move(moduleJson));
        continue;
      }

      bool hasInstances = false;

      for (mlir::Region &region : module->getRegions()) {
        for (mlir::Block &block : region.getBlocks()) {
          for (mlir::Operation &op : block.getOperations()) {

            NodeRef node = &op;
            HWModuleOp moduleOp = mlir::dyn_cast<circt::hw::HWModuleOp>(module);

            llvm::json::Object jsonObj{
                {"label", jsonGraphTraits.getNodeLabel(node, moduleOp)},
                {"attrs", jsonGraphTraits.getNodeAttributes(node, moduleOp)},
                {"id", getUniqueId(node, nextPair.second)},
                {"namespace", nextPair.second}};

            if (isa<InstanceOp>(op)) {
              InstanceOp instanceOp = cast<InstanceOp>(op);
              hasInstances = true;
              os << "Found InstanceOp: " << instanceOp.getReferencedModuleName()
                 << "\n";

              // Is it possible to get the value of the iterator, instead of
              // having to [] again? Also would adding std::move around the
              // string concat change anything?
              auto refModuleName = instanceOp.getReferencedModuleName();
              std::string newNamespace =
                  nextPair.second + "/" + refModuleName.str();
              auto it = moduleMap.find(refModuleName);
              if (it != moduleMap.end())
                modulesToProcess.push_back(
                    {moduleMap[refModuleName], newNamespace});
              else
                modulesToProcess.push_back({nullptr, newNamespace});

              // inter-module dependency, so we want (module -> hw.instance)
              jsonObj["incomingEdges"] = llvm::json::Array{llvm::json::Object{
                  {"sourceNodeId",
                   getUniqueId(moduleMap[refModuleName],
                               nextPair.second + "/" + refModuleName.str())},
                  {"sourceNodeOutputId", "0"},
                  {"targetNodeInputId", "0"}}};
            } else {
              // intra-module dependency, get from module graph
              jsonObj["incomingEdges"] =
                  getIncomingEdges(&op, moduleOp, nextPair.second);
            }

            outputJsonObjects.push_back(std::move(jsonObj));
          }
        }
      }

      // If this is a self contained module, we will display it as a graph
      // node.
      if (!hasInstances) {
        llvm::json::Object moduleJson;
        HWModuleOp moduleOp = mlir::dyn_cast<circt::hw::HWModuleOp>(module);
        moduleJson["id"] = getUniqueId(module, nextPair.second);
        moduleJson["namespace"] = nextPair.second;
        moduleJson["label"] = moduleOp.getNameAttr().getValue();
        moduleJson["attrs"] =
            jsonGraphTraits.getNodeAttributes(module, moduleOp);
        moduleJson["incomingEdges"] =
            getIncomingEdges(module, moduleOp, nextPair.second);

        outputJsonObjects.push_back(std::move(moduleJson));
        nextNodeId++;
      }
    }

    // Do some final wraps of our JSON Node Array, as needed by Model Explorer
    llvm::json::Object graphWrapper;
    graphWrapper["id"] = "test_mlir_file";
    graphWrapper["nodes"] = std::move(outputJsonObjects);

    llvm::json::Array graphArrayWrapper;
    graphArrayWrapper.push_back(std::move(graphWrapper));

    llvm::json::Object fileWrapper;
    fileWrapper["label"] = "model.json";
    fileWrapper["subgraphs"] = std::move(graphArrayWrapper);

    // Output final JSON
    llvm::json::Array jsonOutput{llvm::json::Value(std::move(fileWrapper))};
    os << "JSON:\n\n";
    os << llvm::json::Value(std::move(jsonOutput));

    // We print out the raw MLIR later for some reason? New lines to space out
    // the raw MLIR
    os << "\n\n";
  } // namespace

  void populateIncomingEdges(HWModuleOp module) {
    std::stack<NodeRef> nodesToVisit;
    llvm::SmallPtrSet<NodeRef, 32> visited;

    for (auto it = HWModuleOpGraphTraits::nodes_begin(module),
              end = HWModuleOpGraphTraits::nodes_end(module);
         it != end; ++it) {
      nodesToVisit.push(*it);
    }

    while (!nodesToVisit.empty()) {
      NodeRef current = nodesToVisit.top();
      nodesToVisit.pop();

      if (!visited.insert(current).second)
        continue;

      for (auto it = HWModuleOpGraphTraits::child_begin(current),
                end = HWModuleOpGraphTraits::child_end(current);
           it != end; ++it) {
        NodeRef child = *it;
        // os << child->getName() << " <- " << current->getName() << "\n";
        incomingEdges[child].push_back(current);
        nodesToVisit.push(child);
      }
    }
  }

  llvm::json::Array getIncomingEdges(NodeRef node, HWModuleOp module,
                                     std::string namesp) {
    llvm::json::Array edges;
    for (NodeRef parent : incomingEdges[node]) {
      edges.push_back(
          llvm::json::Object{{"sourceNodeId", getUniqueId(parent, namesp)},
                             {"sourceNodeOutputId", "0"},
                             {"targetNodeInputId", "0"}});
    }
    return edges;
  }

  std::string getUniqueId(mlir::Operation *node, const std::string &namesp) {
    if (node == nullptr)
      return namesp + "_" + std::to_string(nextNodeId++);

    return namesp + "_" +
           std::to_string(
               mlir::cast<IntegerAttr>(node->getAttr("hw.unique_id")).getInt());
  }

  raw_ostream &os;
  HWModuleOpJSONGraphTraits jsonGraphTraits;

  // Locate all modules in MLIR
  llvm::StringMap<mlir::Operation *> moduleMap;

  uint64_t nextNodeId = 0;

  llvm::DenseMap<NodeRef, std::vector<NodeRef>> incomingEdges;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::hw::createPrintHWModuleJsonPass() {
  return std::make_unique<PrintHWModuleJsonPass>(llvm::errs());
}

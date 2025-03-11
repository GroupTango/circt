//===- HWJsonInstanceGraph.cpp - Model graph JSON generation ----*- C++ -*-===//
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

// Graph generator for Instance Graphs.
class InstanceGraphGenerator : public GraphGenerator {
public:
  InstanceGraphGenerator(HWOperationRef baseOperation, llvm::raw_ostream *os)
      : GraphGenerator(os), baseOperation(baseOperation) {}

  std::string generateGraphJson() override {
    // Discover all modules to graph.
    forEachOperation(baseOperation, [&](mlir::Operation &op) {
      llvm::TypeSwitch<HWOperationRef>(&op)
          .Case<hw::HWModuleOp>([&](auto module) {
            if (os)
              *os << "Found HWModuleOp: " << module.getName() << "\n";
            moduleMap[module.getName()] = &op;
          })
          .Case<hw::HWModuleExternOp>([&](auto module) {
            if (os)
              *os << "Found HWModuleExternOp: " << module.getName() << "\n";
            moduleMap[module.getName()] = &op;
          })
          .Case<hw::HWModuleGeneratedOp>([&](auto module) {
            if (os)
              *os << "Found HWModuleGeneratedOp: " << module.getName() << "\n";
            moduleMap[module.getName()] = &op;
          })
          .Default([&](auto) {
            if (os)
              *os << "Found unknown module type: "
                  << op.getAttrOfType<StringAttr>(
                           mlir::SymbolTable::getSymbolAttrName())
                         .getValue()
                  << "\n";
            moduleMap[op.getAttrOfType<StringAttr>(
                            mlir::SymbolTable::getSymbolAttrName())
                          .getValue()] = &op;
          });
    });

    // Add modules for processing.
    std::stack<std::tuple<HWOperationRef, std::string, int64_t>> treesToProcess;
    for (auto const &entry : moduleMap) {
      HWOperationRef baseModule = moduleMap[entry.getKey()];
      treesToProcess.push({baseModule, entry.getKey().str(), -1});
    }

    // Process modules.
    while (treesToProcess.size() > 0) {
      HWOperationRef module;
      std::string currentNamespace;
      int64_t parentId;

      std::tie(module, currentNamespace, parentId) = treesToProcess.top();
      treesToProcess.pop();

      // Iterate over sub-operations to find instances.
      bool hasInstances = false;
      int64_t instanceChoiceId = 0;
      forEachOperation(module, [&](mlir::Operation &op) {
        if (auto instance = dyn_cast<InstanceOp>(&op)) {
          // Generate model explorer node
          generateInstanceNode(instance.getReferencedModuleName(), nextNodeId,
                               currentNamespace, parentId);
          // Push back for processing of children
          if (moduleMap.count(instance.getReferencedModuleName()))
            treesToProcess.push({moduleMap[instance.getReferencedModuleName()],
                                 currentNamespace, nextNodeId});
          nextNodeId++;
          hasInstances = true;
        } else if (auto choiceInstance = dyn_cast<InstanceChoiceOp>(&op)) {
          // Loop over all possible instances in a InstanceChoiceOp and generate
          // nodes
          mlir::ArrayAttr moduleNames = choiceInstance.getModuleNamesAttr();
          std::string newNamespace = currentNamespace + "/CHOICE (" +
                                     std::to_string(instanceChoiceId) + ")";
          for (auto attr : moduleNames) {
            llvm::StringRef instanceName =
                cast<FlatSymbolRefAttr>(attr).getAttr().getValue();
            generateInstanceNode(instanceName, nextNodeId, newNamespace,
                                 parentId);
            if (moduleMap.count(instanceName))
              treesToProcess.push(
                  {moduleMap[instanceName], newNamespace, nextNodeId});
            nextNodeId++;
          }
          instanceChoiceId++;
          hasInstances = true;
        }
      });

      // If this is a top level and independant module, we will display
      // appropriate node.
      if (!hasInstances && parentId == -1) {
        llvm::TypeSwitch<HWOperationRef>(module)
            .Case<hw::HWModuleOp>([&](auto mod) {
              generateInstanceNode(llvm::StringRef{"Self-Contained"},
                                   nextNodeId, currentNamespace, -1);
            })
            .Case<hw::HWModuleExternOp>([&](auto mod) {
              generateInstanceNode(llvm::StringRef{"External"}, nextNodeId,
                                   currentNamespace, -1);
            })
            .Case<hw::HWModuleGeneratedOp>([&](auto mod) {
              generateInstanceNode(llvm::StringRef{"External (Generated)"},
                                   nextNodeId, currentNamespace, -1);
            })
            .Default([&](auto mod) {
              generateInstanceNode(llvm::StringRef{"Unknown Module"},
                                   nextNodeId, currentNamespace, -1);
            });
        nextNodeId++;
      }
    }

    // Finish proicessing, wrap output in required Json structure.
    return wrapJson(outputJsonObjects);
  }

protected:
  HWOperationRef baseOperation;

  void generateInstanceNode(llvm::StringRef label, int64_t nextNodeId,
                            std::string &newNamespace, int64_t parentId) {
    llvm::json::Object instanceJson{{"id", std::to_string(nextNodeId)},
                                    {"namespace", newNamespace},
                                    {"label", label}};

    // If a node is top level (has no parent), then it is labelled as such.
    if (parentId != -1) {
      instanceJson["incomingEdges"] = llvm::json::Array{
          llvm::json::Object{{"sourceNodeId", std::to_string(parentId)}}};
      instanceJson["attrs"] = llvm::json::Array{
          llvm::json::Object{{"key", "type"}, {"value", "Lower Level"}}};
    } else {
      instanceJson["attrs"] = llvm::json::Array{
          llvm::json::Object{{"key", "type"}, {"value", "Top Level"}}};
    }

    outputJsonObjects.push_back(std::move(instanceJson));
  }
};

} // end anonymous namespace

namespace circt {
namespace hw {

// Public API functions instantiates the corresponding generator.
std::string MlirToInstanceGraphJson(HWOperationRef baseModule,
                                    llvm::raw_ostream *os) {
  InstanceGraphGenerator generator(baseModule, os);
  return generator.generateGraphJson();
}

} // end namespace hw
} // end namespace circt
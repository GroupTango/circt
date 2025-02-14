//===- HWModelExplorerInterfaces.cpp - Implement HWModelExplorerInterfaces  ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements HWModelExplorerInterfaces related functionality.
//
//===----------------------------------------------------------------------===//


#include "circt/Dialect/HW/HWModelExplorerInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LLVM.h"

#include <stack>

using namespace circt;
using namespace circt::hw;

std::string hw::MlirToInstanceGraphJson(mlir::Operation *baseModule, raw_ostream *os)
{
  //NEED TO DO SOME CHANGES TO DISPLAY MORE INFO MISSING MODULES BETTER (Currently passing nullptr so can't print any info such as missing module names)
  llvm::StringMap<mlir::Operation*> moduleMap;
  std::stack<std::pair<mlir::Operation*, std::string>> modulesToProcess;

  llvm::json::Array outputJsonObjects;
  uint64_t nextNodeId = 0;    

  // Find all top level modules, populate moduleMap
  for (mlir::Region &region : baseModule->getRegions()) {
    for (mlir::Block &block : region.getBlocks()) {
      for (mlir::Operation &op : block.getOperations()) {
        llvm::TypeSwitch<mlir::Operation *>(&op)
            .Case<circt::hw::HWModuleOp>([&](auto module) {
              if (os) *os << "Found HWModuleOp: " << module.getName() << "\n";
              moduleMap[module.getName()] = &op;
            })
            .Case<circt::hw::HWModuleExternOp>([&](auto module) {
              if (os) *os << "Found HWModuleExternOp: " << module.getName() << "\n";                
              moduleMap[module.getName()] = &op;
            })
            .Case<circt::hw::HWModuleGeneratedOp>([&](auto module) {
              if (os) *os << "Found HWModuleGeneratedOp: " << module.getName() << "\n";
              moduleMap[module.getName()] = &op;
            })
            .Default([&](auto) {
              if (os) *os << "Found unknown top level module type: " << op.getName() 
                  << " SKIPPING\n";
            });
      }
    }
  }

  // Start processing Modules to JSON
  for (auto const& x : moduleMap)  {
    mlir::Operation* op = moduleMap[x.getKey()];
    modulesToProcess.push(std::make_pair(op, x.getKey().str()));

    if (os) *os << "Adding top level Module for processing - Name: " << x.getKey()
        << " Type: " << op->getName() 
        << "\n";
  }

  while (modulesToProcess.size() > 0)  {
    std::pair<mlir::Operation*, std::string> nextPair = modulesToProcess.top();
    modulesToProcess.pop();
    auto* module = nextPair.first;

    if (module == nullptr) {
      llvm::json::Object moduleJson {      
        {"id", std::to_string(nextNodeId)},
        {"label", "Unknown Module"},
        {"namespace", nextPair.second}};

      outputJsonObjects.push_back(std::move(moduleJson));
      nextNodeId++;     
      continue;
    }

    bool hasInstances = false;
    uint64_t instanceID = 0;

    for (mlir::Region &region : module->getRegions()) {
      for (mlir::Block &block : region.getBlocks()) {
        for (mlir::Operation &op : block.getOperations()) {
          if (auto instance = dyn_cast<InstanceOp>(&op)) {            
            std::string newNamespace = nextPair.second + "/" + instance.getReferencedModuleName().str() + " (I" + std::to_string(instanceID) + ")";

            auto it = moduleMap.find(instance.getReferencedModuleName());
            if (it != moduleMap.end()) modulesToProcess.push(std::make_pair(moduleMap[instance.getReferencedModuleName()], newNamespace));
            else modulesToProcess.push(std::make_pair(nullptr, newNamespace));
            
            instanceID++;
            hasInstances = true;
          }
          else if (auto choiceInstance = dyn_cast<InstanceChoiceOp>(&op)) {
            mlir::ArrayAttr moduleNames = choiceInstance.getModuleNamesAttr();      
            for  (auto attr : moduleNames)
            {
              mlir::StringRef instanceName = llvm::cast<FlatSymbolRefAttr>(attr).getValue();
              std::string newNamespace = nextPair.second + "/INSTANCE CHOICE (I" + std::to_string(instanceID) + ")/" + instanceName.str();

              auto it = moduleMap.find(instanceName);
              if (it != moduleMap.end()) modulesToProcess.push(std::make_pair(moduleMap[instanceName], newNamespace));
              else modulesToProcess.push(std::make_pair(nullptr, newNamespace));
            }

            instanceID++;
            hasInstances = true;
          }
        }
      }
    }

    // If this is a self contained module, we will display it as a graph node.
    if (!hasInstances)
    {
      llvm::json::Object moduleJson {
        {"id", std::to_string(nextNodeId)},
        {"namespace", nextPair.second}};
      
      llvm::TypeSwitch<mlir::Operation *>(module)
            .Case<circt::hw::HWModuleOp>([&](auto module) {
              moduleJson["label"] = "Self-Contained";
            })
            .Case<circt::hw::HWModuleExternOp>([&](auto module) {
              moduleJson["label"] = "External";
            })
            .Case<circt::hw::HWModuleGeneratedOp>([&](auto module) {              
              moduleJson["label"] = "External (Generated)";
            })
            .Default([&](auto) {              
              moduleJson["label"] = "UNKNOWN ERROR";
            });

      outputJsonObjects.push_back(std::move(moduleJson));
      nextNodeId++;        
    }
  }    

  // Do some final wraps of our JSON Node Array, as needed by Model Explorer
  llvm::json::Object graphWrapper {
    {"id", std::to_string(nextNodeId)},
    {"nodes", std::move(outputJsonObjects)}};

  llvm::json::Array graphArrayWrapper;
  graphArrayWrapper.push_back(std::move(graphWrapper));

  llvm::json::Object fileWrapper {
    {"label", "model.json"},
    {"subgraphs", std::move(graphArrayWrapper)}};

  llvm::json::Array fileArrayWrapper{llvm::json::Value(std::move(fileWrapper))};

  // Output final JSON
  std::string jsonString; 
  llvm::raw_string_ostream jsonStream(jsonString);
  jsonStream << llvm::json::Value(std::move(fileArrayWrapper));

  return jsonString;
}

std::string hw::MlirToOperationGraphJson(mlir::Operation *baseModule, raw_ostream *os)
{
  return "EMPTY";
}
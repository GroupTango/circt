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
#include "circt/Dialect/HW/HWModuleGraph.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LLVM.h"

#include <stack>

using namespace circt;
using namespace circt::hw;

// Use the GraphTraits specialized for circt::hw::HWModuleOp to traverse the
// module graph.
using NodeRef = mlir::Operation *;
using HWModuleOpGraphTraits = llvm::GraphTraits<hw::HWModuleOp>;
using HWModuleOpJSONGraphTraits = hw::JSONGraphTraits<hw::HWModuleOp>;

std::string hw::MlirToInstanceGraphJson(mlir::Operation *baseModule, raw_ostream *os) {
  llvm::StringMap<mlir::Operation*> moduleMap;
  std::stack<std::pair<mlir::Operation*, std::string>> modulesToProcess;

  llvm::json::Array outputJsonObjects;
  uint64_t nextNodeId = 0;    

  // Find all top level modules, populate moduleMap
  for (mlir::Region &region : baseModule->getRegions()) {
    for (mlir::Block &block : region.getBlocks()) {
      for (mlir::Operation &op : block.getOperations()) {
        llvm::TypeSwitch<mlir::Operation *>(&op)
            .Case<hw::HWModuleOp>([&](auto module) {
              if (os) *os << "Found HWModuleOp: " << module.getName() << "\n";
              moduleMap[module.getName()] = &op;
            })
            .Case<hw::HWModuleExternOp>([&](auto module) {
              if (os) *os << "Found HWModuleExternOp: " << module.getName() << "\n";                
              moduleMap[module.getName()] = &op;
            })
            .Case<hw::HWModuleGeneratedOp>([&](auto module) {
              if (os) *os << "Found HWModuleGeneratedOp: " << module.getName() << "\n";
              moduleMap[module.getName()] = &op;
            })
            .Default([&](auto) {
              if (os) *os << "Found unknown module: " << op.getName() << "\n";
              modulesToProcess.push({nullptr, op.getName().getStringRef().str()});
            });
      }
    }
  }

  for (auto const& x : moduleMap)  {
    mlir::Operation* op = moduleMap[x.getKey()];
    modulesToProcess.push({op, x.getKey().str()});

    if (os) *os << "Adding top level Module for processing - Name: " << x.getKey()
        << " Type: " << op->getName() 
        << "\n";
  }

  while (modulesToProcess.size() > 0)  {
    std::pair<mlir::Operation *, std::string> nextPair = modulesToProcess.top();
    modulesToProcess.pop();
    mlir::Operation *module = nextPair.first;

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
            if (it != moduleMap.end()) 
              modulesToProcess.push({moduleMap[instance.getReferencedModuleName()], newNamespace});
            else 
              modulesToProcess.push({nullptr, newNamespace});
            
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
              if (it != moduleMap.end()) 
                modulesToProcess.push({moduleMap[instanceName], newNamespace});
              else 
                modulesToProcess.push({nullptr, newNamespace});
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
            .Case<hw::HWModuleOp>([&](auto module) {
              moduleJson["label"] = "Self-Contained";
            })
            .Case<hw::HWModuleExternOp>([&](auto module) {
              moduleJson["label"] = "External";
            })
            .Case<hw::HWModuleGeneratedOp>([&](auto module) {              
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

std::string getUniqueId(mlir::Operation *node, uint64_t &nextNodeId, std::string namesp) {
  if (node == nullptr)
    return namesp + "_" + std::to_string(nextNodeId++);

  return namesp + "_" +
          std::to_string(
              mlir::cast<IntegerAttr>(node->getAttr("hw.unique_id")).getInt());
}

void populateIncomingEdges(hw::HWModuleOp module, llvm::DenseMap<NodeRef, std::vector<NodeRef>> &incomingEdges) {
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

llvm::json::Array getIncomingEdges(NodeRef node, hw::HWModuleOp module, llvm::DenseMap<NodeRef, std::vector<NodeRef>> &incomingEdges, uint64_t &nextNodeId, std::string namesp) {
  llvm::json::Array edges;
  for (NodeRef parent : incomingEdges[node]) {
    edges.push_back(
        llvm::json::Object{{"sourceNodeId", getUniqueId(parent, nextNodeId, namesp)},
                            {"sourceNodeOutputId", "0"},
                            {"targetNodeInputId", "0"}});
  }
  return edges;
}

std::string hw::MlirToOperationGraphJson(mlir::Operation *baseModule, raw_ostream *os) {  
  llvm::StringMap<mlir::Operation *> moduleMap;
  std::stack<std::pair<mlir::Operation *, std::string>> modulesToProcess;

  llvm::json::Array outputJsonObjects;
  HWModuleOpJSONGraphTraits jsonGraphTraits;
  
  uint64_t nextNodeId = 0;
  llvm::DenseMap<NodeRef, std::vector<NodeRef>> incomingEdges;

  // Assign unique IDs to all operations
  int64_t counter = 0;
  baseModule->walk([&](mlir::Operation *op) {
    auto id = mlir::IntegerAttr::get(
        mlir::IntegerType::get(op->getContext(), 64), counter++);
    op->setAttr("hw.unique_id", id);
  });

  // Find all top level modules, populate moduleMap and incomingEdges
  for (mlir::Region &region : baseModule->getRegions()) {
    for (mlir::Block &block : region.getBlocks()) {
      for (mlir::Operation &op : block.getOperations()) {
        llvm::TypeSwitch<mlir::Operation *>(&op)
            .Case<hw::HWModuleOp>([&](auto module) {
              if (os) *os << "Found HWModuleOp: " << module.getName() << "\n";
              moduleMap[module.getName()] = &op;
            })
            .Case<hw::HWModuleExternOp>([&](auto module) {
              if (os) *os << "Found HWModuleExternOp: " << module.getName() << "\n";   
              modulesToProcess.push({nullptr, module.getName().str()});
            })
            .Case<hw::HWModuleGeneratedOp>([&](auto module) {
              if (os) *os << "Found HWModuleGeneratedOp: " << module.getName() << "\n";
              modulesToProcess.push({nullptr, module.getName().str()});
            })
            .Default([&](auto) {
              if (os) *os << "Found unknown module: " << op.getName() << "\n";
              modulesToProcess.push({nullptr, op.getName().getStringRef().str()});
            });
      }
    }
  }

  for (auto const& x : moduleMap)  {
    mlir::Operation* op = moduleMap[x.getKey()];
    modulesToProcess.push({op, x.getKey().str()});

    if (os) *os << "Adding top level Module for processing - Name: " << x.getKey()
        << " Type: " << op->getName() 
        << "\n";
  }

  while (modulesToProcess.size() > 0) {
    std::pair<mlir::Operation *, std::string> nextPair = modulesToProcess.top();
    modulesToProcess.pop();
    mlir::Operation *module = nextPair.first;

    if (module == nullptr) {
      llvm::json::Object moduleJson {      
        {"id", getUniqueId(module, nextNodeId, nextPair.second)},
        {"label", "Unknown Module"},
        {"namespace", nextPair.second}};

      outputJsonObjects.push_back(std::move(moduleJson));
      continue;
    }

    bool hasInstances = false;

    for (mlir::Region &region : module->getRegions()) {
      for (mlir::Block &block : region.getBlocks()) {
        for (mlir::Operation &op : block.getOperations()) {

          NodeRef node = &op;
          hw::HWModuleOp moduleOp = mlir::dyn_cast<hw::HWModuleOp>(module);

          llvm::json::Object jsonObj{
                {"label", jsonGraphTraits.getNodeLabel(node, moduleOp)},
                {"attrs", jsonGraphTraits.getNodeAttributes(node, moduleOp)},
                {"id", getUniqueId(node, nextNodeId, nextPair.second)},
                {"namespace", nextPair.second}};

          if (auto instanceOp = dyn_cast<InstanceOp>(op)) {  
            if (os) *os << "Found InstanceOp: " << instanceOp.getReferencedModuleName()
                << "\n";

            hasInstances = true;

            auto refModuleName = instanceOp.getReferencedModuleName();
            std::string newNamespace =nextPair.second + "/" + refModuleName.str();

            auto it = moduleMap.find(refModuleName);
            if (it != moduleMap.end())
              modulesToProcess.push({moduleMap[refModuleName], newNamespace});
            else
              modulesToProcess.push({nullptr, newNamespace});

            // inter-module dependency, so we want (module -> hw.instance)
            jsonObj["incomingEdges"] = llvm::json::Array{llvm::json::Object{
                {"sourceNodeId", getUniqueId(moduleMap[refModuleName], nextNodeId, nextPair.second + "/" + refModuleName.str())},
                {"sourceNodeOutputId", "0"},
                {"targetNodeInputId", "0"}}};
          } 
          else {
            // intra-module dependency, get from module graph
            jsonObj["incomingEdges"] =
                getIncomingEdges(&op, moduleOp, incomingEdges, nextNodeId, nextPair.second);
          }

          outputJsonObjects.push_back(std::move(jsonObj));
        }
      }
    }

    // If this is a self contained module, we will display it as a graph
    // node.
    if (!hasInstances) {
      HWModuleOp moduleOp = mlir::dyn_cast<circt::hw::HWModuleOp>(module);

      llvm::json::Object moduleJson {
        {"id", getUniqueId(module, nextNodeId, nextPair.second)},
        {"namespace", nextPair.second},
        {"label", moduleOp.getNameAttr().getValue()},
        {"attrs", jsonGraphTraits.getNodeAttributes(module, moduleOp)},
        {"incomingEdges", getIncomingEdges(module, moduleOp, incomingEdges, nextNodeId, nextPair.second)}};

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

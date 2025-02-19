//===- HWModelExplorer.cpp - Model graph JSON generation ------------*- C++
//-*-===//
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

// Use the GraphTraits specialized for circt::hw::HWModuleOp to traverse the
// module graph.
using NodeRef = mlir::Operation *;
using HWModuleOpGraphTraits = llvm::GraphTraits<hw::HWModuleOp>;
using HWModuleOpJSONGraphTraits = hw::JSONGraphTraits<hw::HWModuleOp>;

namespace {

template <typename Fn>
void forEachOperation(mlir::Operation *op, Fn f) {
  for (mlir::Region &region : op->getRegions())
    for (mlir::Block &block : region.getBlocks())
      for (mlir::Operation &childOp : block.getOperations())
        f(childOp);
}

class GraphGenerator {
public:
  GraphGenerator(mlir::Operation *baseModule, llvm::raw_ostream *os)
      : baseModule(baseModule), os(os), nextNodeId(0) {}

  virtual ~GraphGenerator() = default;

  // Main entry point: initialize, process modules, and wrap the output.
  std::string generateGraphJson() {
    initializeModules();
    processModules();
    return wrapJson(outputJsonObjects);
  }

protected:
  mlir::Operation *baseModule;
  llvm::raw_ostream *os;
  uint64_t nextNodeId;
  llvm::StringMap<mlir::Operation *> moduleMap;
  std::stack<std::pair<mlir::Operation *, std::string>> modulesToProcess;
  llvm::json::Array outputJsonObjects;

  // Discover top-level modules from the base module.
  virtual void initializeModules() {
    forEachOperation(baseModule, [&](mlir::Operation &op) {
      llvm::TypeSwitch<mlir::Operation *>(&op)
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
              *os << "Found unknown module: " << op.getName() << "\n";
            // If the module type is unknown, enqueue it with a null
            // pointer.
            modulesToProcess.push({nullptr, op.getName().getStringRef().str()});
          });
    });
    // Enqueue discovered modules for further processing.
    for (auto const &entry : moduleMap) {
      mlir::Operation *op = moduleMap[entry.getKey()];
      modulesToProcess.push({op, entry.getKey().str()});
      if (os)
        *os << "Adding top level Module for processing - Name: "
            << entry.getKey() << " Type: " << op->getName() << "\n";
    }
  }

  // Process all enqueued modules.
  virtual void processModules() {
    while (!modulesToProcess.empty()) {
      auto nextPair = modulesToProcess.top();
      modulesToProcess.pop();
      processModule(nextPair.first, nextPair.second);
    }
  }

  // Pure virtual: process a module given its operation pointer and namespace.
  virtual void processModule(mlir::Operation *module,
                             const std::string &ns) = 0;

  std::string wrapJson(llvm::json::Array nodes) {
    llvm::json::Object graphWrapper{{"id", std::to_string(nextNodeId)},
                                    {"nodes", std::move(nodes)}};
    llvm::json::Array graphArrayWrapper;
    graphArrayWrapper.push_back(std::move(graphWrapper));
    llvm::json::Object fileWrapper{{"label", "model.json"},
                                   {"subgraphs", std::move(graphArrayWrapper)}};
    llvm::json::Array fileArrayWrapper{
        llvm::json::Value(std::move(fileWrapper))};

    std::string jsonString;
    llvm::raw_string_ostream jsonStream(jsonString);
    jsonStream << llvm::json::Value(std::move(fileArrayWrapper));
    return jsonString;
  }

  // Generate a unique ID for a node using its existing attribute if present.
  std::string getUniqueId(mlir::Operation *node, const std::string &ns) {
    if (!node)
      return ns + "_" + std::to_string(nextNodeId++);
    return ns + "_" +
           std::to_string(
               mlir::cast<IntegerAttr>(node->getAttr("hw.unique_id")).getInt());
  }
};

// Graph generator for Instance Graphs.
class InstanceGraphGenerator : public GraphGenerator {
public:
  InstanceGraphGenerator(mlir::Operation *baseModule, llvm::raw_ostream *os)
      : GraphGenerator(baseModule, os) {}

protected:
  void processModule(mlir::Operation *module, const std::string &ns) override {
    if (!module) {
      llvm::json::Object moduleJson{{"id", std::to_string(nextNodeId)},
                                    {"label", "Unknown Module"},
                                    {"namespace", ns}};
      outputJsonObjects.push_back(std::move(moduleJson));
      nextNodeId++;
      return;
    }

    bool hasInstances = false;
    uint64_t instanceID = 0;

    // Iterate over sub-operations to find instances.
    forEachOperation(module, [&](mlir::Operation &op) {
      if (auto instance = dyn_cast<InstanceOp>(&op)) {
        std::string newNamespace = ns + "/" +
                                   instance.getReferencedModuleName().str() +
                                   " (I" + std::to_string(instanceID) + ")";
        if (moduleMap.count(instance.getReferencedModuleName()))
          modulesToProcess.push(
              {moduleMap[instance.getReferencedModuleName()], newNamespace});
        else
          modulesToProcess.push({nullptr, newNamespace});
        instanceID++;        
        hasInstances = true;
      } else if (auto choiceInstance = dyn_cast<InstanceChoiceOp>(&op)) {
        mlir::ArrayAttr moduleNames = choiceInstance.getModuleNamesAttr();
        for (auto attr : moduleNames) {
          mlir::StringRef instanceName =
              cast<FlatSymbolRefAttr>(attr).getValue();
          std::string newNamespace = ns + "/INSTANCE CHOICE (I" +
                                     std::to_string(instanceID) + ")/" +
                                     instanceName.str();
          if (moduleMap.count(instanceName))
            modulesToProcess.push({moduleMap[instanceName], newNamespace});
          else
            modulesToProcess.push({nullptr, newNamespace});
        }
        instanceID++;
        hasInstances = true;
      }
    });

    // If this is a independant module, we will display it as a graph node.
    if (!hasInstances)
    {      
      llvm::json::Object moduleJson{{"id", std::to_string(nextNodeId)},
                                    {"namespace", ns}};
      llvm::TypeSwitch<mlir::Operation *>(module)
          .Case<hw::HWModuleOp>(
              [&](auto mod) { moduleJson["label"] = "Self-Contained"; })
          .Case<hw::HWModuleExternOp>(
              [&](auto mod) { moduleJson["label"] = "External"; })
          .Case<hw::HWModuleGeneratedOp>(
              [&](auto mod) { moduleJson["label"] = "External (Generated)"; })
          .Default([&](auto mod) { moduleJson["label"] = "Unknown Module"; });
      outputJsonObjects.push_back(std::move(moduleJson));
      nextNodeId++;
    }
  }
};

// Graph generator for Operation Graphs.
class OperationGraphGenerator : public GraphGenerator {
public:
  OperationGraphGenerator(mlir::Operation *baseModule, llvm::raw_ostream *os)
      : GraphGenerator(baseModule, os) {}

protected:
  llvm::DenseMap<NodeRef, std::vector<NodeRef>> incomingEdges;
  HWModuleOpJSONGraphTraits jsonGraphTraits;

  // In addition to module discovery, assign unique IDs to all operations and
  // populate intra-module dependency edges.
  void initializeModules() override {
    int64_t counter = 0;
    baseModule->walk([&](mlir::Operation *op) {
      auto id = mlir::IntegerAttr::get(
          mlir::IntegerType::get(op->getContext(), 64), counter++);
      op->setAttr("hw.unique_id", id);
    });

    forEachOperation(baseModule, [&](mlir::Operation &op) {
      llvm::TypeSwitch<mlir::Operation *>(&op)
          .Case<hw::HWModuleOp>([&](auto module) {
            if (os)
              *os << "Found HWModuleOp: " << module.getName() << "\n";
            moduleMap[module.getName()] = &op;
            populateIncomingEdges(module, incomingEdges);
          })
          .Case<hw::HWModuleExternOp>([&](auto module) {
            if (os)
              *os << "Found HWModuleExternOp: " << module.getName() << "\n";
            modulesToProcess.push({nullptr, module.getName().str()});
          })
          .Case<hw::HWModuleGeneratedOp>([&](auto module) {
            if (os)
              *os << "Found HWModuleGeneratedOp: " << module.getName() << "\n";
            modulesToProcess.push({nullptr, module.getName().str()});
          })
          .Default([&](auto) {
            if (os)
              *os << "Found unknown module: " << op.getName() << "\n";
            modulesToProcess.push({nullptr, op.getName().getStringRef().str()});
          });
    });

    for (auto const &entry : moduleMap) {
      mlir::Operation *op = moduleMap[entry.getKey()];
      modulesToProcess.push({op, entry.getKey().str()});
      if (os)
        *os << "Adding top level Module for processing - Name: "
            << entry.getKey() << " Type: " << op->getName() << "\n";
    }
  }

  // Process a module by iterating over its operations and generating JSON
  // nodes.
  void processModule(mlir::Operation *module, const std::string &ns) override {
    if (!module) {
      llvm::json::Object moduleJson{{"id", getUniqueId(module, ns)},
                                    {"label", "Unknown Module"},
                                    {"namespace", ns}};
      outputJsonObjects.push_back(std::move(moduleJson));
      return;
    }

    forEachOperation(
        module, [&](mlir::Operation &op) {
          NodeRef node = &op;
          auto moduleOp = mlir::dyn_cast<hw::HWModuleOp>(module);
          llvm::json::Object jsonObj{
              {"label", jsonGraphTraits.getNodeLabel(node, moduleOp)},
              {"attrs", jsonGraphTraits.getNodeAttributes(node, moduleOp)},
              {"id", getUniqueId(node, ns)},
              {"namespace", ns}};

          if (auto instanceOp = dyn_cast<InstanceOp>(&op)) {
            if (os)
              *os << "Found InstanceOp: "
                  << instanceOp.getReferencedModuleName() << "\n";
            std::string refModuleName =
                instanceOp.getReferencedModuleName().str();
            std::string newNamespace = ns + "/" + refModuleName;
            if (moduleMap.count(refModuleName))
              modulesToProcess.push({moduleMap[refModuleName], newNamespace});
            else
              modulesToProcess.push({nullptr, newNamespace});
            // Inter-module dependency.
            jsonObj["incomingEdges"] = llvm::json::Array{llvm::json::Object{
                {"sourceNodeId", getUniqueId(moduleMap[refModuleName],
                                             ns + "/" + refModuleName)},
                {"sourceNodeOutputId", "0"},
                {"targetNodeInputId", "0"}}};
          } else {
            // Intra-module dependency.
            jsonObj["incomingEdges"] = getIncomingEdges(node, moduleOp, ns);
          }
          outputJsonObjects.push_back(std::move(jsonObj));
        });
        
    // Also add a JSON node for the module itself.
    auto moduleOp = mlir::dyn_cast<hw::HWModuleOp>(module);
    llvm::json::Object moduleJson{
        {"id", getUniqueId(module, ns)},
        {"namespace", ns},
        {"label", moduleOp.getNameAttr().getValue()},
        {"attrs", jsonGraphTraits.getNodeAttributes(module, moduleOp)},
        {"incomingEdges", getIncomingEdges(module, moduleOp, ns)}};
    outputJsonObjects.push_back(std::move(moduleJson));
  }

  // Generate incoming edge JSON for a node.
  llvm::json::Array getIncomingEdges(NodeRef node, hw::HWModuleOp module,
                                     const std::string &ns) {
    llvm::json::Array edges;
    for (NodeRef parent : incomingEdges[node]) {
      edges.push_back(
          llvm::json::Object{{"sourceNodeId", getUniqueId(parent, ns)},
                             {"sourceNodeOutputId", "0"},
                             {"targetNodeInputId", "0"}});
    }
    return edges;
  }

  // Populate incoming edge relationships using GraphTraits.
  void populateIncomingEdges(
      hw::HWModuleOp module,
      llvm::DenseMap<NodeRef, std::vector<NodeRef>> &edgesMap) {
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
        edgesMap[child].push_back(current);
        nodesToVisit.push(child);
      }
    }
  }
};

} // end anonymous namespace

namespace circt {
namespace hw {

// Public API functions now simply instantiate the corresponding generator.
std::string MlirToInstanceGraphJson(mlir::Operation *baseModule,
                                    llvm::raw_ostream *os) {
  InstanceGraphGenerator generator(baseModule, os);
  return generator.generateGraphJson();
}

std::string MlirToOperationGraphJson(mlir::Operation *baseModule,
                                     llvm::raw_ostream *os) {
  OperationGraphGenerator generator(baseModule, os);
  return generator.generateGraphJson();
}

} // end namespace hw
} // end namespace circt
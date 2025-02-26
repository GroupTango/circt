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
#include <vector>

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
  GraphGenerator(llvm::raw_ostream *os)
      : os(os), nextNodeId(0) {}

  virtual ~GraphGenerator() = default;

  // Main entry point: initialize, process modules, and wrap the output.
  virtual std::string generateGraphJson() = 0;

protected:
  llvm::raw_ostream *os;
  llvm::StringMap<mlir::Operation *> moduleMap;  
  int64_t nextNodeId;
  llvm::json::Array outputJsonObjects;

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
  InstanceGraphGenerator(mlir::Operation *baseOperation, llvm::raw_ostream *os)
      : GraphGenerator(os), baseOperation(baseOperation) {}
  
  std::string generateGraphJson() override {
    // Discover all modules to graph.
    forEachOperation(baseOperation, [&](mlir::Operation &op) {
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
              *os << "Found unknown module type: " << op.getName() << "\n";            
            moduleMap[op.getName().getStringRef()] = &op;
          });
    });

    // Process modules.
    std::stack<std::tuple<mlir::Operation *, std::string, int64_t>> treesToProcess;

    for (auto const &entry : moduleMap) {
      mlir::Operation *baseModule = moduleMap[entry.getKey()];
      treesToProcess.push({baseModule, entry.getKey().str(), -1});
      //if (os)
      //  *os << "Queuing top level Module - Name: "
      //      << entry.getKey() << " Type: " << baseModule->getName() << "\n";   
    }

    while (treesToProcess.size() > 0)
    {
      mlir::Operation *module;
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
          generateInstanceNode(instance.getReferencedModuleName(), nextNodeId, currentNamespace, parentId);
          // Push back for processing of children
          if (moduleMap.count(instance.getReferencedModuleName()))
            treesToProcess.push({moduleMap[instance.getReferencedModuleName()], currentNamespace, nextNodeId});
          nextNodeId++;        
          hasInstances = true;
        } else if (auto choiceInstance = dyn_cast<InstanceChoiceOp>(&op)) {
          mlir::ArrayAttr moduleNames = choiceInstance.getModuleNamesAttr();
          std::string newNamespace = currentNamespace + "/CHOICE (" +
                                      std::to_string(instanceChoiceId) + ")";
          for (auto attr : moduleNames) {
            llvm::StringRef instanceName = cast<FlatSymbolRefAttr>(attr).getAttr().getValue();
            generateInstanceNode(instanceName, nextNodeId, newNamespace, parentId);
            if (moduleMap.count(instanceName))
              treesToProcess.push({moduleMap[instanceName], newNamespace, nextNodeId});      
            nextNodeId++;   
          }
          instanceChoiceId++;
          hasInstances = true;
        }
      });

      // If this is a top level and independant module, we will display appropriate node.
      if (!hasInstances && parentId == -1)
      {       
        llvm::TypeSwitch<mlir::Operation *>(module)
            .Case<hw::HWModuleOp>(
                [&](auto mod) { generateInstanceNode(llvm::StringRef {"Self-Contained"}, nextNodeId, currentNamespace, -1); })
            .Case<hw::HWModuleExternOp>(
                [&](auto mod) { generateInstanceNode(llvm::StringRef {"External"}, nextNodeId, currentNamespace, -1); })
            .Case<hw::HWModuleGeneratedOp>(
                [&](auto mod) { generateInstanceNode(llvm::StringRef {"External (Generated)"}, nextNodeId, currentNamespace, -1); })
            .Default([&](auto mod) 
                              { generateInstanceNode(llvm::StringRef {"Unknown Module"}, nextNodeId, currentNamespace, -1); });        
        nextNodeId++;
      }
    }   

    return wrapJson(outputJsonObjects);
  }

protected:
  mlir::Operation *baseOperation;

  void generateInstanceNode(llvm::StringRef label, int64_t nextNodeId, std::string &newNamespace, int64_t parentId) {
    llvm::json::Object instanceJson{{"id", std::to_string(nextNodeId)},
                                    {"namespace", newNamespace},
                                    {"label", label}};
    if (parentId != -1) {
      instanceJson["incomingEdges"] = llvm::json::Array{llvm::json::Object{
        {"sourceNodeId", std::to_string(parentId)}}};    
      instanceJson["attrs"] = llvm::json::Array {
        llvm::json::Object{{"key", "type"}, {"value", "Lower Level"}}};
    } else {
       instanceJson["attrs"] = llvm::json::Array {
        llvm::json::Object{{"key", "type"}, {"value", "Top Level"}}};
    }
    outputJsonObjects.push_back(std::move(instanceJson));  
  }
};

// Graph generator for Operation Graphs.
class OperationGraphGenerator : public GraphGenerator {
public:
  OperationGraphGenerator(mlir::Operation *baseModule, llvm::raw_ostream *os)
      : GraphGenerator(os), baseModule(baseModule) {}

  std::string generateGraphJson() override {
    initializeModules();
    while (!modulesToProcess.empty()) {
      auto nextPair = modulesToProcess.top();
      modulesToProcess.pop();
      processModule(nextPair.first, nextPair.second);
    }
    return wrapJson(outputJsonObjects);
  }

protected:  
  mlir::Operation *baseModule;
  std::stack<std::pair<mlir::Operation *, std::string>> modulesToProcess;

  llvm::DenseMap<NodeRef, std::vector<NodeRef>> incomingEdges;
  llvm::DenseMap<NodeRef, std::vector<std::string>> incomingInputEdges;
  HWModuleOpJSONGraphTraits jsonGraphTraits;

  /// Discover all top-level modules, assign unique IDs to all operations via
  /// @c hw.unique_id , populate intra-module dependency edges, and process
  /// input nodes and dependency edges.
  void initializeModules() {
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
      std::string namespaceStr = entry.getKey().str();
      modulesToProcess.push({op, namespaceStr});
      processInputs(op, namespaceStr);
      if (os)
        *os << "Adding top level Module for processing - Name: "
            << entry.getKey() << " Type: " << op->getName() << "\n";
    }
  }

  // Process a module by iterating over its operations and generating JSON
  // nodes.
  void processModule(mlir::Operation *module, const std::string &ns) {
    if (!module) {
      llvm::json::Object moduleJson{{"id", getUniqueId(module, ns)},
                                    {"label", "Unknown Module"},
                                    {"namespace", ns}};
      outputJsonObjects.push_back(std::move(moduleJson));
      return;
    }

    // Process child operations
    forEachOperation(
        module, [&](mlir::Operation &op) {   
          NodeRef node = &op;   
          if (auto instanceOp = dyn_cast<InstanceOp>(&op)) {
            if (os)
              *os << "Found InstanceOp: "
                  << instanceOp.getReferencedModuleName() << "\n";
            std::string refModuleName =
                instanceOp.getReferencedModuleName().str();
            
            // TEMP FIX: Do not do recursive modules. Instead, just do an hw.instance(ReferencedModuleName)
            // and point to it the correct inputs, as well as a copy of the referenced module.

            // std::string newNamespace = ns + "/" + refModuleName;
            // if (moduleMap.count(refModuleName))
            //   modulesToProcess.push({moduleMap[refModuleName], newNamespace});
            // else
            //   modulesToProcess.push({nullptr, newNamespace});

            llvm::json::Object jsonObj{
              {"label", refModuleName},
              {"attrs", llvm::json::Array({
                llvm::json::Object({{"key", "type"}, {"value", "instance"}})
              })},
              {"id", getUniqueId(&op, ns)},
              {"namespace", ns},
              {"incomingEdges", getIncomingEdges(node, ns)}};
            outputJsonObjects.push_back(std::move(jsonObj));
          } else {
            // Intra-module dependency.
            auto moduleOp = mlir::dyn_cast<hw::HWModuleOp>(module);
            llvm::json::Object jsonObj{
              {"label", jsonGraphTraits.getNodeLabel(node, moduleOp)},
              {"attrs", jsonGraphTraits.getNodeAttributes(node, moduleOp)},
              {"id", getUniqueId(node, ns)},
              {"namespace", ns},
              {"incomingEdges", getIncomingEdges(node, ns)}};
            outputJsonObjects.push_back(std::move(jsonObj));
          }          
        });
        
  }

  // Generate incoming edge JSON for a node.
  llvm::json::Array getIncomingEdges(NodeRef node,
                                     const std::string &ns) {
    llvm::json::Array edges;
    for (NodeRef parent : incomingEdges[node]) {
      edges.push_back(
          llvm::json::Object{{"sourceNodeId", getUniqueId(parent, ns)},
                             {"sourceNodeOutputId", "0"},
                             {"targetNodeInputId", "0"}});
    }
    for (std::string inputParent : incomingInputEdges[node]) {
      edges.push_back(
          llvm::json::Object{{"sourceNodeId", ns + "_" + inputParent},
                             {"sourceNodeOutputId", "0"},
                             {"targetNodeInputId", "0"}}
      );
    }
    if (auto instanceOp = dyn_cast<hw::InstanceOp>(node)) {
      edges.push_back(
        llvm::json::Object{{"sourceNodeId", instanceOp.getReferencedModuleName().str()},
                           {"sourceNodeOutputId", "0"},
                           {"targetNodeInputId", "0"}}
    );
    }
    return edges;
  }

  // For HWModuleOp: Process input ports
  void processInputs(
      mlir::Operation *module,
      const std::string &ns) {
    auto moduleOp = mlir::dyn_cast<circt::hw::HWModuleOp>(module);
    circt::hw::ModulePortInfo iports(moduleOp.getPortList());

    // generate input nodes as well as their outgoing edges
    for (auto [info, arg] :
          llvm::zip(iports.getInputs(), moduleOp.getBodyBlock()->getArguments())) {
      std::string inputName = info.getName().str();
      std::string inputId = ns + "_" + inputName;
      llvm::json::Object jsonObj{
        {"label", inputName},
        {"attrs", jsonGraphTraits.getInputNodeAttributes()},
        {"id", inputId},
        {"namespace", ns}};
      outputJsonObjects.push_back(std::move(jsonObj));
      for (auto *user : arg.getUsers()) {
        incomingInputEdges[user].push_back(inputName);
      }
    }
    for (auto info : iports.getOutputs()) {
      std::string outputName = info.getName().str();
    }

    // TODO: figure out a compromise for generating edges from parent module ops to child input nodes
    
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

// Graph generator for Diff Graphs.
class InstanceDiffGraphGenerator : GraphGenerator {
public:
  InstanceDiffGraphGenerator(mlir::Operation *baseOriginalModule, mlir::Operation *baseNewModule, llvm::raw_ostream *os)
      : GraphGenerator(os), baseOriginalModule(baseOriginalModule), baseNewModule(baseNewModule) {}
  
  std::string generateGraphJson() override {
    /*// Discover all modules to graph.
    forEachOperation(baseOperation, [&](mlir::Operation &op) {
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
              *os << "Found unknown module type: " << op.getName() << "\n";            
            moduleMap[op.getName().getStringRef()] = &op;
          });
    });

    // Process modules.
    std::stack<std::tuple<mlir::Operation *, std::string, int64_t>> treesToProcess;

    for (auto const &entry : moduleMap) {
      mlir::Operation *baseModule = moduleMap[entry.getKey()];
      treesToProcess.push({baseModule, entry.getKey().str(), -1});

      //if (os)
      //  *os << "Queuing top level Module - Name: "
      //      << entry.getKey() << " Type: " << baseModule->getName() << "\n";   
    }

    while (treesToProcess.size() > 0)
    {
      mlir::Operation *module;
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
          generateInstanceNode(instance.getReferencedModuleName(), nextNodeId, currentNamespace, parentId);
          // Push back for processing of children
          if (moduleMap.count(instance.getReferencedModuleName()))
            treesToProcess.push({moduleMap[instance.getReferencedModuleName()], currentNamespace, nextNodeId});
          nextNodeId++;        
          hasInstances = true;
        } else if (auto choiceInstance = dyn_cast<InstanceChoiceOp>(&op)) {
          mlir::ArrayAttr moduleNames = choiceInstance.getModuleNamesAttr();
          std::string newNamespace = currentNamespace + "/CHOICE (" +
                                      std::to_string(instanceChoiceId) + ")";
          for (auto attr : moduleNames) {
            llvm::StringRef instanceName = cast<FlatSymbolRefAttr>(attr).getAttr().getValue();
            generateInstanceNode(instanceName, nextNodeId, newNamespace, parentId);
            if (moduleMap.count(instanceName))
              treesToProcess.push({moduleMap[instanceName], newNamespace, nextNodeId});      
            nextNodeId++;   
          }
          instanceChoiceId++;
          hasInstances = true;
        }
      });

      // If this is a top level and independant module, we will display appropriate node.
      if (!hasInstances && parentId == -1)
      {      
        llvm::json::Object moduleJson{{"id", std::to_string(nextNodeId)},
                                      {"namespace", currentNamespace},
                                      {"attrs", llvm::json::Array {
                                        llvm::json::Object{
                                          {"key", "type"}, 
                                          {"value", "Top Level"}}}}};        
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
    */
    return wrapJson(outputJsonObjects);    
  }

protected:
  mlir::Operation *baseOriginalModule;
  mlir::Operation *baseNewModule;  
  std::stack<std::pair<mlir::Operation *, int64_t>> modulesToProcess;
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

std::string MlirInstanceDiffGraphJson(mlir::Operation *baseOriginalModule, mlir::Operation *baseNewModule,
                                     llvm::raw_ostream *os) {\
  InstanceDiffGraphGenerator generator(baseOriginalModule, baseNewModule, os);
  return generator.generateGraphJson();
}

} // end namespace hw
} // end namespace circt
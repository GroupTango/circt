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

// Shallow iteration over all operations in the top-level module.
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
    if (ns.empty()) 
      return NULL;
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

    while (!modulesToDrawIO.empty()) {
      auto [module, ns, parent_instance, parent_ns] = modulesToDrawIO.top(); // requires C++17
      if (parent_ns.empty()) {
        processGraphJson();
        resetIOEdgeMaps();
      }
      modulesToDrawIO.pop();
      drawModuleIO(module, ns, parent_instance, parent_ns);
    }
    processGraphJson();
    return wrapJson(outputJsonObjects);
  }

  void processGraphJson() {
    while (!modulesToProcess.empty()) {
      auto [module, ns, parent_instance, parent_ns] = modulesToProcess.front();
      modulesToProcess.pop_front();
      processModule(module, ns, parent_instance, parent_ns);
    }
  }

protected:  
  mlir::Operation *baseModule;
  std::stack<std::tuple<mlir::Operation *, std::string, mlir::Operation *, std::string>> modulesToDrawIO;
  std::deque<std::tuple<mlir::Operation *, std::string, mlir::Operation *, std::string>> modulesToProcess;

  // lmao this is such wack code. can probably use a union or sth
  llvm::DenseMap<NodeRef, std::vector<NodeRef>> incomingEdges;
  // incomingFromIOEdges[Node] = vector[ (name of IO port, namespace diff between node and IO port) ]
  llvm::DenseMap<NodeRef, std::set<std::pair<std::string, std::string>>> incomingFromIOEdges;
  // IOIncomingEdges[ IO port ID ] = vector[ (Node, namespace) ]
  llvm::StringMap<std::vector<std::pair<NodeRef, std::string>>> IOIncomingEdges;
  // Currently used for output to input only. IOFromIOEdges[ Input port ID ] = vector[ Output port IDs ]
  llvm::StringMap<std::vector<std::string>> IOFromIOEdges;
  HWModuleOpJSONGraphTraits jsonGraphTraits;

  std::string getUniqueIOId(const std::string &portName, const std::string &ns) {
    return ns + "_" + portName;
  }

  // This function is necessary as all the IO edge maps store the IO nodes' (namespaced) IDs, not labels.
  void resetIOEdgeMaps() {
    IOFromIOEdges.clear();
    IOIncomingEdges.clear();
    incomingFromIOEdges.clear();
  }

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
            modulesToProcess.push_back({nullptr, module.getName().str(), nullptr, ""});
          })
          .Case<hw::HWModuleGeneratedOp>([&](auto module) {
            if (os)
              *os << "Found HWModuleGeneratedOp: " << module.getName() << "\n";
            modulesToProcess.push_back({nullptr, module.getName().str(), nullptr, ""});
          })
          .Default([&](auto) {
            if (os)
              *os << "Found unknown module: " << op.getName() << "\n";
            modulesToProcess.push_back({nullptr, op.getName().getStringRef().str(), nullptr, ""});
          });
    });

    for (auto const &entry : moduleMap) {
      mlir::Operation *op = moduleMap[entry.getKey()];
      std::string namespaceStr = entry.getKey().str();
      modulesToDrawIO.push({op, namespaceStr, nullptr, ""});
      if (os)
        *os << "Adding top level Module for processing - Name: "
            << entry.getKey() << " Type: " << op->getName() << "\n";
    }
  }

  void drawModuleIO(mlir::Operation *module, const std::string &ns, 
                    mlir::Operation *parentInstance, const std::string &parent_ns) {
    if (!module) {
      return;
    }
    auto moduleOp = mlir::dyn_cast<circt::hw::HWModuleOp>(module);

    *os << "Now drawing IO for module " << moduleOp.getName().str() << " with namespace " << ns << " and parent namespace " << parent_ns << "\n";

    // Process inputs and outputs

    if (parentInstance) {
      drawInputs(moduleOp, ns, parentInstance, parent_ns);
    } else {
      drawInputs(moduleOp, ns, nullptr, "");
    }

    forEachOperation(
        module, [&](mlir::Operation &op) {
          if (auto outputOp = mlir::dyn_cast<OutputOp>(op)) {
            drawOutputs(outputOp, moduleOp, ns, parentInstance, parent_ns);
          }
        }
    );  
    
    modulesToProcess.push_back({moduleOp, ns, parentInstance, parent_ns});
    
    forEachOperation(
      module, [&](mlir::Operation &op) {   
        NodeRef node = &op;
        if (InstanceOp op = mlir::dyn_cast<InstanceOp>(node)) {
          std::string refModuleName =
              op.getReferencedModuleName().str();
          std::string instanceName = op.getInstanceName().str();
          std::string newNs = ns + "/" + refModuleName + "_" + instanceName;
          if (moduleMap.count(refModuleName))
            modulesToDrawIO.push({moduleMap[refModuleName], newNs, node, ns});
          else
            modulesToDrawIO.push({nullptr, newNs, node, ns});
        }
      });
  }

  // Process a module by iterating over its operations and generating JSON
  // nodes.
  void processModule(mlir::Operation *module, const std::string &ns, 
                     mlir::Operation *parentInstance, const std::string &parent_ns) {
    if (!module) {
      llvm::json::Object moduleJson{{"id", getUniqueId(module, ns)},
                                    {"label", "Unknown Module"},
                                    {"namespace", ns}};
      outputJsonObjects.push_back(std::move(moduleJson));
      return;
    }
    auto moduleOp = mlir::dyn_cast<circt::hw::HWModuleOp>(module);

    *os << "Now processing " << moduleOp.getName().str() << " with namespace " << ns << " and parent namespace " << parent_ns << "\n";

    if (parentInstance)
      processInputs(moduleOp, ns, parentInstance, parent_ns);
    else
      processInputs(moduleOp, ns, nullptr, "");
    processOutputs(moduleOp, ns, parentInstance, parent_ns);

    // MAIN LOOP: Process child operations
    forEachOperation(
        module, [&](mlir::Operation &op) {   
          NodeRef node = &op;
          return llvm::TypeSwitch<NodeRef, void>(node)
              .Case<circt::hw::InstanceOp>([&](InstanceOp op) { 
                    // do nothing
                  })
              .Case<circt::hw::OutputOp>([&](OutputOp &op) { 
                    // do nothing
                  })
              .Default([&](NodeRef op) { 
                    llvm::json::Object jsonObj{
                      {"label", jsonGraphTraits.getNodeLabel(node, moduleOp)},
                      {"attrs", jsonGraphTraits.getNodeAttributes(node, moduleOp)},
                      {"id", getUniqueId(node, ns)},
                      {"namespace", ns},
                      {"incomingEdges", getIncomingEdges(node, ns)}};
                    outputJsonObjects.push_back(std::move(jsonObj));
                  });         
        });
        
  }

  // Generate incoming edge JSON for a node.
  llvm::json::Array getIncomingEdges(NodeRef node, const std::string &ns) {
    llvm::json::Array edges;
    for (NodeRef parent : incomingEdges[node]) {
      edges.push_back(
          llvm::json::Object{{"sourceNodeId", getUniqueId(parent, ns)},
                             {"sourceNodeOutputId", "0"},
                             {"targetNodeInputId", "0"}});
    }
    for (std::pair IOp : incomingFromIOEdges[node]) {
      std::string new_ns = ns;
      if (!IOp.second.empty()) new_ns = new_ns + "/" + IOp.second;
      std::string IOPort = getUniqueIOId(IOp.first, new_ns);
      edges.push_back(
          llvm::json::Object{{"sourceNodeId", IOPort},
                             {"sourceNodeOutputId", "0"},
                             {"targetNodeInputId", "0"}}
      );
    }
    return edges;
  }

  llvm::json::Array getIOIncomingEdges(const std::string &portId, const std::string &ns) {
    llvm::json::Array edges;
    for (std::pair parent : IOIncomingEdges[portId]) {
      edges.push_back(
            llvm::json::Object{{"sourceNodeId", getUniqueId(parent.first, parent.second)},
                              {"sourceNodeOutputId", "0"},
                              {"targetNodeInputId", "0"}});
    }
    for (std::string IOPort : IOFromIOEdges[portId]) {
      edges.push_back(
          llvm::json::Object{{"sourceNodeId", IOPort},
                            {"sourceNodeOutputId", "0"},
                            {"targetNodeInputId", "0"}}
      );
    }
    return edges;
  }

  /// Draw input nodes and populate all edges involving them
  void drawInputs(
      circt::hw::HWModuleOp moduleOp,
      const std::string &ns,
      NodeRef parentInstance,
      const std::string &parent_ns) {

    circt::hw::ModulePortInfo ports(moduleOp.getPortList());
    if (ports.sizeInputs() == 0) return;
    auto input_ports = ports.getInputs(); // returns InOut ports as well
    auto module_args = moduleOp.getBodyBlock()->getArguments();
    bool isTopLevel = (parent_ns.empty()) ? true : false;

    for (auto [iport, arg] :
          llvm::zip(input_ports, module_args)) {
      
      // The input node itself
      std::string iportName = iport.getName().str();

      // Generate outgoing edges from the input node.
      // These are always regular nodes from the same module. 
      for (auto *user : arg.getUsers()) {
        if (OutputOp destOutput = mlir::dyn_cast<OutputOp>(user)) {
          // TODO: input that points directly to output
        } else {
          incomingFromIOEdges[user].emplace(std::make_pair(iportName, ""));
        }
      }
    }

    // Generate incoming edges into the input node. These affect IOFromIOEdges and IOIncomingEdges
    if (!isTopLevel) {
      auto parentInstanceOp = mlir::dyn_cast<circt::hw::InstanceOp>(parentInstance);

      for (auto [iport, arg, oper] :
        llvm::zip(input_ports, module_args, parentInstanceOp.getOperands())) {

        std::string iportName = iport.getName().str();
        std::string iportId = getUniqueIOId(iportName, ns);
        *os << "The input operand " << oper << " for port " << iportId;

        if (NodeRef operSource = oper.getDefiningOp()) {

          // The edge could come from the hw.output of another instance...
          if (InstanceOp sourceInstance = mlir::dyn_cast<InstanceOp>(operSource)) {
            *os << " comes from instance op\n";

            circt::hw::ModulePortInfo ports(sourceInstance.getPortList());
            llvm::SmallVector<mlir::Value> values;
            sourceInstance.getValues(values, ports);
            
            // sadly there is no better way to check exactly which output port it comes from,
            // than to iterate over all the output values and identify the one with the same location
            for (auto [port, val] : llvm::zip(sourceInstance.getPortList(), values)) {
              if (port.dir == PortInfo::Direction::Output) {
                if (oper.getLoc() == val.getLoc()) {
                  std::string refModuleName = sourceInstance.getReferencedModuleName().str();
                  std::string instanceName = sourceInstance.getInstanceName().str();
                  std::string newNs = parent_ns + "/" + refModuleName + "_" + instanceName;
                  IOFromIOEdges[iportId].push_back(getUniqueIOId(port.getName().str(), newNs));
                  break;
                }
              } else { continue; }
            }

          // ...or a plane jane Operation from the parent module...
          } else {
            IOIncomingEdges[iportId].push_back(std::make_pair(operSource, parent_ns));
            *os << " comes from a plain jane operation " << operSource->getName() << "\n";
          }
        // ...or a BlockArgument from the parent module
        } else {
          auto arg = dyn_cast<mlir::BlockArgument>(oper);
          auto parentModule = parentInstance->getParentOp();
          if (auto parentModuleOp = mlir::dyn_cast<circt::hw::HWModuleOp>(parentModule)) {
            std::string sourceIport = parentModuleOp.getInputName(arg.getArgNumber()).str();
            IOFromIOEdges[iportId].push_back(getUniqueIOId(sourceIport, parent_ns));
            *os << " comes from a block argument\n";
          } else {
            *os << parentModule->getName() << " is not a HWModuleOp!\n";
          }
        }
      }
    }
  }

  /// Generate output node JSONs
  void processInputs(
      circt::hw::HWModuleOp moduleOp,
      const std::string &ns,
      NodeRef parentInstance,
      const std::string &parent_ns) {

    circt::hw::ModulePortInfo ports(moduleOp.getPortList());
    if (ports.sizeInputs() == 0) return;
    auto input_ports = ports.getInputs(); // returns InOut ports as well
    bool isTopLevel = (parent_ns.empty()) ? true : false;

    // Top-level input nodes don't have any incoming edges.
    if (isTopLevel) {
      for (auto iport : input_ports) {
        std::string iportName = iport.getName().str();
        std::string iportId = getUniqueIOId(iportName, ns);
        llvm::json::Object jsonObj{
          {"label", iportName},
          {"attrs", jsonGraphTraits.getInputNodeAttributes()},
          {"id", iportId},
          {"namespace", getIONamespace(iport, ns)}};
        outputJsonObjects.push_back(std::move(jsonObj));
      }
      
    // Non-top-level input nodes have incoming edges.
    } else {
      for (auto iport : input_ports) {
        std::string iportName = iport.getName().str();
        std::string iportId = getUniqueIOId(iportName, ns);
        llvm::json::Object jsonObj{
          {"label", iportName},
          {"attrs", jsonGraphTraits.getInputNodeAttributes()},
          {"id", iportId},
          {"namespace", getIONamespace(iport, ns)},
          {"incomingEdges", getIOIncomingEdges(iportId, ns)}};
        outputJsonObjects.push_back(std::move(jsonObj));
      }
    }
  }

  /// Draw output nodes and populate all edges involving them
  void drawOutputs(
    circt::hw::OutputOp output,
    circt::hw::HWModuleOp moduleOp,
    const std::string &ns,
    NodeRef parentInstance,
    const std::string &parent_ns) {

      circt::hw::ModulePortInfo ports(moduleOp.getPortList());
      if (ports.sizeOutputs() == 0) return;
      auto output_ports = ports.getOutputs();
      bool isTopLevel = (parent_ns.empty()) ? true : false;

      // Generate output nodes
      for (auto [outputOper, oport] : 
          llvm::zip(output.getOperands(), output_ports)) {
        std::string oportName = oport.getName().str();
        std::string oportId = getUniqueIOId(oportName, ns);

        if (NodeRef sourceOp = outputOper.getDefiningOp()) {
          // Generate edges from generic nodes into our output node
          IOIncomingEdges[oportId].push_back(std::make_pair(sourceOp, ns));
          
        // TODO: special case where an input node leads directly into our output node
        } else {
          *os << "InOut port was detected as output port " << oportName << "\n";
        }
      }

      // Generate edges from output nodes to other nodes. These affect incomingFromIOEdges and IOFromIOEdges
      if (!isTopLevel) {
        InstanceOp parentInstanceOp = mlir::dyn_cast<InstanceOp>(parentInstance);
        std::string refModuleName = parentInstanceOp.getReferencedModuleName().str();
        std::string instanceName = parentInstanceOp.getInstanceName().str();

        for (auto [outputOper, oport, result] : 
          llvm::zip(output.getOperands(), output_ports, parentInstance->getResults())) {
            std::string oportName = oport.getName().str();
            std::string oportId = getUniqueIOId(oportName, ns);

            *os << "Output operand " << oportName << " users in namespace " << parent_ns << ": ";
            for (NodeRef user : result.getUsers()) {
              *os << user->getName() << " ";

              // Case 1: output points to another instance's input. Handled by drawInputs()
              if (auto destInstance = mlir::dyn_cast<InstanceOp>(user)) {
                *os << "(instance), ";

              // Case 2: output points to parent module's output.
              } else if (auto destOutput = mlir::dyn_cast<circt::hw::OutputOp>(user)) {
                circt::hw::HWModuleOp parentModule = destOutput.getParentOp();
                circt::hw::ModulePortInfo parentPorts(parentModule.getPortList());
                auto parentOutputPorts = parentPorts.getOutputs();
                
                // once again there is no better way to identify the correct output port of the parent module,
                // than to iterate over all the output values and identify the one with the same location
                *os << result.getLoc() << " ";
                for (auto [destOper, destOport] : 
                  llvm::zip(destOutput.getOperands(), parentOutputPorts)) {
                  *os << destOper.getLoc() << " ";
                  if (result.getLoc() == destOper.getLoc()) {
                    std::string destPortId = getUniqueIOId(destOport.getName().str(), parent_ns);
                    IOFromIOEdges[destPortId].push_back(oportId);
                    *os << "FOUND (" << destPortId << ", " << oportId << ") ";
                    break;
                  }
                }
                *os << "(parent output), ";

              // Case 3: output points to generic node in the parent instance.
              } else {
                std::string nsDiff = refModuleName + "_" + instanceName;
                incomingFromIOEdges[user].emplace(std::make_pair(oportName, nsDiff));
                *os << "(node), ";
              }
            }
            *os << "\n";

            // edges from output nodes to other input nodes
          }
      }
  }

  /// Generate output node JSONs
  void processOutputs(
    circt::hw::HWModuleOp moduleOp,
    const std::string &ns,
    NodeRef parentInstance,
    const std::string &parent_ns) {

      circt::hw::ModulePortInfo ports(moduleOp.getPortList());
      if (ports.sizeOutputs() == 0) return;
      auto output_ports = ports.getOutputs();

      // Generate output nodes
      for (auto oport : output_ports) {
        std::string oportName = oport.getName().str();
        std::string oportId = getUniqueIOId(oportName, ns);
        llvm::json::Object jsonObj{
          {"label", oportName},
          {"attrs", jsonGraphTraits.getInputNodeAttributes()},
          {"id", oportId},
          {"namespace", getIONamespace(oport, ns)},
          {"incomingEdges", getIOIncomingEdges(oportId, ns)}};
        outputJsonObjects.push_back(std::move(jsonObj));
      }
    }

  // Populate incoming edge relationships for *non-IO operation nodes* using GraphTraits.
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
        if (auto c = mlir::dyn_cast<InstanceOp>(child)) {
          // pass
        } else {
          edgesMap[child].push_back(current);
          nodesToVisit.push(child);
        }
      }
    }
  }

  std::string getIONamespace(circt::hw::PortInfo port, const std::string &ns) {
    if (port.dir == ModulePort::Direction::Input)
      return ns + "/Inputs";
    else if (port.dir == ModulePort::Direction::Output)
      return ns + "/Outputs";
    else return ns + "/IONamespaceError";
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
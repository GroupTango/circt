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

llvm::cl::opt<std::string> outFile("outfile",
                                   llvm::cl::desc("Specify output file"),
                                   llvm::cl::value_desc("filename"),
                                   llvm::cl::init("out.json"));

namespace {

// Shallow iteration over all operations in the top-level module.
template <typename Fn>
void forEachOperation(NodeRef op, Fn f) {
  for (mlir::Region &region : op->getRegions())
    for (mlir::Block &block : region.getBlocks())
      for (mlir::Operation &childOp : block.getOperations())
        f(childOp);
}

class GraphGenerator {
public:
  GraphGenerator(llvm::raw_ostream *os) : os(os), nextNodeId(0) {}

  virtual ~GraphGenerator() = default;

  // Main entry point: initialize, process modules, and wrap the output.
  virtual std::string generateGraphJson() = 0;

protected:
  llvm::raw_ostream *os;
  llvm::StringMap<NodeRef> moduleMap;
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
    llvm::json::OStream jso(jsonStream, /*IndentSize=*/2);
    jso.value(llvm::json::Value(std::move(fileArrayWrapper)));
    return jsonStream.str();
  }

  // Generate a unique ID for a node using its existing attribute if present.
  std::string getUniqueId(NodeRef node, const std::string &ns) {
    if (ns.empty())
      return NULL;

    if (!node)
      return ns + "_" + std::to_string(nextNodeId++);

    return ns + "_" +
           std::to_string(
               mlir::cast<IntegerAttr>(node->getAttr("hw.unique_id")).getInt());
  }

  bool isCombOp(NodeRef op) {
    return op->getName().getDialectNamespace() == "comb";
  }
};

// Graph generator for Instance Graphs.
class InstanceGraphGenerator : public GraphGenerator {
public:
  InstanceGraphGenerator(NodeRef baseOperation, llvm::raw_ostream *os)
      : GraphGenerator(os), baseOperation(baseOperation) {}

  std::string generateGraphJson() override {
    // Discover all modules to graph.
    forEachOperation(baseOperation, [&](mlir::Operation &op) {
      llvm::TypeSwitch<NodeRef>(&op)
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
    std::stack<std::tuple<NodeRef, std::string, int64_t>> treesToProcess;

    for (auto const &entry : moduleMap) {
      NodeRef baseModule = moduleMap[entry.getKey()];
      treesToProcess.push({baseModule, entry.getKey().str(), -1});
    }

    while (treesToProcess.size() > 0) {
      NodeRef module;
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
        llvm::TypeSwitch<NodeRef>(module)
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

    return wrapJson(outputJsonObjects);
  }

protected:
  NodeRef baseOperation;

  void generateInstanceNode(llvm::StringRef label, int64_t nextNodeId,
                            std::string &newNamespace, int64_t parentId) {
    llvm::json::Object instanceJson{{"id", std::to_string(nextNodeId)},
                                    {"namespace", newNamespace},
                                    {"label", label}};
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

// Graph generator for Operation Graphs.
class OperationGraphGenerator : public GraphGenerator {
public:
  OperationGraphGenerator(NodeRef baseModule, llvm::raw_ostream *os)
      : GraphGenerator(os), baseModule(baseModule) {}

  std::string generateGraphJson() override {
    initializeModules();

    while (!modulesToDrawIO.empty()) {
      auto [module, ns, parent_instance, parent_ns] =
          modulesToDrawIO.top(); // requires C++17
      if (parent_ns.empty()) {
        processGraphJson();
        resetIOEdgeMaps();
      }
      modulesToDrawIO.pop();
      drawModuleIO(module, ns, parent_instance, parent_ns);
    }
    processGraphJson();
    processCombGroups();
    updateIncomingEdges();
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
  NodeRef baseModule;
  std::stack<std::tuple<NodeRef, std::string, NodeRef, std::string>>
      modulesToDrawIO;
  std::deque<std::tuple<NodeRef, std::string, NodeRef, std::string>>
      modulesToProcess;

  // lmao this is such wack code. can probably use a union or sth
  llvm::DenseMap<NodeRef, std::vector<NodeRef>> incomingEdges;
  // incomingFromIOEdges[Node] = vector[ (name of IO port, namespace diff
  // between node and IO port) ]
  llvm::DenseMap<NodeRef, std::set<std::pair<std::string, std::string>>>
      incomingFromIOEdges;
  // ioIncomingEdges[ IO port ID ] = vector[ (Node, namespace) ]
  llvm::StringMap<std::vector<std::pair<NodeRef, std::string>>> ioIncomingEdges;
  // Currently used for output to input only. iOFromIOEdges[ Input port ID ] =
  // vector[ Output port IDs ]
  llvm::StringMap<std::vector<std::string>> iOFromIOEdges;
  HWModuleOpJSONGraphTraits jsonGraphTraits;
  llvm::DenseMap<NodeRef, std::vector<std::string>> incomingInputEdges;
  llvm::StringMap<NodeRef> idToNodeMap;
  llvm::StringMap<std::vector<NodeRef>> outputIncomingEdges;

  std::vector<std::pair<NodeRef, std::string>> combOps;
  std::map<std::string, std::string> combOpIdMap;
  std::set<std::pair<NodeRef, std::string>> globalVisited;

  /*
   * Collects a group of comb operations starting from the given operation.
   */
  std::vector<NodeRef> collectCombGroup(NodeRef op, const std::string &ns) {
    std::stack<NodeRef> stack;
    std::vector<NodeRef> group;
    stack.push(op);

    while (!stack.empty()) {
      NodeRef currentOp = stack.top();
      stack.pop();

      if (this->globalVisited.count(
              std::pair<NodeRef, std::string>(currentOp, ns)))
        continue;
      globalVisited.insert(std::pair<NodeRef, std::string>(currentOp, ns));

      if (isCombOp(currentOp))
        group.push_back(currentOp);

      for (auto it = HWModuleOpGraphTraits::child_begin(currentOp),
                end = HWModuleOpGraphTraits::child_end(currentOp);
           it != end; ++it) {
        NodeRef succ = *it;
        // Only follow successors if they are comb ops.
        if (isCombOp(succ))
          stack.push(succ);
      }
    }

    return group;
  }

  void processCombGroups() {
    for (auto &pair : combOps) {
      NodeRef op = pair.first;
      std::string ns = pair.second;

      if (globalVisited.count(std::pair<NodeRef, std::string>(op, ns)))
        continue;

      if (incomingEdges[op].size() == 0)
        continue;

      std::vector<NodeRef> group = collectCombGroup(incomingEdges[op][0], ns);
      if (group.size() == 0)
        continue;

      // Generate a comb group node.
      std::string groupNs = ns + "/" + getUniqueId(op, "CombGroup");

      for (NodeRef combOp : group) {
        std::string originalId = getUniqueId(combOp, ns);
        std::string combId = getUniqueId(combOp, groupNs);
        combOpIdMap[originalId] = combId;

        *os << "Mapping comb op: originalId=" << originalId
            << ", combId=" << combId << "\n";

        llvm::json::Object jsonObj{
            {"label", jsonGraphTraits.getNodeLabel(combOp, nullptr)},
            {"attrs", jsonGraphTraits.getNodeAttributes(combOp, nullptr)},
            {"id", combId},
            {"namespace", groupNs},
        };
        outputJsonObjects.push_back(std::move(jsonObj));
        idToNodeMap[combId] = combOp;
      }
    }
  }

  std::string getUniqueIOId(const std::string &portName,
                            const std::string &ns) {
    return ns + "_" + portName;
  }

  // This function is necessary as all the IO edge maps store the IO nodes'
  // (namespaced) IDs, not labels.
  void resetIOEdgeMaps() {
    iOFromIOEdges.clear();
    ioIncomingEdges.clear();
    incomingFromIOEdges.clear();
  }

  /// Discover all top-level modules, assign unique IDs to all operations via
  /// @c hw.unique_id , populate intra-module dependency edges, and process
  /// input nodes and dependency edges.
  void initializeModules() {
    int64_t counter = 0;
    baseModule->walk([&](NodeRef op) {
      auto id = mlir::IntegerAttr::get(
          mlir::IntegerType::get(op->getContext(), 64), counter++);
      op->setAttr("hw.unique_id", id);
    });

    forEachOperation(baseModule, [&](mlir::Operation &op) {
      llvm::TypeSwitch<NodeRef>(&op)
          .Case<hw::HWModuleOp>([&](auto module) {
            if (os)
              *os << "Found HWModuleOp: " << module.getName() << "\n";
            moduleMap[module.getName()] = &op;
            populateIncomingEdges(module, incomingEdges);
          })
          .Case<hw::HWModuleExternOp>([&](auto module) {
            if (os)
              *os << "Found HWModuleExternOp: " << module.getName() << "\n";
            modulesToProcess.push_back(
                {nullptr, module.getName().str(), nullptr, ""});
          })
          .Case<hw::HWModuleGeneratedOp>([&](auto module) {
            if (os)
              *os << "Found HWModuleGeneratedOp: " << module.getName() << "\n";
            modulesToProcess.push_back(
                {nullptr, module.getName().str(), nullptr, ""});
          })
          .Default([&](auto) {
            if (os)
              *os << "Found unknown module: " << op.getName() << "\n";
            modulesToProcess.push_back(
                {nullptr, op.getName().getStringRef().str(), nullptr, ""});
          });
    });

    for (auto const &entry : moduleMap) {
      NodeRef op = moduleMap[entry.getKey()];
      std::string namespaceStr = entry.getKey().str();
      modulesToDrawIO.push({op, namespaceStr, nullptr, ""});
      if (os)
        *os << "Adding top level Module for "
               "processing - Name: "
            << entry.getKey() << " Type: " << op->getName() << "\n";
    }
  }

  void drawModuleIO(NodeRef module, const std::string &ns,
                    NodeRef parentInstance, const std::string &parentNs) {
    if (!module) {
      return;
    }
    auto moduleOp = mlir::dyn_cast<circt::hw::HWModuleOp>(module);

    *os << "Now drawing IO for module " << moduleOp.getName().str()
        << " with namespace " << ns << " and parent namespace " << parentNs
        << "\n";

    // Process inputs and outputs

    if (parentInstance) {
      drawInputs(moduleOp, ns, parentInstance, parentNs);
    } else {
      drawInputs(moduleOp, ns, nullptr, "");
    }

    forEachOperation(module, [&](mlir::Operation &op) {
      if (auto outputOp = mlir::dyn_cast<OutputOp>(op)) {
        drawOutputs(outputOp, moduleOp, ns, parentInstance, parentNs);
      }
    });

    modulesToProcess.push_back({moduleOp, ns, parentInstance, parentNs});

    forEachOperation(module, [&](mlir::Operation &op) {
      NodeRef node = &op;
      if (InstanceOp op = mlir::dyn_cast<InstanceOp>(node)) {
        std::string refModuleName = op.getReferencedModuleName().str();
        std::string instanceName = op.getInstanceName().str();
        std::string newNs = ns + "/" + refModuleName + "_" + instanceName;
        if (moduleMap.count(refModuleName))
          modulesToDrawIO.push({moduleMap[refModuleName], newNs, node, ns});
        else
          modulesToDrawIO.push({nullptr, newNs, node, ns});
      }
    });
  }

  // Process a module by iterating over its operations
  // and generating JSON nodes.
  void processModule(NodeRef module, const std::string &ns,
                     NodeRef parentInstance, const std::string &parentNs) {
    if (!module) {
      llvm::json::Object moduleJson{{"id", getUniqueId(module, ns)},
                                    {"label", "Unknown Module"},
                                    {"namespace", ns}};
      outputJsonObjects.push_back(std::move(moduleJson));
      idToNodeMap[getUniqueId(module, ns)] = module;
      return;
    }
    auto moduleOp = mlir::dyn_cast<circt::hw::HWModuleOp>(module);

    *os << "Now processing " << moduleOp.getName().str() << " with namespace "
        << ns << " and parent namespace " << parentNs << "\n";

    if (parentInstance)
      processInputs(moduleOp, ns, parentInstance, parentNs);
    else
      processInputs(moduleOp, ns, nullptr, "");
    processOutputs(moduleOp, ns, parentInstance, parentNs);

    // MAIN LOOP: Process child operations
    forEachOperation(module, [&](mlir::Operation &op) {
      NodeRef node = &op;
      // Skip over hw.instance and hw.output: these are already handled by
      // the recursive module functionality, and the output node drawer.
      if (auto inst = mlir::dyn_cast<InstanceOp>(node)) {

      } else if (auto output = mlir::dyn_cast<OutputOp>(node)) {

      }
      // We collect comb ops to be grouped later.
      else if (node->getDialect() && node->getDialect()->getNamespace() == "comb") {
        combOps.push_back(std::pair<NodeRef, std::string>(node, ns));
      } else {
        llvm::json::Object jsonObj{
            {"label", jsonGraphTraits.getNodeLabel(node, moduleOp)},
            {"attrs", jsonGraphTraits.getNodeAttributes(node, moduleOp)},
            {"id", getUniqueId(node, ns)},
            {"namespace", ns}};
        outputJsonObjects.push_back(std::move(jsonObj));
        idToNodeMap[getUniqueId(node, ns)] = node;
      }
    });
  }

  // Generate incoming edge JSON for a node.
  llvm::json::Array getIncomingEdges(NodeRef node, const std::string &ns) {
    llvm::json::Array edges;
    std::string ns1 = ns;
    if (isCombOp(node)) {
      // The node's namespace contains the comb group, we want the original
      // namespace.
      size_t lastSlashPos = ns.find_last_of('/');
      if (lastSlashPos != std::string::npos)
        ns1 = ns.substr(0, lastSlashPos);
    }
    for (NodeRef parent : incomingEdges[node]) {
      std::string uniqueId = getUniqueId(parent, ns1);

      if (isCombOp(parent))
        *os << "Looking up parent: originalId=" << uniqueId << ", found combId="
            << (combOpIdMap.count(uniqueId) ? combOpIdMap[uniqueId]
                                            : "NOT FOUND")
            << "\n";

      edges.push_back(llvm::json::Object{
          // If the parent is a comb op, we have to translate the original ID to
          // the ID within the comb group namespace.
          {"sourceNodeId", isCombOp(parent) ? combOpIdMap[uniqueId] : uniqueId},
          {"sourceNodeOutputId", "0"},
          {"targetNodeInputId", "0"}});
    }
    for (const std::pair<std::string, std::string> &iop :
         incomingFromIOEdges[node]) {
      std::string newNs = ns1;
      if (!iop.second.empty())
        newNs = newNs + "/" + iop.second;
      std::string ioPort = getUniqueIOId(iop.first, newNs);
      edges.push_back(llvm::json::Object{{"sourceNodeId", ioPort},
                                         {"sourceNodeOutputId", "0"},
                                         {"targetNodeInputId", "1"}});
    }
    return edges;
  }

  llvm::json::Array getioIncomingEdges(const std::string &portId,
                                       const std::string &ns) {
    llvm::json::Array edges;
    for (const std::pair<NodeRef, std::string> &parent :
         ioIncomingEdges[portId]) {
      std::string uniqueId = getUniqueId(parent.first, parent.second);
      edges.push_back(llvm::json::Object{
          {"sourceNodeId", isCombOp(parent.first) ? combOpIdMap[uniqueId] : uniqueId},
          {"sourceNodeOutputId", "1"},
          {"targetNodeInputId", "0"}});
    }
    for (std::string ioPort : iOFromIOEdges[portId]) {
      edges.push_back(llvm::json::Object{{"sourceNodeId", ioPort},
                                         {"sourceNodeOutputId", "1"},
                                         {"targetNodeInputId", "1"}});
    }
    return edges;
  }

  llvm::json::Array getOutputIncomingEdges(mlir::Value &outputOper,
                                           std::string &outputId,
                                           const std::string &ns) {
    llvm::json::Array edges;
    if (NodeRef operSource = outputOper.getDefiningOp()) {
      outputIncomingEdges[outputId].push_back(operSource);
      if (os)
        *os << "Operation " << operSource->getName().getStringRef().str()
            << " used output port " << outputId << "\n";

      for (NodeRef parent : outputIncomingEdges[outputId]) {
        edges.push_back(
            llvm::json::Object{{"sourceNodeId", getUniqueId(parent, ns)},
                               {"sourceNodeOutputId", "0"},
                               {"targetNodeInputId", "0"}});
      }
      outputIncomingEdges.erase(outputId);
    }
    return edges;
  }

  void updateIncomingEdges() {
    for (llvm::json::Value &nodeVal : outputJsonObjects) {
      if (auto *obj = nodeVal.getAsObject()) {
        // Process only nodes with a "namespace" field.
        if (obj->find("namespace") == obj->end())
          continue;

        std::string id = obj->getString("id")->str();
        std::string ns = obj->getString("namespace")->str();

        if (idToNodeMap.count(id))
          (*obj)["incomingEdges"] = getIncomingEdges(idToNodeMap[id], ns);
      }
    }
  }

  /// Draw input nodes and populate all edges involving them
  void drawInputs(circt::hw::HWModuleOp moduleOp, const std::string &ns,
                  NodeRef parentInstance, const std::string &parentNs) {

    circt::hw::ModulePortInfo ports(moduleOp.getPortList());
    if (ports.sizeInputs() == 0)
      return;
    auto inputPorts = ports.getInputs(); // returns InOut ports as
                                         // well
    auto moduleArgs = moduleOp.getBodyBlock()->getArguments();
    bool isTopLevel = parentNs.empty();

    for (auto [iport, arg] : llvm::zip(inputPorts, moduleArgs)) {

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

    // Generate incoming edges into the input node.
    // These affect iOFromIOEdges and ioIncomingEdges
    if (!isTopLevel) {
      auto parentInstanceOp =
          mlir::dyn_cast<circt::hw::InstanceOp>(parentInstance);

      for (auto [iport, arg, oper] :
           llvm::zip(inputPorts, moduleArgs, parentInstanceOp.getOperands())) {

        std::string iportName = iport.getName().str();
        std::string iportId = getUniqueIOId(iportName, ns);
        *os << "The input operand " << oper << " for port " << iportId;

        if (NodeRef operSource = oper.getDefiningOp()) {

          // The edge could come from the hw.output of
          // another instance...
          if (InstanceOp sourceInstance =
                  mlir::dyn_cast<InstanceOp>(operSource)) {
            *os << " comes from instance op\n";

            circt::hw::ModulePortInfo ports(sourceInstance.getPortList());
            llvm::SmallVector<mlir::Value> values;
            sourceInstance.getValues(values, ports);

            // sadly there is no better way to check
            // exactly which output port it comes
            // from, than to iterate over all the
            // output values and identify the one with
            // the same location
            for (auto [port, val] :
                 llvm::zip(sourceInstance.getPortList(), values)) {
              if (port.dir == PortInfo::Direction::Output) {
                if (oper.getLoc() == val.getLoc()) {
                  std::string refModuleName =
                      sourceInstance.getReferencedModuleName().str();
                  std::string instanceName =
                      sourceInstance.getInstanceName().str();
                  std::string newNs =
                      parentNs + "/" + refModuleName + "_" + instanceName;
                  iOFromIOEdges[iportId].push_back(
                      getUniqueIOId(port.getName().str(), newNs));
                  break;
                }
              } else {
                continue;
              }
            }

            // ...or a plane jane Operation from the
            // parent module...
          } else {
            ioIncomingEdges[iportId].push_back(
                std::make_pair(operSource, parentNs));
            *os << " comes from a plain jane "
                   "operation "
                << operSource->getName() << "\n";
          }
          // ...or a BlockArgument from the parent
          // module
        } else {
          auto arg = dyn_cast<mlir::BlockArgument>(oper);
          auto *parentModule = parentInstance->getParentOp();
          if (auto parentModuleOp =
                  mlir::dyn_cast<circt::hw::HWModuleOp>(parentModule)) {
            std::string sourceIport =
                parentModuleOp.getInputName(arg.getArgNumber()).str();
            iOFromIOEdges[iportId].push_back(
                getUniqueIOId(sourceIport, parentNs));
            *os << " comes from a block argument\n";
          } else {
            *os << parentModule->getName() << " is not a HWModuleOp!\n";
          }
        }
      }
    }
  }

  /// Generate output node JSONs
  void processInputs(circt::hw::HWModuleOp moduleOp, const std::string &ns,
                     NodeRef parentInstance, const std::string &parentNs) {

    circt::hw::ModulePortInfo ports(moduleOp.getPortList());
    if (ports.sizeInputs() == 0)
      return;
    auto inputPorts = ports.getInputs(); // returns InOut ports as
                                         // well
    bool isTopLevel = parentNs.empty();

    // Top-level input nodes don't have any incoming
    // edges.
    if (isTopLevel) {
      for (auto iport : inputPorts) {
        std::string iportName = iport.getName().str();
        std::string iportId = getUniqueIOId(iportName, ns);
        llvm::json::Object jsonObj{
            {"label", iportName},
            {"attrs", jsonGraphTraits.getInputNodeAttributes()},
            {"id", iportId},
            {"namespace", getIONamespace(iport, ns)}};
        outputJsonObjects.push_back(std::move(jsonObj));
      }

      // Non-top-level input nodes have incoming
      // edges.
    } else {
      for (auto iport : inputPorts) {
        std::string iportName = iport.getName().str();
        std::string iportId = getUniqueIOId(iportName, ns);
        llvm::json::Object jsonObj{
            {"label", iportName},
            {"attrs", jsonGraphTraits.getInputNodeAttributes()},
            {"id", iportId},
            {"namespace", getIONamespace(iport, ns)},
            {"incomingEdges", getioIncomingEdges(iportId, ns)}};
        outputJsonObjects.push_back(std::move(jsonObj));
      }
    }
  }

  /// Draw output nodes and populate all edges
  /// involving them
  void drawOutputs(circt::hw::OutputOp output, circt::hw::HWModuleOp moduleOp,
                   const std::string &ns, NodeRef parentInstance,
                   const std::string &parentNs) {

    circt::hw::ModulePortInfo ports(moduleOp.getPortList());
    if (ports.sizeOutputs() == 0)
      return;
    auto outputPorts = ports.getOutputs();
    bool isTopLevel = parentNs.empty();

    // Generate output nodes
    for (auto [outputOper, oport] :
         llvm::zip(output.getOperands(), outputPorts)) {
      std::string oportName = oport.getName().str();
      std::string oportId = getUniqueIOId(oportName, ns);

      if (NodeRef sourceOp = outputOper.getDefiningOp()) {
        // Generate edges from generic nodes into our
        // output node
        ioIncomingEdges[oportId].push_back(std::make_pair(sourceOp, ns));

        // TODO: special case where an input node
        // leads directly into our output node
      } else {
        *os << "InOut port was detected as output "
               "port "
            << oportName << "\n";
      }
    }

    // Generate edges from output nodes to other
    // nodes. These affect incomingFromIOEdges and
    // iOFromIOEdges
    if (!isTopLevel) {
      InstanceOp parentInstanceOp = mlir::dyn_cast<InstanceOp>(parentInstance);
      std::string refModuleName =
          parentInstanceOp.getReferencedModuleName().str();
      std::string instanceName = parentInstanceOp.getInstanceName().str();

      for (auto [outputOper, oport, result] :
           llvm::zip(output.getOperands(), outputPorts,
                     parentInstance->getResults())) {
        std::string oportName = oport.getName().str();
        std::string oportId = getUniqueIOId(oportName, ns);

        *os << "Output operand " << oportName << " users in namespace "
            << parentNs << ": ";
        for (NodeRef user : result.getUsers()) {
          *os << user->getName() << " ";

          // Case 1: output points to another
          // instance's input. Handled by drawInputs()
          if (auto destInstance = mlir::dyn_cast<InstanceOp>(user)) {
            *os << "(instance), ";

            // Case 2: output points to parent
            // module's output.
          } else if (auto destOutput =
                         mlir::dyn_cast<circt::hw::OutputOp>(user)) {
            circt::hw::HWModuleOp parentModule = destOutput.getParentOp();
            circt::hw::ModulePortInfo parentPorts(parentModule.getPortList());
            auto parentOutputPorts = parentPorts.getOutputs();

            // once again there is no better way to
            // identify the correct output port of the
            // parent module, than to iterate over all
            // the output values and identify the one
            // with the same location
            *os << result.getLoc() << " ";
            for (auto [destOper, destOport] :
                 llvm::zip(destOutput.getOperands(), parentOutputPorts)) {
              *os << destOper.getLoc() << " ";
              if (result.getLoc() == destOper.getLoc()) {
                std::string destPortId =
                    getUniqueIOId(destOport.getName().str(), parentNs);
                iOFromIOEdges[destPortId].push_back(oportId);
                *os << "FOUND (" << destPortId << ", " << oportId << ") ";
                break;
              }
            }
            *os << "(parent output), ";

            // Case 3: output points to generic node
            // in the parent instance.
          } else {
            std::string nsDiff = refModuleName + "_" + instanceName;
            incomingFromIOEdges[user].emplace(
                std::make_pair(oportName, nsDiff));
            *os << "(node), ";
          }
        }
        *os << "\n";

        // edges from output nodes to other input
        // nodes
      }
    }
    // TODO: figure out a compromise for generating
    // edges from parent module ops to child input
    // nodes
  }

  /// Generate output node JSONs
  void processOutputs(circt::hw::HWModuleOp moduleOp, const std::string &ns,
                      NodeRef parentInstance, const std::string &parentNs) {

    circt::hw::ModulePortInfo ports(moduleOp.getPortList());
    if (ports.sizeOutputs() == 0)
      return;
    auto outputPorts = ports.getOutputs();

    // Generate output nodes
    for (auto oport : outputPorts) {
      std::string oportName = oport.getName().str();
      std::string oportId = getUniqueIOId(oportName, ns);
      llvm::json::Object jsonObj{
          {"label", oportName},
          {"attrs", jsonGraphTraits.getInputNodeAttributes()},
          {"id", oportId},
          {"namespace", getIONamespace(oport, ns)},
          {"incomingEdges", getioIncomingEdges(oportId, ns)}};
      outputJsonObjects.push_back(std::move(jsonObj));
    }
  }

  // Populate incoming edge relationships for *non-IO
  // operation nodes* using GraphTraits.
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
    if (port.dir == ModulePort::Direction::Output)
      return ns + "/Outputs";
    return ns + "/IONamespaceError";
  }
};

// Graph generator for Diff Graphs.
class InstanceDiffGraphGenerator : GraphGenerator {
public:
  InstanceDiffGraphGenerator(NodeRef baseOriginalModule, NodeRef baseNewModule,
                             llvm::raw_ostream *os)
      : GraphGenerator(os), baseOriginalModule(baseOriginalModule),
        baseNewModule(baseNewModule) {}

  std::string generateGraphJson() override {
    /*// Discover all modules to graph.
    forEachOperation(baseOperation, [&](mlir::Operation &op) {
      llvm::TypeSwitch<NodeRef>(&op)
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
              *os << "Found HWModuleGeneratedOp: " << module.getName() <<
    "\n"; moduleMap[module.getName()] = &op;
          })
          .Default([&](auto) {
            if (os)
              *os << "Found unknown module type: " << op.getName() << "\n";
            moduleMap[op.getName().getStringRef()] = &op;
          });
    });

    // Process modules.
    std::stack<std::tuple<NodeRef, std::string, int64_t>>
    treesToProcess;

    for (auto const &entry : moduleMap) {
      NodeRefbaseModule = moduleMap[entry.getKey()];
      treesToProcess.push({baseModule, entry.getKey().str(), -1});

      //if (os)
      //  *os << "Queuing top level Module - Name: "
      //      << entry.getKey() << " Type: " << baseModule->getName() << "\n";
    }

    while (treesToProcess.size() > 0)
    {
      NodeRefmodule;
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
    currentNamespace, nextNodeId}); nextNodeId++; hasInstances = true; } else
    if (auto choiceInstance = dyn_cast<InstanceChoiceOp>(&op)) {
    mlir::ArrayAttr moduleNames = choiceInstance.getModuleNamesAttr();
    std::string newNamespace = currentNamespace + "/CHOICE (" +
    std::to_string(instanceChoiceId) + ")"; for (auto attr : moduleNames) {
            llvm::StringRef instanceName =
    cast<FlatSymbolRefAttr>(attr).getAttr().getValue();
            generateInstanceNode(instanceName, nextNodeId, newNamespace,
    parentId); if (moduleMap.count(instanceName))
              treesToProcess.push({moduleMap[instanceName], newNamespace,
    nextNodeId}); nextNodeId++;
          }
          instanceChoiceId++;
          hasInstances = true;
        }
      });

      // If this is a top level and independant module, we will display
    appropriate node. if (!hasInstances && parentId == -1)
      {
        llvm::json::Object moduleJson{{"id", std::to_string(nextNodeId)},
                                      {"namespace", currentNamespace},
                                      {"attrs", llvm::json::Array {
                                        llvm::json::Object{
                                          {"key", "type"},
                                          {"value", "Top Level"}}}}};
        llvm::TypeSwitch<NodeRef>(module)
            .Case<hw::HWModuleOp>(
                [&](auto mod) { moduleJson["label"] = "Self-Contained"; })
            .Case<hw::HWModuleExternOp>(
                [&](auto mod) { moduleJson["label"] = "External"; })
            .Case<hw::HWModuleGeneratedOp>(
                [&](auto mod) { moduleJson["label"] = "External (Generated)";
    }) .Default([&](auto mod) { moduleJson["label"] = "Unknown Module"; });
        outputJsonObjects.push_back(std::move(moduleJson));
        nextNodeId++;
      }
    }
    */
    return wrapJson(outputJsonObjects);
  }

protected:
  NodeRef baseOriginalModule;
  NodeRef baseNewModule;
  std::stack<std::pair<NodeRef, int64_t>> modulesToProcess;
};

} // end anonymous namespace

namespace circt {
namespace hw {

// Public API functions now simply instantiate the corresponding generator.
std::string MlirToInstanceGraphJson(NodeRef baseModule, llvm::raw_ostream *os) {
  InstanceGraphGenerator generator(baseModule, os);
  return generator.generateGraphJson();
}

std::string MlirToOperationGraphJson(NodeRef baseModule,
                                     llvm::raw_ostream *os) {
  OperationGraphGenerator generator(baseModule, os);
  return generator.generateGraphJson();
}

std::string MlirInstanceDiffGraphJson(NodeRef baseOriginalModule,
                                      NodeRef baseNewModule,
                                      llvm::raw_ostream *os) {
  InstanceDiffGraphGenerator generator(baseOriginalModule, baseNewModule, os);
  return generator.generateGraphJson();
}

} // end namespace hw
} // end namespace circt
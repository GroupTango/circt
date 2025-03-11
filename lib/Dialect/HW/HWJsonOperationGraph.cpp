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
using namespace circt::hw::detail;

using HWModuleOpGraphTraits = llvm::GraphTraits<hw::HWModuleOp>;
using HWModuleOpJSONGraphTraits = hw::JSONGraphTraits<hw::HWModuleOp>;

namespace {

// Graph generator for Operation Graphs.
class OperationGraphGenerator : public GraphGenerator {
public:
  OperationGraphGenerator(HWOperationRef baseModule, llvm::raw_ostream *os)
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
  HWOperationRef baseModule;
  std::stack<
      std::tuple<HWOperationRef, std::string, HWOperationRef, std::string>>
      modulesToDrawIO;
  std::deque<
      std::tuple<HWOperationRef, std::string, HWOperationRef, std::string>>
      modulesToProcess;

  // lmao this is such wack code. can probably use a union or sth
  llvm::DenseMap<HWOperationRef, std::vector<HWOperationRef>> incomingEdges;
  // incomingFromIOEdges[Node] = vector[ (name of IO port, namespace diff
  // between node and IO port) ]
  llvm::DenseMap<HWOperationRef, std::set<std::pair<std::string, std::string>>>
      incomingFromIOEdges;
  // ioIncomingEdges[ IO port ID ] = vector[ (Node, namespace) ]
  llvm::StringMap<std::vector<std::pair<HWOperationRef, std::string>>>
      ioIncomingEdges;
  // Currently used for output to input only. iOFromIOEdges[ Input port ID ] =
  // vector[ Output port IDs ]
  llvm::StringMap<std::vector<std::string>> iOFromIOEdges;
  HWModuleOpJSONGraphTraits jsonGraphTraits;
  llvm::DenseMap<HWOperationRef, std::vector<std::string>> incomingInputEdges;
  llvm::StringMap<HWOperationRef> idToNodeMap;
  llvm::StringMap<std::vector<HWOperationRef>> outputIncomingEdges;

  std::vector<std::pair<HWOperationRef, std::string>> combOps;
  std::map<std::string, std::string> combOpIdMap;
  std::set<std::pair<HWOperationRef, std::string>> globalVisited;

  /*
   * Collects a group of comb operations starting from the given operation.
   */
  std::vector<HWOperationRef> collectCombGroup(HWOperationRef op,
                                               const std::string &ns) {
    std::stack<HWOperationRef> stack;
    std::vector<HWOperationRef> group;
    stack.push(op);

    while (!stack.empty()) {
      HWOperationRef currentOp = stack.top();
      stack.pop();

      if (this->globalVisited.count(
              std::pair<HWOperationRef, std::string>(currentOp, ns)))
        continue;
      globalVisited.insert(
          std::pair<HWOperationRef, std::string>(currentOp, ns));

      if (isCombOp(currentOp))
        group.push_back(currentOp);

      for (auto it = HWModuleOpGraphTraits::child_begin(currentOp),
                end = HWModuleOpGraphTraits::child_end(currentOp);
           it != end; ++it) {
        HWOperationRef succ = *it;
        // Only follow successors if they are comb ops.
        if (isCombOp(succ))
          stack.push(succ);
      }
    }

    return group;
  }

  void processCombGroups() {
    for (auto &pair : combOps) {
      HWOperationRef op = pair.first;
      std::string ns = pair.second;

      if (globalVisited.count(std::pair<HWOperationRef, std::string>(op, ns)))
        continue;

      if (incomingEdges[op].size() == 0)
        continue;

      std::vector<HWOperationRef> group =
          collectCombGroup(incomingEdges[op][0], ns);
      if (group.size() == 0)
        continue;

      // Generate a comb group node.
      std::string groupNs = ns + "/" + getUniqueId(op, "CombGroup");

      for (HWOperationRef combOp : group) {
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
    baseModule->walk([&](HWOperationRef op) {
      auto id = mlir::IntegerAttr::get(
          mlir::IntegerType::get(op->getContext(), 64), counter++);
      op->setAttr("hw.unique_id", id);
    });

    forEachOperation(baseModule, [&](mlir::Operation &op) {
      llvm::TypeSwitch<HWOperationRef>(&op)
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
              *os << "Found unknown module: "
                  << op.getAttrOfType<StringAttr>(
                           mlir::SymbolTable::getSymbolAttrName())
                         .getValue()
                  << "\n";
            modulesToProcess.push_back(
                {nullptr,
                 op.getAttrOfType<StringAttr>(
                       mlir::SymbolTable::getSymbolAttrName())
                     .getValue()
                     .str(),
                 nullptr, ""});
          });
    });

    for (auto const &entry : moduleMap) {
      HWOperationRef op = moduleMap[entry.getKey()];
      std::string namespaceStr = entry.getKey().str();
      modulesToDrawIO.push({op, namespaceStr, nullptr, ""});
      if (os)
        *os << "Adding top level Module for "
               "processing - Name: "
            << entry.getKey() << " Type: " << op->getName() << "\n";
    }
  }

  void drawModuleIO(HWOperationRef module, const std::string &ns,
                    HWOperationRef parentInstance,
                    const std::string &parentNs) {
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
      HWOperationRef node = &op;
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
  void processModule(HWOperationRef module, const std::string &ns,
                     HWOperationRef parentInstance,
                     const std::string &parentNs) {
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
      HWOperationRef node = &op;
      // Skip over hw.instance and hw.output: these are already handled by
      // the recursive module functionality, and the output node drawer.
      if (auto inst = mlir::dyn_cast<InstanceOp>(node)) {

      } else if (auto output = mlir::dyn_cast<OutputOp>(node)) {

      }
      // We collect comb ops to be grouped later.
      else if (node->getDialect() &&
               node->getDialect()->getNamespace() == "comb") {
        combOps.push_back(std::pair<HWOperationRef, std::string>(node, ns));
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
  llvm::json::Array getIncomingEdges(HWOperationRef node,
                                     const std::string &ns) {
    llvm::json::Array edges;
    std::string ns1 = ns;
    if (isCombOp(node)) {
      // The node's namespace contains the comb group, we want the original
      // namespace.
      size_t lastSlashPos = ns.find_last_of('/');
      if (lastSlashPos != std::string::npos)
        ns1 = ns.substr(0, lastSlashPos);
    }
    for (HWOperationRef parent : incomingEdges[node]) {
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
    for (const std::pair<HWOperationRef, std::string> &parent :
         ioIncomingEdges[portId]) {
      std::string uniqueId = getUniqueId(parent.first, parent.second);
      edges.push_back(llvm::json::Object{
          {"sourceNodeId",
           isCombOp(parent.first) ? combOpIdMap[uniqueId] : uniqueId},
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
    if (HWOperationRef operSource = outputOper.getDefiningOp()) {
      outputIncomingEdges[outputId].push_back(operSource);
      if (os)
        *os << "Operation " << operSource->getName().getStringRef().str()
            << " used output port " << outputId << "\n";

      for (HWOperationRef parent : outputIncomingEdges[outputId]) {
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
                  HWOperationRef parentInstance, const std::string &parentNs) {

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

        if (HWOperationRef operSource = oper.getDefiningOp()) {

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
                     HWOperationRef parentInstance,
                     const std::string &parentNs) {

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
                   const std::string &ns, HWOperationRef parentInstance,
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

      if (HWOperationRef sourceOp = outputOper.getDefiningOp()) {
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
        for (HWOperationRef user : result.getUsers()) {
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
                      HWOperationRef parentInstance,
                      const std::string &parentNs) {

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
      llvm::DenseMap<HWOperationRef, std::vector<HWOperationRef>> &edgesMap) {
    std::stack<HWOperationRef> nodesToVisit;
    llvm::SmallPtrSet<HWOperationRef, 32> visited;
    for (auto it = HWModuleOpGraphTraits::nodes_begin(module),
              end = HWModuleOpGraphTraits::nodes_end(module);
         it != end; ++it) {
      nodesToVisit.push(*it);
    }
    while (!nodesToVisit.empty()) {
      HWOperationRef current = nodesToVisit.top();
      nodesToVisit.pop();
      if (!visited.insert(current).second)
        continue;
      for (auto it = HWModuleOpGraphTraits::child_begin(current),
                end = HWModuleOpGraphTraits::child_end(current);
           it != end; ++it) {
        HWOperationRef child = *it;
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

} // namespace

namespace circt {
namespace hw {

// Public API functions instantiates the corresponding generator.
std::string MlirToOperationGraphJson(HWOperationRef baseModule,
                                     llvm::raw_ostream *os) {
  OperationGraphGenerator generator(baseModule, os);
  return generator.generateGraphJson();
}

} // end namespace hw
} // end namespace circt
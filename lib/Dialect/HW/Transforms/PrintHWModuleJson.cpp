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
#include "llvm/ADT/StringMap.h"

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
// graph.
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

    //NEED TO DO SOME CHANGES TO DISPLAY MORE INFO MISSING MODULES BETTER (Currently passing nullptr so can't print any info such as missing module names)

    //Locate all modules in MLIR
    llvm::StringMap<mlir::Operation*> moduleMap;

    //Built in walkers are annoying in that they don't allow us to get back pointers to ops, so we need to traverse the top level ModuleOP ourselves
    //All base module types are guaranteed to be on this top level region
    mlir::Operation* baseOp = getOperation();
    for (mlir::Region &region : baseOp->getRegions())
    {
      for (mlir::Block &block : region.getBlocks())
      {
        for (mlir::Operation& op : block.getOperations())
        {
          llvm::TypeSwitch<mlir::Operation*>(&op)
          .Case<circt::hw::HWModuleOp>([&](auto module) { 
            os << "Found HWModuleOp: " << module.getName() << "\n"; 
            moduleMap[module.getName()] = &op;
          })
          .Case<circt::hw::HWModuleExternOp>([&](auto module) { os << "Found HWModuleExternOp: " << module.getName() << " SKIPPING\n"; })
          .Case<circt::hw::HWModuleGeneratedOp>([&](auto module) { os << "Found HWModuleGeneratedOp: " << module.getName() << " SKIPPING\n"; })
          .Default([&](auto) { os << "Found unknown top level module type: " << op.getName() << " SKIPPING\n"; });
        }
      }
    }  

    //Start processing Modules to JSON
    std::vector<std::pair<mlir::Operation*, std::string>> modulesToProcess;
    llvm::json::Array outputJsonObjects;
    uint64_t nextNodeId = 0;    

    //Note: Iterator only returns keys, not values
    for (auto const& x : moduleMap)
    {
      mlir::Operation* op = moduleMap[x.getKey()];
      modulesToProcess.push_back(std::make_pair(op, x.getKey().str()));

      os << "Adding top level Module for processing - Name: " << x.getKey() << " Type: " << op->getName() << "\n";
    }

    while (modulesToProcess.size() > 0)
    {
      std::pair<mlir::Operation*, std::string> nextPair = modulesToProcess.back();
      modulesToProcess.pop_back();
      mlir::Operation* module = nextPair.first;      

      if (module == nullptr)
      {
        llvm::json::Object moduleJson;      
        moduleJson["id"] = std::to_string(nextNodeId);
        moduleJson["label"] = "Unknown Module"; //Change to display said module name in future?
        moduleJson["namespace"] = nextPair.second;

        outputJsonObjects.push_back(std::move(moduleJson));
        nextNodeId++;     
        continue;
      }

      bool hasInstances = false;

      //os << "   Regions: " << module->getRegions().size() << "\n";
      for (mlir::Region &region : module->getRegions())
      {
        //os << "       Blocks: " << region.getBlocks().size() << "\n";
        for (mlir::Block &block : region.getBlocks())
        {
          auto filteredOps = block.getOps<circt::hw::InstanceOp>();
          for (circt::hw::InstanceOp instanceOp : filteredOps) 
          {
            hasInstances = true;

            //Is it possible to get the value of of the iterator, isntead of having to [] again? Also would adding std::move around the string concat change anything?
            auto it = moduleMap.find(instanceOp.getReferencedModuleName());
            if (it != moduleMap.end()) modulesToProcess.push_back(std::make_pair(moduleMap[instanceOp.getReferencedModuleName()], nextPair.second + "/" + instanceOp.getReferencedModuleName().str()));
            else modulesToProcess.push_back(std::make_pair(nullptr, nextPair.second + "/" + instanceOp.getReferencedModuleName().str()));
          }
        }
      }
      
      //if (mlir::dyn_cast<InstanceOp>())

      //If this is a self contained module, we will display it as a graph node.
      if (!hasInstances)
      {
        llvm::json::Object moduleJson;      
        moduleJson["id"] = std::to_string(nextNodeId);
        moduleJson["label"] = "Self-Contained";
        moduleJson["namespace"] = nextPair.second;

        //moduleJson["attributes"] = jsonGraphTraits.getNodeAttributes(module, module);

        outputJsonObjects.push_back(std::move(moduleJson));
        nextNodeId++;        
      }
    }    

    //Do some final wraps of our Json Node Array, as needed by Model Explorer
    
    llvm::json::Object graphWrapper;
    graphWrapper["id"] = "test_mlir_file";    
    graphWrapper["nodes"] = std::move(outputJsonObjects);

    llvm::json::Array graphArrayWrapper;
    graphArrayWrapper.push_back(std::move(graphWrapper));

    llvm::json::Object fileWrapper;
    fileWrapper["label"] = "model.json";
    fileWrapper["subgraphs"] = std::move(graphArrayWrapper);

    //Output final JSON
    os << "JSON:\n\n";
    os << "[" << llvm::json::Value(std::move(fileWrapper)) << "]";
    
    //We print out the raw MLIR later for some reason? New lines to space out the raw MLIR
    os << "\n\n";

    /*getOperation().walk([&](mlir::Operation* module) {      
      llvm::SmallPtrSet<NodeRef, 16> visited;

      llvm::json::Object moduleJson;
      llvm::json::Array moduleNodes;
      moduleJson["name"] = module.getNameAttr().getValue();
      moduleJson["label"] = jsonGraphTraits.getNodeLabel(module, module);
      moduleJson["attributes"] =
          jsonGraphTraits.getNodeAttributes(module, module);

      // Iterate over all top-level nodes in the module.
      for (auto it = HWModuleOpGraphTraits::nodes_begin(module),
                end = HWModuleOpGraphTraits::nodes_end(module);
           it != end; ++it) {
        NodeRef node = *it;
        if (visited.count(node) == 0)
          moduleNodes.push_back(visitNode(node, module, visited));
      }
      moduleJson["children"] = std::move(moduleNodes);

      // Output the JSON representation of the module's graph.
      os << llvm::json::Value(std::move(moduleJson)) << "\n";
      
    });
    */
  }

  
  llvm::json::Object visitNode(NodeRef node, const HWModuleOp &module,
                               llvm::SmallPtrSetImpl<NodeRef> &visited) {
    if (visited.count(node) > 0)
      // TODO: should copy the node's JSON representation that we already
      // created
      return llvm::json::Object();

    visited.insert(node);

    llvm::json::Object json;
    json["name"] = node->getName().getStringRef();
    json["label"] = jsonGraphTraits.getNodeLabel(node, module);
    json["attributes"] = jsonGraphTraits.getNodeAttributes(node, module);

    llvm::json::Array children;
    for (auto it = HWModuleOpGraphTraits::child_begin(node),
              end = HWModuleOpGraphTraits::child_end(node);
         it != end; ++it) {
      NodeRef child = *it;
      llvm::json::Object childJson = visitNode(child, module, visited);
      children.push_back(std::move(childJson));
    }
    json["children"] = std::move(children);
    return json;
  }

  raw_ostream &os;
  HWModuleOpJSONGraphTraits jsonGraphTraits;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::hw::createPrintHWModuleJsonPass() {
  return std::make_unique<PrintHWModuleJsonPass>(llvm::errs());
}

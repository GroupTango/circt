//===- HWModuleGraph.h - HWModule graph -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the HWModuleGraph.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWMODULEGRAPH_H
#define CIRCT_DIALECT_HW_HWMODULEGRAPH_H

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/JSONGraphTraits.h"
#include "circt/Support/LLVM.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/DOTGraphTraits.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace hw {
namespace detail {

// Using declaration to avoid polluting global namespace with CIRCT-specific
// graph traits for mlir::Operation.
using HWOperation = mlir::Operation;
using HWOperationRef = mlir::Operation *;

// Shallow iteration over all operations in the top-level module.
template <typename Fn>
void forEachOperation(HWOperationRef op, Fn f) {
  for (mlir::Region &region : op->getRegions())
    for (mlir::Block &block : region.getBlocks())
      for (mlir::Operation &childOp : block.getOperations())
        f(childOp);
}

} // namespace detail
} // namespace hw
} // namespace circt

template <>
struct llvm::GraphTraits<circt::hw::detail::HWOperation *> {
  using NodeType = circt::hw::detail::HWOperation;
  using NodeRef = NodeType *;

  using ChildIteratorType = mlir::Operation::user_iterator;
  static NodeRef getEntryNode(NodeRef op) { return op; }
  static ChildIteratorType child_begin(NodeRef op) { return op->user_begin(); }
  static ChildIteratorType child_end(NodeRef op) { return op->user_end(); }
};

template <>
struct llvm::GraphTraits<circt::hw::HWModuleOp>
    : public llvm::GraphTraits<circt::hw::detail::HWOperation *> {
  using GraphType = circt::hw::HWModuleOp;

  static NodeRef getEntryNode(GraphType mod) {
    return &mod.getBodyBlock()->front();
  }

  using nodes_iterator = pointer_iterator<mlir::Block::iterator>;
  static nodes_iterator nodes_begin(GraphType mod) {
    return nodes_iterator{mod.getBodyBlock()->begin()};
  }
  static nodes_iterator nodes_end(GraphType mod) {
    return nodes_iterator{mod.getBodyBlock()->end()};
  }
};

template <>
struct llvm::DOTGraphTraits<circt::hw::HWModuleOp>
    : public llvm::DefaultDOTGraphTraits {
  using DefaultDOTGraphTraits::DefaultDOTGraphTraits;

  static std::string getNodeLabel(circt::hw::detail::HWOperation *node,
                                  circt::hw::HWModuleOp) {
    return llvm::TypeSwitch<mlir::Operation *, std::string>(node)
        .Case<circt::comb::AddOp>([&](auto) { return "+"; })
        .Case<circt::comb::SubOp>([&](auto) { return "-"; })
        .Case<circt::comb::AndOp>([&](auto) { return "&"; })
        .Case<circt::comb::OrOp>([&](auto) { return "|"; })
        .Case<circt::comb::XorOp>([&](auto) { return "^"; })
        .Case<circt::comb::MulOp>([&](auto) { return "*"; })
        .Case<circt::comb::MuxOp>([&](auto) { return "mux"; })
        .Case<circt::comb::ShrSOp, circt::comb::ShrUOp>(
            [&](auto) { return ">>"; })
        .Case<circt::comb::ShlOp>([&](auto) { return "<<"; })
        .Case<circt::comb::ICmpOp>([&](auto op) {
          switch (op.getPredicate()) {
          case circt::comb::ICmpPredicate::eq:
          case circt::comb::ICmpPredicate::ceq:
          case circt::comb::ICmpPredicate::weq:
            return "==";
          case circt::comb::ICmpPredicate::wne:
          case circt::comb::ICmpPredicate::cne:
          case circt::comb::ICmpPredicate::ne:
            return "!=";
          case circt::comb::ICmpPredicate::uge:
          case circt::comb::ICmpPredicate::sge:
            return ">=";
          case circt::comb::ICmpPredicate::ugt:
          case circt::comb::ICmpPredicate::sgt:
            return ">";
          case circt::comb::ICmpPredicate::ule:
          case circt::comb::ICmpPredicate::sle:
            return "<=";
          case circt::comb::ICmpPredicate::ult:
          case circt::comb::ICmpPredicate::slt:
            return "<";
          }
          llvm_unreachable("unhandled ICmp predicate");
        })
        .Case<circt::seq::FirRegOp>([&](auto op) { return op.getName().str(); })
        .Case<circt::seq::CompRegOp>([&](auto op) -> std::string {
          if (auto name = op.getName())
            return name->str();
          return "reg";
        })
        .Case<circt::hw::ConstantOp>([&](auto op) {
          llvm::SmallString<64> valueString;
          op.getValue().toString(valueString, 10, false);
          return valueString.str().str();
        })
        .Default([&](auto op) { return op->getName().getStringRef().str(); });
  }

  std::string getNodeAttributes(circt::hw::detail::HWOperation *node,
                                circt::hw::HWModuleOp) {
    return llvm::TypeSwitch<mlir::Operation *, std::string>(node)
        .Case<circt::hw::ConstantOp>(
            [&](auto) { return "fillcolor=darkgoldenrod1,style=filled"; })
        .Case<circt::comb::MuxOp>([&](auto) {
          return "shape=invtrapezium,fillcolor=bisque,style=filled";
        })
        .Case<circt::hw::OutputOp>(
            [&](auto) { return "fillcolor=lightblue,style=filled"; })
        .Default([&](auto op) {
          return llvm::TypeSwitch<mlir::Dialect *, std::string>(
                     op->getDialect())
              .Case<circt::comb::CombDialect>([&](auto) {
                return "shape=oval,fillcolor=bisque,style=filled";
              })
              .template Case<circt::seq::SeqDialect>([&](auto) {
                return "shape=folder,fillcolor=gainsboro,style=filled";
              })
              .Default([&](auto) { return ""; });
        });
  }

  static void
  addCustomGraphFeatures(circt::hw::HWModuleOp mod,
                         llvm::GraphWriter<circt::hw::HWModuleOp> &g) {

    // Add module input args.
    auto &os = g.getOStream();
    os << "subgraph cluster_entry_args {\n";
    os << "label=\"Input arguments\";\n";
    circt::hw::ModulePortInfo iports(mod.getPortList());
    for (auto [info, arg] :
         llvm::zip(iports.getInputs(), mod.getBodyBlock()->getArguments())) {
      g.emitSimpleNode(reinterpret_cast<void *>(&arg), "",
                       info.getName().str());
    }
    os << "}\n";
    for (auto [info, arg] :
         llvm::zip(iports.getInputs(), mod.getBodyBlock()->getArguments())) {
      for (auto *user : arg.getUsers()) {
        g.emitEdge(reinterpret_cast<void *>(&arg), 0, user, -1, "");
      }
    }
  }

  template <typename Iterator>
  static std::string getEdgeAttributes(circt::hw::detail::HWOperation *node,
                                       Iterator it, circt::hw::HWModuleOp mod) {

    mlir::OpOperand &operand = *it.getCurrent();
    mlir::Value v = operand.get();
    std::string str;
    llvm::raw_string_ostream os(str);
    auto verboseEdges = mod->getAttrOfType<mlir::BoolAttr>("dot_verboseEdges");
    if (verboseEdges.getValue()) {
      os << "label=\"" << operand.getOperandNumber() << " (" << v.getType()
         << ")\"";
    }

    int64_t width = circt::hw::getBitWidth(v.getType());
    if (width > 1)
      os << " style=bold";

    return os.str();
  }
};

template <>
struct circt::hw::JSONGraphTraits<circt::hw::HWModuleOp>
    : public llvm::DOTGraphTraits<circt::hw::HWModuleOp> {
  JSONGraphTraits(bool isSimple = false) : DOTGraphTraits(isSimple) {}

  // Same attributes from DOTGraphTraits, but in JSON format.
  llvm::json::Array getNodeAttributes(circt::hw::detail::HWOperation *node,
                                      HWModuleOp mod) {
    return llvm::TypeSwitch<mlir::Operation *, llvm::json::Array>(node)
        .Case<circt::hw::ConstantOp>([&](auto op) -> llvm::json::Array {
          return llvm::json::Array(
              {llvm::json::Object(
                   {{"key", "fillcolor"}, {"value", "darkgoldenrod1"}}),
               llvm::json::Object({{"key", "style"}, {"value", "filled"}}),
               llvm::json::Object({{"key", "type"}, {"value", "hw"}})
              });
        })
        .Case<circt::comb::MuxOp>([&](auto op) -> llvm::json::Array {
          return llvm::json::Array({
              llvm::json::Object({{"key", "shape"}, {"value", "invtrapezium"}}),
              llvm::json::Object({{"key", "fillcolor"}, {"value", "bisque"}}),
              llvm::json::Object({{"key", "style"}, {"value", "filled"}}),
              llvm::json::Object({{"key", "type"}, {"value", "comb"}})
          });
        })
        .Case<circt::hw::OutputOp>([&](auto op) -> llvm::json::Array {
          return llvm::json::Array({
              llvm::json::Object(
                  {{"key", "fillcolor"}, {"value", "lightblue"}}),
              llvm::json::Object({{"key", "style"}, {"value", "filled"}}),
              llvm::json::Object({{"key", "type"}, {"value", "hw"}})
          });
        })
        .Default([&](auto op) -> llvm::json::Array {
          return llvm::TypeSwitch<mlir::Dialect *, llvm::json::Array>(
                     op->getDialect())
              .Case<circt::comb::CombDialect>([&](auto) -> llvm::json::Array {
                return llvm::json::Array({
                    llvm::json::Object({{"key", "shape"}, {"value", "oval"}}),
                    llvm::json::Object(
                        {{"key", "fillcolor"}, {"value", "bisque"}}),
                    llvm::json::Object({{"key", "style"}, {"value", "filled"}}),
                    llvm::json::Object({{"key", "type"}, {"value", "comb"}})
                });
              })
              .template Case<circt::seq::SeqDialect>([&](auto)
                                                         -> llvm::json::Array {
                return llvm::json::Array({
                    llvm::json::Object({{"key", "shape"}, {"value", "folder"}}),
                    llvm::json::Object(
                        {{"key", "fillcolor"}, {"value", "gainsboro"}}),
                    llvm::json::Object({{"key", "style"}, {"value", "filled"}}),
                    llvm::json::Object({{"key", "type"}, {"value", "seq"}})
                });
              })
              .Default([&](auto) -> llvm::json::Array {
                return llvm::json::Array();
              });
        });
  }

  llvm::json::Array getInputNodeAttributes() {
    return llvm::json::Array({llvm::json::Object({{"key", "type"}, {"value", "I/O"}})});
  }

  template <typename Iterator>
  llvm::json::Object getEdgeAttributes(circt::hw::detail::HWOperation *node,
                                       Iterator it, HWModuleOp mod) {
    mlir::OpOperand &operand = *it.getCurrent();
    mlir::Value v = operand.get();
    llvm::json::Object obj;

    auto verboseEdges = mod->getAttrOfType<mlir::BoolAttr>("dot_verboseEdges");
    if (verboseEdges.getValue()) {
      std::string str;
      llvm::raw_string_ostream os(str);
      os << operand.getOperandNumber() << " (" << v.getType() << ")";
      obj["label"] = os.str();
    }

    int64_t width = circt::hw::getBitWidth(v.getType());
    if (width > 1)
      obj["style"] = "bold";

    return obj;
  }
};

class GraphGenerator {
public:
  GraphGenerator(llvm::raw_ostream *os) : os(os), nextNodeId(0) {}

  virtual ~GraphGenerator() = default;

  // Main entry point: initialize, process modules, and wrap the output.
  virtual std::string generateGraphJson() = 0;

protected:
  llvm::raw_ostream *os;
  llvm::StringMap<circt::hw::detail::HWOperationRef> moduleMap;
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
  std::string getUniqueId(circt::hw::detail::HWOperationRef node, const std::string &ns) {
    if (ns.empty())
      return "";

    if (!node)
      return ns + "_" + std::to_string(nextNodeId++);

    return ns + "_" +
           std::to_string(
               mlir::cast<circt::IntegerAttr>(node->getAttr("hw.unique_id")).getInt());
  }

  bool isCombOp(circt::hw::detail::HWOperationRef op) {
    return op->getName().getDialectNamespace() == "comb";
  }
};

#endif // CIRCT_DIALECT_HW_HWMODULEGRAPH_H

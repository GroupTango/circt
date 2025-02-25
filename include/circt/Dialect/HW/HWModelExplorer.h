#ifndef CIRCT_DIALECT_HW_MODELEXPLORER_H
#define CIRCT_DIALECT_HW_MODELEXPLORER_H

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace hw {

std::string MlirToInstanceGraphJson(mlir::Operation *baseModule,
                                    llvm::raw_ostream *os = nullptr);

std::string MlirToOperationGraphJson(mlir::Operation *baseModule,
                                     llvm::raw_ostream *os = nullptr);

std::string MlirDiffGraphJson(mlir::Operation *baseOriginalModule, 
                              mlir::Operation *baseNewModule,
                              llvm::raw_ostream *os = nullptr);

} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_HW_MODELEXPLORER_H

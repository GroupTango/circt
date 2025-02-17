#ifndef CIRCT_DIALECT_HW_MODELEXPLORERINTERFACES_H
#define CIRCT_DIALECT_HW_MODELEXPLORERINTERFACES_H

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace circt {
namespace hw {

std::string MlirToInstanceGraphJson(mlir::Operation *baseModule,
                                    llvm::raw_ostream *os = nullptr);

std::string MlirToOperationGraphJson(mlir::Operation *baseModule,
                                     llvm::raw_ostream *os = nullptr);

} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_HW_MODELEXPLORERINTERFACES_H

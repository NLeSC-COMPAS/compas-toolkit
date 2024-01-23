#include <cuda.h>

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "context.h"

namespace compas {

CudaContext make_context(int device) {
    return {kmm::build_runtime(), kmm::Cuda(device)};
}

}  // namespace compas
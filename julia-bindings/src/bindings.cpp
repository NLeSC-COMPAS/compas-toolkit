#include "context.h"
#include "jlcxx/jlcxx.hpp"

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
    mod.add_type<compas::CudaContext>("CudaContext");
    mod.method("make_context", compas::make_context);
}
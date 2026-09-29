#pragma once

#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"

#include <filesystem>
#include <memory>
#include <string>

// The environment must outlive the compiled model; members are destroyed in
// reverse order, so `model` goes first.
struct CompiledLiteRt {
    std::unique_ptr<litert::Environment> env;
    std::unique_ptr<litert::CompiledModel> model;
};

// 0 or less means one thread per hardware thread.
int resolve_threads(int threads);

// CPU (XNNPACK) compiled model. With a weight cache file, XNNPACK maps one
// packed copy of the weights from disk instead of packing them into anonymous
// memory once per signature.
CompiledLiteRt make_compiled_model(const std::filesystem::path& model_file, int threads,
                                const std::string& weight_cache);

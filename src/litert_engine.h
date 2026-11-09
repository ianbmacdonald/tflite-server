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
    // What the model actually compiled for: "cpu", or "npu" when the NPU
    // path came up (it can differ from the request after a fallback).
    std::string accelerator = "cpu";
};

// Which hardware runs inference. "cpu" is the XNNPACK path and the default.
// "npu" compiles the model for the platform NPU through LiteRT's dispatch
// runtime, which loads the platform's libLiteRtDispatch_*.so at run time;
// dispatch_library_dir, when set, is the directory LiteRT searches for it.
// An unavailable NPU falls back to CPU with one line on stderr.
struct AcceleratorConfig {
    std::string kind = "cpu";  // "cpu" or "npu"
    std::string dispatch_library_dir;
};

// 0 or less means one thread per hardware thread.
int resolve_threads(int threads);

// Compiled model on the accelerator `accel` selects; CPU (XNNPACK) by
// default. With a weight cache file, XNNPACK maps one packed copy of the
// weights from disk instead of packing them into anonymous memory once per
// signature (the cache applies to the CPU path only).
CompiledLiteRt make_compiled_model(const std::filesystem::path& model_file, int threads,
                                const std::string& weight_cache,
                                const AcceleratorConfig& accel = {});

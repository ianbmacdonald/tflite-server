#include "litert_engine.h"

#include "model_variant.h"

#include "litert/cc/litert_common.h"
#include "litert/cc/litert_environment_options.h"
#include "litert/cc/litert_options.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

int resolve_threads(int threads) {
    if (threads > 0) return threads;
    return static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
}

namespace {

CompiledLiteRt compile_on(bool npu, const std::filesystem::path& model_file, int threads,
                          const std::string& weight_cache, const AcceleratorConfig& accel) {
    CompiledLiteRt m;
    std::vector<litert::EnvironmentOptions::Option> env_options;
    if (npu && !accel.dispatch_library_dir.empty()) {
        env_options.push_back({litert::EnvironmentOptions::Tag::kDispatchLibraryDir,
                               accel.dispatch_library_dir.c_str()});
    }
    auto env = litert::Environment::Create(litert::EnvironmentOptions(env_options));
    if (!env) throw std::runtime_error("LiteRT environment: " + env.Error().Message());
    m.env = std::make_unique<litert::Environment>(std::move(*env));
    auto options = litert::Options::Create();
    if (!options) throw std::runtime_error("LiteRT options: " + options.Error().Message());
    if (npu) {
        if (!options->SetHardwareAccelerators(litert::HwAccelerators::kNpu)) {
            throw std::runtime_error("cannot select the NPU accelerator");
        }
    } else {
        options->SetHardwareAccelerators(litert::HwAccelerators::kCpu);
        auto cpu = options->GetCpuOptions();
        if (!cpu || !cpu->SetNumThreads(resolve_threads(threads))) {
            throw std::runtime_error("cannot set the LiteRT CPU thread count");
        }
        if (!weight_cache.empty() && !cpu->SetXNNPackWeightCachePath(weight_cache.c_str())) {
            throw std::runtime_error("cannot set the XNNPACK weight cache path");
        }
    }
    auto compiled = litert::CompiledModel::Create(*m.env, model_file.string(), *options);
    if (!compiled) {
        throw std::runtime_error("cannot load " + model_file.string() + ": " +
                                 compiled.Error().Message());
    }
    m.model = std::make_unique<litert::CompiledModel>(std::move(*compiled));
    m.accelerator = npu ? "npu" : "cpu";
    return m;
}

}  // namespace

CompiledLiteRt make_compiled_model(const std::filesystem::path& model_file, int threads,
                                const std::string& weight_cache,
                                const AcceleratorConfig& accel) {
    if (accel.kind != "cpu" && accel.kind != "npu") {
        throw std::runtime_error("unknown accelerator '" + accel.kind + "' (use cpu or npu)");
    }
    if (accel.kind == "npu") {
        // An ahead-of-time payload, when the model directory ships one; the
        // CPU fallback below keeps the portable model, which is the only one
        // it can run.
        const std::filesystem::path npu_file = select_model_file(model_file, true);
        try {
            return compile_on(true, npu_file, threads, weight_cache, accel);
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                         "tflite-server: NPU unavailable for %s (%s); falling back to CPU\n",
                         npu_file.string().c_str(), e.what());
        }
    }
    return compile_on(false, model_file, threads, weight_cache, accel);
}

#include "litert_engine.h"

#include "litert/cc/litert_common.h"
#include "litert/cc/litert_options.h"

#include <algorithm>
#include <stdexcept>
#include <thread>

int resolve_threads(int threads) {
    if (threads > 0) return threads;
    return static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
}

CompiledLiteRt make_compiled_model(const std::filesystem::path& model_file, int threads,
                                const std::string& weight_cache) {
    CompiledLiteRt m;
    auto env = litert::Environment::Create({});
    if (!env) throw std::runtime_error("LiteRT environment: " + env.Error().Message());
    m.env = std::make_unique<litert::Environment>(std::move(*env));
    auto options = litert::Options::Create();
    if (!options) throw std::runtime_error("LiteRT options: " + options.Error().Message());
    options->SetHardwareAccelerators(litert::HwAccelerators::kCpu);
    auto cpu = options->GetCpuOptions();
    if (!cpu || !cpu->SetNumThreads(resolve_threads(threads))) {
        throw std::runtime_error("cannot set the LiteRT CPU thread count");
    }
    if (!weight_cache.empty() && !cpu->SetXNNPackWeightCachePath(weight_cache.c_str())) {
        throw std::runtime_error("cannot set the XNNPACK weight cache path");
    }
    auto compiled = litert::CompiledModel::Create(*m.env, model_file.string(), *options);
    if (!compiled) {
        throw std::runtime_error("cannot load " + model_file.string() + ": " +
                                 compiled.Error().Message());
    }
    m.model = std::make_unique<litert::CompiledModel>(std::move(*compiled));
    return m;
}

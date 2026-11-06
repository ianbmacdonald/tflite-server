#pragma once

#include <filesystem>
#include <system_error>

// Which model file an accelerator compiles.
//
// Some NPUs run only an ahead-of-time build of the model, compiled by a
// vendor tool and kept beside the portable one: "model.tflite" gets a
// "model.npu.tflite" sibling. The NPU takes that payload when it is there;
// every other case, including the fallback to CPU, takes the portable model.
inline std::filesystem::path select_model_file(const std::filesystem::path& model_file, bool npu) {
    if (!npu) return model_file;
    std::filesystem::path payload = model_file;
    payload.replace_extension();        // model.tflite -> model
    payload += ".npu";                  //              -> model.npu
    payload += model_file.extension();  //              -> model.npu.tflite
    std::error_code ec;
    return std::filesystem::exists(payload, ec) ? payload : model_file;
}

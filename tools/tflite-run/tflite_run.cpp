// tflite-run: run one signature of a .tflite through the same LiteRT CompiledModel path, and the same
// selected-op registration, as tflite-server, on raw input files; every output is written as float32.
// It proves that a given op set runs a model tflite-server has no endpoint for.
//
//   tflite-run <model.tflite> <signature key | #index> [--threads N] [--out-shape <i>=<d0,d1,...>]...
//              -- <out_prefix> <name|#i>=<file>:<dtype>:<d0,d1,...> ... [-- <out_prefix> ...]
//
// dtype: f32 i32 i64 u8 i8 bool (must match the model input). A shape that differs from the model's
// input shape resizes that input (non-strict). Output i goes to <out_prefix>.<i>.f32. --out-shape sizes
// the buffer of an output whose shape depends on a resized input (LiteRT sizes output buffers from the
// model's static shape, e.g. YAMNet's [frames,64] log-mel output).
#include "litert_engine.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct InputSpec {
    std::string name, file, dtype;
    std::vector<int> shape;
};

size_t dtype_size(const std::string& d) {
    if (d == "f32" || d == "i32") return 4;
    if (d == "i64") return 8;
    if (d == "u8" || d == "i8" || d == "bool") return 1;
    return 0;
}

const char* et_name(litert::ElementType t) {
    switch (t) {
        case litert::ElementType::Float32: return "f32";
        case litert::ElementType::Int32: return "i32";
        case litert::ElementType::Int64: return "i64";
        case litert::ElementType::UInt8: return "u8";
        case litert::ElementType::Int8: return "i8";
        case litert::ElementType::Bool: return "bool";
        default: return "?";
    }
}

bool parse_spec(const std::string& s, InputSpec& o) {
    auto eq = s.find('='), c2 = s.rfind(':');
    if (eq == std::string::npos || c2 == std::string::npos || c2 < eq) return false;
    auto c1 = s.rfind(':', c2 - 1);
    if (c1 == std::string::npos || c1 < eq) return false;
    o.name = s.substr(0, eq);
    o.file = s.substr(eq + 1, c1 - eq - 1);
    o.dtype = s.substr(c1 + 1, c2 - c1 - 1);
    std::stringstream ss(s.substr(c2 + 1));
    for (std::string d; std::getline(ss, d, ',');) o.shape.push_back(std::atoi(d.c_str()));
    return dtype_size(o.dtype) != 0;
}

std::vector<float> to_float(const std::vector<uint8_t>& raw, litert::ElementType t) {
    std::vector<float> f;
    auto conv = [&](auto tag) {
        using T = decltype(tag);
        size_t n = raw.size() / sizeof(T);
        f.resize(n);
        for (size_t i = 0; i < n; ++i) {
            T v;
            std::memcpy(&v, raw.data() + i * sizeof(T), sizeof(T));
            f[i] = static_cast<float>(v);
        }
    };
    switch (t) {
        case litert::ElementType::Float32: conv(float{}); break;
        case litert::ElementType::Int32: conv(int32_t{}); break;
        case litert::ElementType::Int64: conv(int64_t{}); break;
        case litert::ElementType::UInt8: conv(uint8_t{}); break;
        case litert::ElementType::Int8: conv(int8_t{}); break;
        case litert::ElementType::Bool: conv(uint8_t{}); break;
        default: break;
    }
    return f;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
                     "usage: tflite-run <model.tflite> <signature|#index> [--threads N] -- <out_prefix> "
                     "<name|#i>=<file>:<dtype>:<d0,...> ... [-- ...]\n");
        return 2;
    }
    int threads = 2, a = 3;
    std::vector<std::pair<size_t, std::vector<int32_t>>> out_shapes;
    while (a + 1 < argc && std::string(argv[a]) != "--") {
        std::string opt = argv[a];
        if (opt == "--threads") {
            threads = std::atoi(argv[a + 1]);
        } else if (opt == "--out-shape") {
            std::string v = argv[a + 1];
            auto eq = v.find('=');
            std::vector<int32_t> dims;
            std::stringstream ss(v.substr(eq + 1));
            for (std::string d; std::getline(ss, d, ',');) dims.push_back(std::atoi(d.c_str()));
            out_shapes.push_back({std::strtoul(v.c_str(), nullptr, 10), dims});
        } else {
            std::fprintf(stderr, "unknown option %s\n", opt.c_str());
            return 2;
        }
        a += 2;
    }
    std::vector<std::pair<std::string, std::vector<InputSpec>>> runs;
    for (; a < argc; ++a) {
        if (std::string(argv[a]) == "--") {
            if (a + 1 >= argc) break;
            runs.push_back({argv[++a], {}});
            continue;
        }
        InputSpec s;
        if (runs.empty() || !parse_spec(argv[a], s)) {
            std::fprintf(stderr, "bad argument: %s\n", argv[a]);
            return 2;
        }
        runs.back().second.push_back(std::move(s));
    }
    try {
        auto lm = make_compiled_model(argv[1], threads, "");
        auto& model = *lm.model;
        std::string key = argv[2];
        size_t si = 0;
        if (!key.empty() && key[0] == '#') {
            si = std::strtoul(key.c_str() + 1, nullptr, 10);
        } else {
            auto idx = model.GetSignatureIndex(key);
            if (!idx) {
                std::fprintf(stderr, "no signature %s\n", key.c_str());
                return 1;
            }
            si = *idx;
        }
        auto in_names = model.GetSignatureInputNames(si);
        auto out_names = model.GetSignatureOutputNames(si);
        if (!in_names || !out_names) {
            std::fprintf(stderr, "cannot read signature %zu\n", si);
            return 1;
        }
        std::printf("tflite-run signature=%zu inputs=%zu outputs=%zu\n", si, in_names->size(),
                    out_names->size());
        // Buffers are created once and reused while the input shapes stay the same: re-creating them
        // after a dynamic-shape resize fails inside LiteRT on the second request.
        std::vector<litert::TensorBuffer> inputs_v, outputs_v;
        bool have_buffers = false;
        for (auto& [prefix, specs] : runs) {
            std::vector<size_t> index(specs.size());
            bool resized = false;
            for (size_t k = 0; k < specs.size(); ++k) {
                const auto& n = specs[k].name;
                if (!n.empty() && n[0] == '#') {
                    index[k] = std::strtoul(n.c_str() + 1, nullptr, 10);
                } else {
                    index[k] = SIZE_MAX;
                    for (size_t j = 0; j < in_names->size(); ++j)
                        if (std::string((*in_names)[j].data(), (*in_names)[j].size()) == n) index[k] = j;
                    if (index[k] == SIZE_MAX) {
                        std::fprintf(stderr, "no input named %s\n", n.c_str());
                        return 1;
                    }
                }
                auto lay = model.GetInputTensorLayout(si, index[k]);  // reflects earlier resizes
                if (!lay) {
                    std::fprintf(stderr, "cannot read input %zu layout\n", index[k]);
                    return 1;
                }
                std::vector<int> cur;
                for (auto d : lay->Dimensions()) cur.push_back(static_cast<int>(d));
                if (cur != specs[k].shape) {
                    auto r = model.ResizeInputTensorNonStrict(
                        si, index[k], litert::Span<const int>(specs[k].shape.data(), specs[k].shape.size()));
                    if (!r) {
                        std::fprintf(stderr, "resize input %zu failed: %s\n", index[k], r.Error().Message().c_str());
                        return 1;
                    }
                    resized = true;
                }
            }
            if (!have_buffers || resized) {
                // Dynamic output shapes (e.g. YAMNet's frame count) follow resized inputs only once the
                // allocation is refreshed; otherwise the output buffers keep the pre-resize size.
                if (!model.GetOutputTensorLayouts(si, /*update_allocation=*/true)) {
                    std::fprintf(stderr, "cannot update output layouts\n");
                    return 1;
                }
                auto inputs = model.CreateInputBuffers(si);
                auto outputs = model.CreateOutputBuffers(si);
                if (!inputs || !outputs) {
                    std::fprintf(stderr, "cannot allocate buffers\n");
                    return 1;
                }
                for (auto& [oi, dims] : out_shapes) {
                    if (oi >= outputs->size()) return 2;
                    auto ot = (*outputs)[oi].TensorType();
                    auto et = ot->ElementType();
                    size_t n = 1;
                    for (auto d : dims) n *= static_cast<size_t>(d);
                    size_t esz = et == litert::ElementType::Int64 ? 8 : (et == litert::ElementType::Float32 || et == litert::ElementType::Int32) ? 4 : 1;
                    litert::Dimensions dd(dims.begin(), dims.end());
                    auto b = litert::TensorBuffer::CreateManagedHostMemory(
                        litert::RankedTensorType(et, litert::Layout(dd)), n * esz);
                    if (!b) {
                        std::fprintf(stderr, "cannot create output %zu buffer: %s\n", oi, b.Error().Message().c_str());
                        return 1;
                    }
                    (*outputs)[oi] = std::move(*b);
                }
                inputs_v = std::move(*inputs);
                outputs_v = std::move(*outputs);
                have_buffers = true;
            }
            auto* inputs = &inputs_v;
            auto* outputs = &outputs_v;
            for (size_t k = 0; k < specs.size(); ++k) {
                auto& buf = (*inputs)[index[k]];
                auto tt = buf.TensorType();
                if (std::string(et_name(tt->ElementType())) != specs[k].dtype) {
                    std::fprintf(stderr, "input %s is %s, given %s\n", specs[k].name.c_str(),
                                 et_name(tt->ElementType()), specs[k].dtype.c_str());
                    return 1;
                }
                std::ifstream in(specs[k].file, std::ios::binary);
                std::vector<uint8_t> raw((std::istreambuf_iterator<char>(in)), {});
                auto w = buf.Write<uint8_t>(litert::Span<const uint8_t>(raw.data(), raw.size()));
                if (!w) {
                    std::fprintf(stderr, "write %s: %s\n", specs[k].file.c_str(), w.Error().Message().c_str());
                    return 1;
                }
            }
            if (auto r = model.Run(si, *inputs, *outputs); !r) {
                // A data-dependent output shape is only known after the interpreter has propagated
                // shapes once; refresh the output allocation and run again before giving up.
                std::fprintf(stderr, "tflite-run: first run failed (%s); refreshing output layouts\n",
                             r.Error().Message().c_str());
                auto lay = model.GetOutputTensorLayouts(si, /*update_allocation=*/true);
                if (lay) {
                    for (size_t i = 0; i < lay->size(); ++i) {
                        std::string dd;
                        for (auto d : (*lay)[i].Dimensions()) dd += (dd.empty() ? "" : ",") + std::to_string(d);
                        std::fprintf(stderr, "tflite-run: output %zu layout [%s]\n", i, dd.c_str());
                    }
                }
                auto fresh = model.CreateOutputBuffers(si);
                if (!fresh) return 1;
                outputs_v = std::move(*fresh);
                if (auto r2 = model.Run(si, *inputs, *outputs); !r2) {
                    std::fprintf(stderr, "run failed: %s\n", r2.Error().Message().c_str());
                    return 1;
                }
            }
            for (size_t i = 0; i < outputs->size(); ++i) {
                auto& buf = (*outputs)[i];
                auto tt = buf.TensorType();
                auto sz = buf.PackedSize();
                std::vector<uint8_t> raw(*sz);
                if (!buf.Read<uint8_t>(litert::Span<uint8_t>(raw.data(), raw.size()))) return 1;
                auto f = to_float(raw, tt->ElementType());
                size_t best = 0;
                for (size_t j = 1; j < f.size(); ++j)
                    if (f[j] > f[best]) best = j;
                std::string dims;
                for (auto d : tt->Layout().Dimensions()) dims += (dims.empty() ? "" : ",") + std::to_string(d);
                std::printf("%s out[%zu] %.*s %s [%s] n=%zu argmax=%zu max=%.6g\n", prefix.c_str(), i,
                            static_cast<int>((*out_names)[i].size()), (*out_names)[i].data(),
                            et_name(tt->ElementType()), dims.c_str(), f.size(), best, f.empty() ? 0.0 : f[best]);
                std::ofstream(prefix + "." + std::to_string(i) + ".f32", std::ios::binary)
                    .write(reinterpret_cast<const char*>(f.data()), f.size() * sizeof(float));
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "tflite-run: %s\n", e.what());
        return 1;
    }
    return 0;
}

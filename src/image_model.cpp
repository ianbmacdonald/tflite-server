#include "image_model.h"

#include "counting_semaphore.h"
#include "errors.h"
#include "image_manifest.h"
#include "litert_engine.h"

#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_tensor_buffer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr auto kSlotWait = std::chrono::seconds(30);

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

struct ImageModel::Impl {
    ImageManifest manifest;
    std::vector<std::string> labels;
    ImageServeOptions options;
    CompiledLiteRt lm;
    std::vector<litert::TensorBuffer> inputs;
    std::vector<litert::TensorBuffer> outputs;
    size_t signature = 0;
    int in_h = 0;
    int in_w = 0;
    CountingSemaphore decode_slots;
    std::mutex run_mutex;

    Impl(const fs::path& dir, int threads, const std::string& weight_cache,
         const ImageServeOptions& opts, bool verbose)
        : manifest(load_image_manifest(dir)),
          labels(load_labels(dir / manifest.labels_file)),
          options(opts),
          decode_slots(opts.max_concurrent_decodes) {
        threads = resolve_threads(threads);
        lm = make_compiled_model(dir / "model.tflite", threads, weight_cache);
        // A model exported without SignatureDefs (the TF MobileNet .tflite
        // files) still gets LiteRT's default signature at index 0.
        if (lm.model->GetNumSignatures() < 1) throw std::runtime_error("model.tflite has no signatures");
        auto in = lm.model->CreateInputBuffers(signature);
        auto out = lm.model->CreateOutputBuffers(signature);
        if (!in || !out) throw std::runtime_error("cannot allocate LiteRT input/output buffers");
        inputs = std::move(*in);
        outputs = std::move(*out);
        if (inputs.size() != 1) {
            throw std::runtime_error("image models must have exactly one input; this one has " +
                                     std::to_string(inputs.size()));
        }
        if (outputs.size() != 1) {
            throw std::runtime_error("image models must have exactly one output; this one has " +
                                     std::to_string(outputs.size()));
        }

        auto it = inputs[0].TensorType();
        if (!it) throw std::runtime_error("cannot read the model's input tensor type");
        const auto et = it->ElementType();
        if (et == litert::ElementType::Int8 || et == litert::ElementType::UInt8 ||
            et == litert::ElementType::Int16) {
            throw std::runtime_error("quantized-input models not supported yet");
        }
        if (et != litert::ElementType::Float32) throw std::runtime_error("model input must be float32");
        auto dims = it->Layout().Dimensions();
        if (dims.size() == 4 && dims[0] == 1 && dims[1] == 3 && dims[3] != 3) {
            throw std::runtime_error("NCHW input not supported in this build");
        }
        if (dims.size() != 4 || dims[0] != 1 || dims[1] < 1 || dims[2] < 1 || dims[3] != 3) {
            throw std::runtime_error("model input must be [1, height, width, 3]");
        }
        in_h = static_cast<int>(dims[1]);
        in_w = static_cast<int>(dims[2]);

        auto ot = outputs[0].TensorType();
        if (!ot) throw std::runtime_error("cannot read the model's output tensor type");
        if (ot->ElementType() != litert::ElementType::Float32) {
            throw std::runtime_error("model output must be float32");
        }
        size_t n = 1;
        for (auto d : ot->Layout().Dimensions()) n *= static_cast<size_t>(std::max<int64_t>(d, 0));
        if (n != labels.size()) {
            throw std::runtime_error("model output has " + std::to_string(n) + " scores but " +
                                     manifest.labels_file + " has " + std::to_string(labels.size()) +
                                     " labels");
        }
        if (verbose) {
            std::fprintf(stderr,
                         "tflite-server: image-classification, input %dx%d, %zu labels, %d thread(s), "
                         "%d decode slot(s)\n",
                         in_w, in_h, labels.size(), threads, opts.max_concurrent_decodes);
        }
    }

    json classify(std::string_view bytes, int top_k) {
        if (bytes.size() > options.max_image_bytes) {
            throw PayloadTooLarge("image is " + std::to_string(bytes.size()) + " bytes; the limit is " +
                                  std::to_string(options.max_image_bytes));
        }
        if (bytes.empty()) throw InvalidInput("image is empty");
        if (imgproc::sniff_image_format(bytes) == imgproc::ImageFormat::Unknown) {
            throw InvalidInput("unsupported image format (JPEG or PNG)");
        }

        double decode_ms = 0, preprocess_ms = 0, inference_ms = 0;
        int src_w = 0, src_h = 0;
        std::vector<float> tensor;
        {
            if (!decode_slots.acquire_for(kSlotWait)) {
                throw Busy("busy: no image decode slot became free within 30 s");
            }
            SlotGuard slot(decode_slots);
            auto t0 = std::chrono::steady_clock::now();
            imgproc::DecodeResult d = imgproc::decode_rgb8(bytes, options.limits);
            decode_ms = ms_since(t0);
            if (d.error != imgproc::DecodeError::None) throw InvalidInput(d.message);
            src_w = d.image.width;
            src_h = d.image.height;
            t0 = std::chrono::steady_clock::now();
            tensor = imgproc::resample_triangle(d.image.pixels.get(), src_w, src_h, 3, in_w, in_h);
            d.image.pixels.reset();
            imgproc::normalize_in_place(tensor, manifest.mean, manifest.std);
            preprocess_ms = ms_since(t0);
        }

        std::vector<float> scores(labels.size());
        {
            std::lock_guard<std::mutex> lock(run_mutex);
            const auto t0 = std::chrono::steady_clock::now();
            if (!inputs[0].Write<float>(litert::Span<const float>(tensor.data(), tensor.size()))) {
                throw std::runtime_error("cannot write the model input");
            }
            std::vector<float>().swap(tensor);
            if (!lm.model->Run(signature, inputs, outputs)) throw std::runtime_error("LiteRT inference failed");
            if (!outputs[0].Read<float>(litert::Span<float>(scores.data(), scores.size()))) {
                throw std::runtime_error("cannot read the model output");
            }
            inference_ms = ms_since(t0);
        }

        // score_normalization "none": the model must already emit probabilities.
        for (float s : scores) {
            if (!(s >= -1e-6f && s <= 1.0f + 1e-6f)) {
                throw std::runtime_error("model output is not probabilities");
            }
        }

        const size_t n = labels.size();
        size_t k = static_cast<size_t>(top_k > 0 ? top_k : manifest.top_k_default);
        k = std::min(k, n);
        std::vector<size_t> order(n);
        std::iota(order.begin(), order.end(), size_t{0});
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k), order.end(),
                          [&](size_t a, size_t b) {
                              return scores[a] != scores[b] ? scores[a] > scores[b] : a < b;
                          });

        json predictions = json::array();
        json by_label = json::object();
        for (size_t i = 0; i < k; ++i) {
            const size_t idx = order[i];
            predictions.push_back({{"index", idx}, {"label", labels[idx]}, {"score", scores[idx]}});
            // ImageNet repeats some names ("crane", "maillot"); keep the higher score.
            if (!by_label.contains(labels[idx])) by_label[labels[idx]] = scores[idx];
        }
        return json{{"predictions", predictions},
                    {"labels", by_label},
                    {"input", {{"width", src_w}, {"height", src_h}}},
                    {"timings",
                     {{"decode_ms", decode_ms}, {"preprocess_ms", preprocess_ms}, {"inference_ms", inference_ms}}}};
    }
};

ImageModel::ImageModel(const fs::path& dir, int threads, const std::string& weight_cache,
                       const ImageServeOptions& options, bool verbose)
    : impl_(std::make_unique<Impl>(dir, threads, weight_cache, options, verbose)) {}

ImageModel::~ImageModel() = default;

json ImageModel::classify(std::string_view image_bytes, int top_k) {
    return impl_->classify(image_bytes, top_k);
}

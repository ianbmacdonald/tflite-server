// tflite-server: LiteRT (TFLite) model server for Lemonade (see README).
//
// One process serves one model directory. A manifest.json with
// "task": "image-classification" selects the image path (/classify/image);
// anything else is the text path (/classify), derived from
// lemonade-sdk/ort-server (see text_model.cpp and NOTICE).

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "errors.h"
#include "image_model.h"
#include "image_preprocess.h"
#include "text_model.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr const char* kVersion = "0.2.0";
constexpr long long kMaxTopK = 1000000;

struct Args {
    std::string model_path;
    int port = 0;
    int threads = 0;  // 0: one per hardware thread
    std::string weight_cache;  // XNNPACK packed-weight cache file; empty = in memory
    bool verbose = false;
    uint64_t max_image_bytes = 16u << 20;
    uint64_t max_image_pixels = 4000000;
    uint64_t decode_budget_factor = 16;
    int max_concurrent_decodes = 1;
    int http_threads = 4;
    int oom_score_adj = 0;  // 0: leave unchanged
};

const char* kUsage =
    "usage: tflite-server --model-path <dir> --port <n> [--threads N] [--weight-cache FILE] [--verbose]\n"
    "  image models: [--max-image-bytes N] [--max-image-pixels N] [--max-concurrent-decodes 1..2]\n"
    "                [--decode-budget-factor N]\n"
    "  all models:   [--http-threads 2..16] [--oom-score-adj 0..1000]";

long long parse_int(const std::string& flag, const char* v) {
    size_t pos = 0;
    long long n = 0;
    try {
        n = std::stoll(v, &pos);
    } catch (const std::exception&) {
        pos = 0;
    }
    if (pos == 0 || v[pos] != '\0') throw std::runtime_error(flag + " needs an integer\n" + kUsage);
    return n;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string f = argv[i];
        const bool has_value = i + 1 < argc;
        if (f == "--model-path" && has_value) a.model_path = argv[++i];
        else if (f == "--port" && has_value) a.port = static_cast<int>(parse_int(f, argv[++i]));
        else if (f == "--threads" && has_value) a.threads = static_cast<int>(parse_int(f, argv[++i]));
        else if (f == "--weight-cache" && has_value) a.weight_cache = argv[++i];
        else if (f == "--verbose") a.verbose = true;
        else if (f == "--max-image-bytes" && has_value) a.max_image_bytes = parse_int(f, argv[++i]);
        else if (f == "--max-image-pixels" && has_value) a.max_image_pixels = parse_int(f, argv[++i]);
        else if (f == "--decode-budget-factor" && has_value) a.decode_budget_factor = parse_int(f, argv[++i]);
        else if (f == "--max-concurrent-decodes" && has_value) {
            a.max_concurrent_decodes = static_cast<int>(parse_int(f, argv[++i]));
        } else if (f == "--http-threads" && has_value) {
            a.http_threads = static_cast<int>(parse_int(f, argv[++i]));
        } else if (f == "--oom-score-adj" && has_value) {
            a.oom_score_adj = static_cast<int>(parse_int(f, argv[++i]));
        }
    }
    if (a.model_path.empty() || a.port == 0) throw std::runtime_error(kUsage);
    if (a.max_image_bytes < 1024 || a.max_image_bytes > (256ull << 20)) {
        throw std::runtime_error("--max-image-bytes must be from 1024 to 268435456");
    }
    if (a.max_image_pixels < 1 || a.max_image_pixels > 16384ull * 16384ull) {
        throw std::runtime_error("--max-image-pixels must be from 1 to 268435456");
    }
    if (a.decode_budget_factor < 4 || a.decode_budget_factor > 64) {
        throw std::runtime_error("--decode-budget-factor must be from 4 to 64");
    }
    if (a.max_concurrent_decodes < 1 || a.max_concurrent_decodes > 2) {
        const int clamped = std::clamp(a.max_concurrent_decodes, 1, 2);
        std::fprintf(stderr, "tflite-server: --max-concurrent-decodes %d clamped to %d\n",
                     a.max_concurrent_decodes, clamped);
        a.max_concurrent_decodes = clamped;
    }
    if (a.http_threads < 2 || a.http_threads > 16) {
        throw std::runtime_error("--http-threads must be from 2 to 16");
    }
    if (a.oom_score_adj < 0 || a.oom_score_adj > 1000) {
        throw std::runtime_error("--oom-score-adj must be from 0 to 1000");
    }
    return a;
}

// Raising the score needs no privilege. It makes this process the OOM killer's
// first choice over lemond or an LLM backend sharing the host.
void apply_oom_score_adj(int value) {
    if (value == 0) return;
#ifdef __linux__
    std::ofstream f("/proc/self/oom_score_adj");
    f << value;
    f.flush();
    if (!f) throw std::runtime_error("cannot write /proc/self/oom_score_adj");
#else
    std::fprintf(stderr, "tflite-server: --oom-score-adj is only supported on Linux; ignored\n");
#endif
}

std::string manifest_task(const fs::path& dir) {
    std::ifstream f(dir / "manifest.json");
    if (!f) return "";
    try {
        json j;
        f >> j;
        if (j.is_object() && j.contains("task") && j["task"].is_string()) return j["task"].get<std::string>();
    } catch (const std::exception&) {
    }
    return "";
}

void send_error(httplib::Response& res, int status, const std::string& message) {
    res.status = status;
    res.set_content(json{{"error", message}}.dump(), "application/json");
}

// top_k follows Lemonade's /v1/classify rule: an integer from 1 to 1,000,000.
int top_k_from_json(const json& v) {
    if (!v.is_number_integer() || v.get<long long>() < 1 || v.get<long long>() > kMaxTopK) {
        throw InvalidInput("top_k must be an integer from 1 to 1000000");
    }
    return static_cast<int>(v.get<long long>());
}

int top_k_from_field(const std::string& s) {
    if (s.empty() || s.size() > 7 || !std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; })) {
        throw InvalidInput("top_k must be an integer from 1 to 1000000");
    }
    return top_k_from_json(json(std::stoll(s)));
}

// Multipart: exactly one file part named "image" or "file", plus an optional
// "top_k" field. JSON: {"image": "<base64 or data:image/(jpeg|png);base64,...>", "top_k": k}.
void parse_image_request(const httplib::Request& req, std::string& bytes, int& top_k) {
    top_k = 0;
    if (req.is_multipart_form_data()) {
        const httplib::MultipartFormData* part = nullptr;
        size_t parts = 0;
        for (const auto& [name, f] : req.files) {
            if (name == "image" || name == "file") {
                ++parts;
                part = &f;
            } else if (name == "top_k") {
                top_k = top_k_from_field(f.content);
            }
        }
        if (parts != 1) throw InvalidInput("exactly one image part ('image' or 'file') required");
        bytes = part->content;
        return;
    }
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception&) {
        throw InvalidInput("request body must be multipart/form-data or JSON");
    }
    if (!body.is_object() || !body.contains("image") || !body["image"].is_string()) {
        throw InvalidInput("'image' (a base64 string) is required");
    }
    if (body.contains("top_k")) top_k = top_k_from_json(body["top_k"]);
    std::string_view s = body["image"].get_ref<const std::string&>();
    if (s.rfind("http://", 0) == 0 || s.rfind("https://", 0) == 0) {
        throw InvalidInput("remote image URLs are not supported; send base64 or a data: URL");
    }
    if (s.rfind("data:", 0) == 0) {
        bool ok = false;
        for (std::string_view prefix : {"data:image/jpeg;base64,", "data:image/png;base64,"}) {
            if (s.rfind(prefix, 0) == 0) {
                s.remove_prefix(prefix.size());
                ok = true;
                break;
            }
        }
        if (!ok) throw InvalidInput("data URLs must be data:image/jpeg;base64 or data:image/png;base64");
    }
    if (!imgproc::strict_base64_decode(s, bytes)) throw InvalidInput("'image' is not valid base64");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        apply_oom_score_adj(args.oom_score_adj);

        std::unique_ptr<TextModel> text;
        std::unique_ptr<ImageModel> image;
        std::string task;
        if (manifest_task(args.model_path) == "image-classification") {
            ImageServeOptions opts;
            opts.max_image_bytes = args.max_image_bytes;
            opts.limits.max_pixels = args.max_image_pixels;
            opts.limits.budget_factor = args.decode_budget_factor;
            opts.max_concurrent_decodes = args.max_concurrent_decodes;
            image = std::make_unique<ImageModel>(args.model_path, args.threads, args.weight_cache, opts,
                                                 args.verbose);
            task = "image-classification";
        } else {
            text = std::make_unique<TextModel>(args.model_path, args.threads, args.weight_cache,
                                               args.verbose);
            task = text->task();
        }

        httplib::Server srv;
        const size_t http_threads = static_cast<size_t>(args.http_threads);
        srv.new_task_queue = [http_threads] { return new httplib::ThreadPool(http_threads); };
        // Base64 inflates by 4/3; 64 KiB covers multipart headers and the JSON wrapper.
        srv.set_payload_max_length(static_cast<size_t>(args.max_image_bytes * 4 / 3 + (64u << 10)));

        srv.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
            res.set_content(json{{"status", "ok"}, {"engine", "litert"}, {"task", task}, {"version", kVersion}}.dump(),
                            "application/json");
        });
        srv.Post("/classify", [&](const httplib::Request& req, httplib::Response& res) {
            if (!text) {
                send_error(res, 400, "this server hosts an image-classification model; POST /classify/image");
                return;
            }
            std::string input;
            int top_k = 0;
            try {
                json body = json::parse(req.body);
                input = body.contains("text") ? body.at("text").get<std::string>()
                                              : body.at("input").get<std::string>();
                top_k = body.value("top_k", 0);
            } catch (const std::exception& e) {
                send_error(res, 400, e.what());
                return;
            }
            try {
                res.set_content(text->classify(input, top_k).dump(), "application/json");
            } catch (const InvalidInput& e) {
                send_error(res, 400, e.what());
            } catch (const std::exception& e) {
                send_error(res, 500, e.what());
            }
        });
        srv.Post("/classify/image", [&](const httplib::Request& req, httplib::Response& res) {
            if (!image) {
                send_error(res, 400, "this server hosts a text model; POST /classify");
                return;
            }
            try {
                std::string bytes;
                int top_k = 0;
                parse_image_request(req, bytes, top_k);
                res.set_content(image->classify(bytes, top_k).dump(), "application/json");
            } catch (const InvalidInput& e) {
                send_error(res, 400, e.what());
            } catch (const PayloadTooLarge& e) {
                send_error(res, 413, e.what());
            } catch (const Busy& e) {
                send_error(res, 503, e.what());
            } catch (const std::exception& e) {
                send_error(res, 500, e.what());
            }
        });

        if (!srv.listen("127.0.0.1", args.port)) {
            std::fprintf(stderr, "tflite-server: failed to bind 127.0.0.1:%d\n", args.port);
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "tflite-server: %s\n", e.what());
        return 1;
    }
}

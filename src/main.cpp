// tflite-server: LiteRT (TFLite) model server for Lemonade (see README).
//
// One process serves one model directory. A manifest.json with
// "task": "image-classification" selects the image path (/classify/image);
// anything else is the text path (/classify), derived from
// lemonade-sdk/ort-server (see text_model.cpp and NOTICE).

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "counting_semaphore.h"
#include "errors.h"
#include "flat_json.h"
#include "image_model.h"
#include "image_preprocess.h"
#include "text_model.h"

#include <algorithm>
#include <chrono>
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

constexpr const char* kVersion = TFLITE_SERVER_VERSION;
constexpr long long kMaxTopK = 1000000;
constexpr auto kAdmissionWait = std::chrono::seconds(30);
constexpr auto kSocketTimeout = std::chrono::seconds(5);

struct Args {
    std::string model_path;
    int port = 0;
    int threads = 0;  // 0: one per hardware thread
    std::string weight_cache;  // XNNPACK packed-weight cache file; empty = in memory
    bool verbose = false;
    uint64_t max_image_bytes = 16u << 20;
    uint64_t max_image_pixels = 4000000;
    uint64_t decode_budget_factor = 16;
    uint64_t max_decode_bytes = 256u << 20;
    int max_concurrent_decodes = 1;
    int http_threads = 4;
    int oom_score_adj = 0;  // 0: leave unchanged
};

const char* kUsage =
    "usage: tflite-server --model-path <dir> --port <n> [--threads N] [--weight-cache FILE] [--verbose]\n"
    "  image models: [--max-image-bytes N] [--max-image-pixels N] [--max-concurrent-decodes 1..2]\n"
    "                [--decode-budget-factor N] [--max-decode-bytes N]\n"
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
        else if (f == "--max-decode-bytes" && has_value) a.max_decode_bytes = parse_int(f, argv[++i]);
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
    if (a.max_decode_bytes < (16ull << 20) || a.max_decode_bytes > (4ull << 30)) {
        throw std::runtime_error("--max-decode-bytes must be from 16777216 to 4294967296");
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

// "" when the directory has no manifest.json (a text model). A manifest that
// exists but cannot be read is a startup error, not a silent text fallback.
std::string manifest_task(const fs::path& dir) {
    const fs::path file = dir / "manifest.json";
    std::error_code ec;
    if (!fs::exists(file, ec)) return "";
    const auto size = fs::file_size(file, ec);
    if (ec) throw std::runtime_error("cannot read " + file.string());
    if (size > (1u << 20)) throw std::runtime_error(file.string() + " is larger than 1 MiB");
    std::ifstream f(file, std::ios::binary);
    json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        throw std::runtime_error(file.string() + " is not valid JSON: " + e.what());
    }
    if (j.is_object() && j.contains("task") && j["task"].is_string()) return j["task"].get<std::string>();
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

uint64_t content_length(const httplib::Request& req) {
    return req.has_header("Content-Length") ? req.get_header_value_u64("Content-Length") : 0;
}

void drain(const httplib::Request& req, const httplib::ContentReader& reader) {
    if (req.is_multipart_form_data()) {
        reader([](const httplib::MultipartFormData&) { return true; }, [](const char*, size_t) { return true; });
    } else {
        reader([](const char*, size_t) { return true; });
    }
}

// Reads the body through httplib's ContentReader, so nothing is buffered
// before the handler has an admission slot. Receivers never abort: an
// oversize or unwanted part is discarded while the stream is still read to
// its end, which keeps the connection usable.
//
// Multipart: exactly one file part named "image" or "file", plus an optional
// "top_k" field. JSON: {"image": "<base64 or data:image/(jpeg|png);base64,...>", "top_k": k}.
void read_image_request(const httplib::Request& req, const httplib::Response& res,
                        const httplib::ContentReader& reader, uint64_t max_image_bytes, uint64_t max_body_bytes, std::string& bytes, int& top_k) {
    top_k = 0;
    bytes.clear();
    bool too_large = false;
    bool ok = false;
    if (req.is_multipart_form_data()) {
        enum class Part { Image, TopK, Other } cur = Part::Other;
        size_t image_parts = 0;
        std::string top_k_field;
        bool has_top_k = false;
        ok = reader(
            [&](const httplib::MultipartFormData& f) {
                if (f.name == "image" || f.name == "file") {
                    cur = ++image_parts == 1 && !too_large ? Part::Image : Part::Other;
                } else if (f.name == "top_k") {
                    cur = Part::TopK;
                    has_top_k = true;
                    top_k_field.clear();
                } else {
                    cur = Part::Other;
                }
                return true;
            },
            [&](const char* d, size_t n) {
                if (cur == Part::Image) {
                    if (bytes.size() + n > max_image_bytes) {
                        too_large = true;
                        cur = Part::Other;
                        std::string().swap(bytes);
                    } else {
                        bytes.append(d, n);
                    }
                } else if (cur == Part::TopK && top_k_field.size() < 16) {
                    top_k_field.append(d, std::min<size_t>(n, 16));
                }
                return true;
            });
        if (ok && too_large) {
            throw PayloadTooLarge("image is larger than the " + std::to_string(max_image_bytes) + " byte limit");
        }
        if (ok) {
            if (image_parts != 1) throw InvalidInput("exactly one image part ('image' or 'file') required");
            if (has_top_k) top_k = top_k_from_field(top_k_field);
            return;
        }
    } else {
        std::string body;
        const uint64_t declared = content_length(req);
        if (declared <= max_body_bytes) body.reserve(static_cast<size_t>(declared));
        ok = reader([&](const char* d, size_t n) {
            if (too_large) return true;
            if (body.size() + n > max_body_bytes) {
                too_large = true;
                std::string().swap(body);
            } else {
                body.append(d, n);
            }
            return true;
        });
        if (ok && too_large) {
            throw PayloadTooLarge("request body is over the " + std::to_string(max_body_bytes) + " byte limit");
        }
        if (ok) {
            json obj;
            std::string error;
            if (!parse_flat_json_object(body, obj, error)) {
                if (error == "request body is not valid JSON") error += " (send multipart/form-data or a JSON object)";
                throw InvalidInput(error);
            }
            std::string().swap(body);
            if (!obj.contains("image") || !obj["image"].is_string()) {
                throw InvalidInput("'image' (a base64 string) is required");
            }
            if (obj.contains("top_k")) top_k = top_k_from_json(obj["top_k"]);
            std::string image = std::move(obj["image"].get_ref<std::string&>());
            obj = json();
            std::string_view s = image;
            if (s.rfind("http://", 0) == 0 || s.rfind("https://", 0) == 0) {
                throw InvalidInput("remote image URLs are not supported; send base64 or a data: URL");
            }
            if (s.rfind("data:", 0) == 0) {
                bool prefix_ok = false;
                for (std::string_view prefix : {"data:image/jpeg;base64,", "data:image/png;base64,"}) {
                    if (s.rfind(prefix, 0) == 0) {
                        s.remove_prefix(prefix.size());
                        prefix_ok = true;
                        break;
                    }
                }
                if (!prefix_ok) throw InvalidInput("data URLs must be data:image/jpeg;base64 or data:image/png;base64");
            }
            if (!imgproc::strict_base64_decode(s, bytes)) throw InvalidInput("'image' is not valid base64");
            return;
        }
    }
    std::string().swap(bytes);
    // A failed read has set res.status: 413 for a Content-Length over the
    // payload limit, 400 for a malformed or truncated body.
    if (res.status == 413) {
        throw PayloadTooLarge("request body is over the " + std::to_string(max_body_bytes) + " byte limit");
    }
    throw InvalidInput("cannot read the request body (malformed multipart or truncated)");
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
            opts.limits.max_budget = args.max_decode_bytes;
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
        // Per-recv idle limits, not a cap on a request's total time.
        srv.set_read_timeout(kSocketTimeout);
        srv.set_write_timeout(kSocketTimeout);
        const size_t http_threads = static_cast<size_t>(args.http_threads);
        srv.new_task_queue = [http_threads] { return new httplib::ThreadPool(http_threads); };
        // Base64 inflates by 4/3; 64 KiB covers multipart headers and the JSON wrapper.
        const uint64_t max_body_bytes = args.max_image_bytes * 4 / 3 + (64u << 10);
        srv.set_payload_max_length(static_cast<size_t>(max_body_bytes));
        // Image requests read their body only after taking one of these, so
        // http_threads bounds waiting connections, not buffered bodies. One
        // more than the decode slots lets the next body arrive during a decode.
        CountingSemaphore admission(args.max_concurrent_decodes + 1);
        const uint64_t decode_budget = std::min<uint64_t>(
            args.decode_budget_factor * args.max_image_pixels + (4u << 20) + 2 * args.max_image_bytes,
            args.max_decode_bytes);
        if (image && args.verbose) {
            const uint64_t per_request = max_body_bytes + args.max_image_bytes * 2;
            std::fprintf(stderr,
                         "tflite-server: request memory bound ~%llu MiB (%d admitted x %llu MiB body and copies) "
                         "+ %d decode(s) x <= %llu MiB budget\n",
                         (unsigned long long)(per_request * (args.max_concurrent_decodes + 1) >> 20),
                         args.max_concurrent_decodes + 1, (unsigned long long)(per_request >> 20),
                         args.max_concurrent_decodes, (unsigned long long)(decode_budget >> 20));
        }
        if (image && args.decode_budget_factor * args.max_image_pixels + (4u << 20) > args.max_decode_bytes) {
            std::fprintf(stderr,
                         "tflite-server: warning: --decode-budget-factor x --max-image-pixels exceeds "
                         "--max-decode-bytes; images near the pixel cap may fail the decode budget\n");
        }

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
                json body;
                std::string error;
                if (!parse_flat_json_object(req.body, body, error)) throw InvalidInput(error);
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
        srv.Post("/classify/image", [&](const httplib::Request& req, httplib::Response& res,
                                        const httplib::ContentReader& reader) {
            if (!image) {
                drain(req, reader);
                send_error(res, 400, "this server hosts a text model; POST /classify");
                return;
            }
            if (content_length(req) > max_body_bytes) {
                drain(req, reader);
                send_error(res, 413, "request body is over the " + std::to_string(max_body_bytes) + " byte limit");
                return;
            }
            if (!admission.acquire_for(kAdmissionWait)) {
                drain(req, reader);
                send_error(res, 503, "busy: no request slot became free within 30 s");
                return;
            }
            SlotGuard slot(admission);
            try {
                std::string bytes;
                int top_k = 0;
                read_image_request(req, res, reader, args.max_image_bytes, max_body_bytes, bytes, top_k);
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

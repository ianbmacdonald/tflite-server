// Standalone tests for src/image_manifest.cpp: accepted values load, reserved
// ones are rejected by name, and unknown keys or values never pass silently.

#include "image_manifest.h"

#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    }
}

std::string error_of(const json& j) {
    try {
        parse_image_manifest(j);
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

void expect_error(const char* text, const std::string& want) {
    const std::string got = error_of(json::parse(text));
    check(got == want, std::string(text) + "\n    want: " + want + "\n    got:  " + got);
}

void test_accepts() {
    auto m = parse_image_manifest(json::parse(R"({
        "task": "image-classification", "labels_file": "labels.txt", "score_normalization": "none",
        "preprocess": {"resize": "stretch", "mean": [127.5, 127.5, 127.5], "std": [127.5, 127.5, 127.5],
                       "channel_order": "RGB", "layout": "auto"},
        "top_k_default": 5})"));
    check(m.labels_file == "labels.txt" && m.top_k_default == 5 && m.mean[1] == 127.5f &&
              m.std[2] == 127.5f && m.layout == TensorLayout::NHWC,
          "full MobileNet manifest");
    auto c = parse_image_manifest(json::parse(
        R"({"task": "image-classification", "preprocess": {"layout": "NCHW"}})"));
    check(c.layout == TensorLayout::NCHW, "layout NCHW");
    auto s = parse_image_manifest(json::parse(
        R"({"task": "image-classification", "preprocess": {"mean": 0, "std": 255, "layout": "NHWC"}})"));
    check(s.mean[0] == 0.0f && s.std[2] == 255.0f && s.labels_file == "labels.txt" && s.top_k_default == 5,
          "scalar mean/std and defaults");
    auto v = parse_image_manifest(json::parse(
        R"({"task": "image-classification", "preprocess": {"mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375]}})"));
    check(v.mean[2] == 103.53f && v.std[0] == 58.395f, "per-channel mean/std");
}

void test_rejects() {
    expect_error(R"({"task": "image-classification", "preprocess": {"resize": "center_crop"}})",
                 "manifest.preprocess.resize 'center_crop' not supported in this build");
    expect_error(R"({"task": "image-classification", "preprocess": {"resize": "resize_shorter"}})",
                 "manifest.preprocess.resize 'resize_shorter' not supported in this build");
    expect_error(R"({"task": "image-classification", "preprocess": {"layout": "NHCW"}})",
                 "manifest.preprocess.layout 'NHCW' is not a known value");
    expect_error(R"({"task": "image-classification", "preprocess": {"channel_order": "BGR"}})",
                 "manifest.preprocess.channel_order 'BGR' not supported in this build");
    expect_error(R"({"task": "image-classification", "score_normalization": "softmax"})",
                 "manifest.score_normalization 'softmax' not supported in this build");
    expect_error(R"({"task": "image-classification", "score_normalization": "sigmoid"})",
                 "manifest.score_normalization 'sigmoid' not supported in this build");
    expect_error(R"({"task": "image-classification", "preprocess": {"resize": "squash"}})",
                 "manifest.preprocess.resize 'squash' is not a known value");
    expect_error(R"({"task": "image-classification", "ignore_indices": [0]})",
                 "manifest.ignore_indices: unknown key");
    expect_error(R"({"task": "image-classification", "preprocess": {"scale": 1}})",
                 "manifest.preprocess.scale: unknown key");
    expect_error(R"({"task": "text-classification"})", "manifest.task must be \"image-classification\"");
    expect_error(R"({"labels_file": "labels.txt"})", "manifest.task must be \"image-classification\"");
    expect_error(R"({"task": "image-classification", "labels_file": "../etc/passwd"})",
                 "manifest.labels_file must be a file name inside the model directory");
    expect_error(R"({"task": "image-classification", "labels_file": "/etc/passwd"})",
                 "manifest.labels_file must be a file name inside the model directory");
    expect_error(R"({"task": "image-classification", "preprocess": {"std": [1, 0, 1]}})",
                 "manifest.preprocess.std values must be finite and > 0");
    expect_error(R"({"task": "image-classification", "preprocess": {"mean": [1, 2]}})",
                 "manifest.preprocess.mean must be a number or an array of 3 numbers");
    expect_error(R"({"task": "image-classification", "top_k_default": 0})",
                 "manifest.top_k_default must be an integer from 1 to 1000000");
    expect_error(R"({"task": "image-classification", "preprocess": {"resize": 1}})",
                 "manifest.preprocess.resize must be a string");
    expect_error(R"([1, 2])", "manifest.json: must be an object");
}

void test_files() {
    const fs::path dir = fs::temp_directory_path() / ("tflite-manifest-test-" + std::to_string(::getpid()));
    fs::create_directories(dir);
    {
        std::ofstream(dir / "labels.txt", std::ios::binary) << "background\r\ntench\r\ncrane\r\ncrane\r\n";
    }
    auto labels = load_labels(dir / "labels.txt");
    check(labels.size() == 4 && labels[1] == "tench" && labels[3] == "crane", "CRLF labels with duplicates");
    {
        std::ofstream(dir / "manifest.json") << std::string(2u << 20, ' ') << "{}";
    }
    bool threw = false;
    try {
        load_image_manifest(dir);
    } catch (const std::exception& e) {
        threw = std::string(e.what()).find("larger than 1 MiB") != std::string::npos;
    }
    check(threw, "a 2 MiB manifest is refused before parsing");
    {
        std::ofstream(dir / "manifest.json") << "{\"task\": \"image-classif";
    }
    threw = false;
    try {
        load_image_manifest(dir);
    } catch (const std::exception& e) {
        threw = std::string(e.what()).find("is not valid JSON") != std::string::npos;
    }
    check(threw, "a truncated manifest is a clean error");
    { std::ofstream(dir / "empty.txt"); }
    threw = false;
    try {
        load_labels(dir / "empty.txt");
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "an empty labels file is refused");
    fs::remove_all(dir);
}

}  // namespace

int main() {
    test_accepts();
    test_rejects();
    test_files();
    std::printf("test_image_manifest: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

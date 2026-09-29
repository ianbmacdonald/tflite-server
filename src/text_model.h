#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <memory>
#include <string>

// Text or token classification over a tokenizer + fixed-length signatures
// model directory (see README, "Text model directory").
class TextModel {
public:
    TextModel(const std::filesystem::path& dir, int threads, const std::string& weight_cache,
              bool verbose);
    ~TextModel();
    TextModel(const TextModel&) = delete;
    TextModel& operator=(const TextModel&) = delete;

    nlohmann::json classify(const std::string& text, int top_k);
    // "text-classification" or "token-classification".
    const std::string& task() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

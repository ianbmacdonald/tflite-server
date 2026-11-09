// Standalone tests for src/model_variant.h: which file each accelerator
// compiles when a directory holds an ahead-of-time NPU payload beside the
// portable model. Usage: test_model_variant

#include "model_variant.h"

#include <cstdio>
#include <filesystem>
#include <fstream>

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond, ...)                                                         \
    do {                                                                         \
        ++g_checks;                                                              \
        if (!(cond)) {                                                           \
            ++g_failures;                                                        \
            std::fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); \
            std::fprintf(stderr, __VA_ARGS__);                                   \
            std::fprintf(stderr, "\n");                                          \
        }                                                                        \
    } while (0)

namespace fs = std::filesystem;

void write_file(const fs::path& p) { std::ofstream(p) << "x"; }

// A fresh directory per case, so one case cannot see another's files.
fs::path make_case_dir(const char* name) {
    fs::path dir = fs::temp_directory_path() / "tflite_model_variant" / name;
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

void test_npu_prefers_the_aot_payload_when_present() {
    fs::path dir = make_case_dir("both");
    write_file(dir / "model.tflite");
    write_file(dir / "model.npu.tflite");

    fs::path chosen = select_model_file(dir / "model.tflite", true);

    CHECK(chosen == dir / "model.npu.tflite", "chose %s", chosen.string().c_str());
}

void test_npu_uses_the_base_model_when_no_payload_exists() {
    fs::path dir = make_case_dir("base_only");
    write_file(dir / "model.tflite");

    fs::path chosen = select_model_file(dir / "model.tflite", true);

    CHECK(chosen == dir / "model.tflite", "chose %s", chosen.string().c_str());
}

void test_cpu_uses_the_base_model_even_beside_a_payload() {
    fs::path dir = make_case_dir("cpu_with_payload");
    write_file(dir / "model.tflite");
    write_file(dir / "model.npu.tflite");

    fs::path chosen = select_model_file(dir / "model.tflite", false);

    CHECK(chosen == dir / "model.tflite", "chose %s", chosen.string().c_str());
}

}  // namespace

int main() {
    test_npu_prefers_the_aot_payload_when_present();
    test_npu_uses_the_base_model_when_no_payload_exists();
    test_cpu_uses_the_base_model_even_beside_a_payload();
    std::fprintf(stderr, "test_model_variant: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

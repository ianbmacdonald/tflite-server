// Standalone tests for src/flat_json.cpp, including the memory cost of hostile
// request bodies. Usage: test_flat_json

#include "flat_json.h"

#include <cstdio>
#include <fstream>
#include <string>

#ifdef __GLIBC__
#include <malloc.h>
#endif

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond, ...)                                                          \
    do {                                                                          \
        ++g_checks;                                                               \
        if (!(cond)) {                                                            \
            ++g_failures;                                                         \
            std::fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); \
            std::fprintf(stderr, __VA_ARGS__);                                    \
            std::fprintf(stderr, "\n");                                           \
        }                                                                         \
    } while (0)

using json = nlohmann::json;

bool parses(const std::string& body, json& out, std::string& error) {
    return parse_flat_json_object(body, out, error);
}

void test_accepts_flat_objects() {
    json out;
    std::string error;
    CHECK(parses(R"({"image":"aGVsbG8=","top_k":3,"f":1.5,"n":null,"b":true,"u":18446744073709551615})", out,
                 error),
          "%s", error.c_str());
    CHECK(out["image"] == "aGVsbG8=" && out["top_k"] == 3 && out["f"] == 1.5 && out["n"].is_null() &&
              out["b"] == true && out["u"].is_number_unsigned(),
          "%s", out.dump().c_str());
    CHECK(parses("  {}  ", out, error) && out.empty(), "empty object: %s", error.c_str());
    CHECK(parses("{\"text\":\"caf\\u00e9\"}", out, error) && out["text"] == "caf\xc3\xa9", "unicode escape");
    CHECK(parses(R"({"s":"a\"b\\c\/d\n\t\ud83d\ude00","x":-1.25e2,"y":false})", out, error) &&
              out["s"] == "a\"b\\c/d\n\t\xf0\x9f\x98\x80" && out["x"] == -125.0 && out["y"] == false,
          "escapes: %s %s", error.c_str(), out.dump().c_str());
    CHECK(parses("{\"k\" : \"v\" ,\n \"n\" : 1 }", out, error) && out["k"] == "v" && out["n"] == 1,
          "whitespace between tokens: %s", error.c_str());
}

void test_rejects(const char* what, const std::string& body, const char* expect) {
    json out;
    std::string error;
    const bool ok = parses(body, out, error);
    CHECK(!ok, "%s must fail", what);
    CHECK(error.find(expect) != std::string::npos, "%s: error '%s' lacks '%s'", what, error.c_str(), expect);
    CHECK(out.is_object() && out.empty(), "%s: output must be reset", what);
}

void test_rejections() {
    test_rejects("nested object", R"({"image":"x","o":{"a":1}})", "flat JSON object");
    test_rejects("array value", R"({"image":"x","a":[1,2]})", "flat JSON object");
    test_rejects("top-level array", "[1,2,3]", "must be a JSON object");
    test_rejects("top-level string", "\"x\"", "must be a JSON object");
    test_rejects("top-level number", "5", "must be a JSON object");
    test_rejects("duplicate key", R"({"image":"a","image":"b"})", "duplicate key 'image'");
    test_rejects("malformed", R"({"image":)", "not valid JSON");
    test_rejects("trailing data", R"({"image":"a"} {)", "not valid JSON");
    test_rejects("empty body", "", "not valid JSON");
    test_rejects("plain text", "hello", "not valid JSON");
    test_rejects("lone low surrogate", R"({"s":"\udc00"})", "not valid JSON");
    test_rejects("lone high surrogate", R"({"s":"\ud800x"})", "not valid JSON");
    test_rejects("bad escape", R"({"s":"\q"})", "not valid JSON");
    test_rejects("raw control character", "{\"s\":\"a\x01b\"}", "not valid JSON");
    test_rejects("invalid UTF-8", "{\"s\":\"\xc3\x28\"}", "not valid JSON");
    test_rejects("overlong UTF-8", "{\"s\":\"\xc0\xaf\"}", "not valid JSON");
    test_rejects("unterminated string", R"({"s":"abc)", "not valid JSON");
    test_rejects("bad literal", R"({"b":tru})", "not valid JSON");
    test_rejects("trailing comma", R"({"a":1,})", "not valid JSON");
    test_rejects("missing colon", R"({"a" 1})", "not valid JSON");
    test_rejects("unquoted key", R"({a:1})", "not valid JSON");
    std::string keys = "{";
    for (int i = 0; i < 33; ++i) keys += "\"k" + std::to_string(i) + "\":0,";
    keys.back() = '}';
    test_rejects("33 keys", keys, "more than 32 keys");
}

#ifdef __linux__
long vm_hwm_kib() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmHWM:", 0) == 0) return std::stol(line.substr(6));
    }
    return -1;
}

// Resets VmHWM to the current RSS (Linux >= 4.0). Free heap pages are
// returned first; otherwise a parse could reuse them without raising RSS.
bool reset_hwm() {
#ifdef __GLIBC__
    malloc_trim(0);
#endif
    std::ofstream f("/proc/self/clear_refs");
    f << "5";
    f.flush();
    return static_cast<bool>(f);
}

// Peak RSS growth, in MiB, of parsing `body` (already resident).
double parse_growth_mib(const std::string& body, bool& ok, std::string& error) {
    reset_hwm();
    const long before = vm_hwm_kib();
    json out;
    ok = parse_flat_json_object(body, out, error);
    const long after = vm_hwm_kib();
    return (after - before) / 1024.0;
}

void test_hostile_bodies_stay_small() {
    if (!reset_hwm() || vm_hwm_kib() < 0) {
        std::printf("  skipped: /proc/self/clear_refs not writable\n");
        return;
    }
    constexpr size_t kHalf = 21u << 19;  // 21 MiB body: the default payload limit
    bool ok = false;
    std::string error;

    const std::string nested = std::string(kHalf, '[') + std::string(kHalf, ']');
    double g = parse_growth_mib(nested, ok, error);
    std::printf("  21 MiB nested arrays: %s, peak growth %.1f MiB\n", error.c_str(), g);
    CHECK(!ok && g < 4.0, "nested arrays: ok=%d growth %.1f MiB", ok, g);

    std::string in_object = "{\"image\":";
    in_object += std::string(kHalf, '[');
    in_object += std::string(kHalf, ']');
    in_object += "}";
    g = parse_growth_mib(in_object, ok, error);
    std::printf("  21 MiB nested arrays inside the object: %s, peak growth %.1f MiB\n", error.c_str(), g);
    CHECK(!ok && g < 4.0, "nested in object: ok=%d growth %.1f MiB", ok, g);

    std::string flat = "[";
    flat.reserve(2 * kHalf + 2);
    for (size_t i = 0; i < kHalf - 1; ++i) flat += "0,";
    flat += "0]";
    g = parse_growth_mib(flat, ok, error);
    std::printf("  21 MiB flat array: %s, peak growth %.1f MiB\n", error.c_str(), g);
    CHECK(!ok && g < 4.0, "flat array: ok=%d growth %.1f MiB", ok, g);

    std::string many_keys = "{";
    for (size_t i = 0; many_keys.size() < 2 * kHalf; ++i) many_keys += "\"k" + std::to_string(i) + "\":0,";
    many_keys += "\"end\":0}";
    g = parse_growth_mib(many_keys, ok, error);
    std::printf("  21 MiB of scalar keys: %s, peak growth %.1f MiB\n", error.c_str(), g);
    CHECK(!ok && g < 4.0, "many keys: ok=%d growth %.1f MiB", ok, g);

    const std::string big = "{\"image\":\"" + std::string(2 * kHalf - 16, 'A') + "\"}";
    g = parse_growth_mib(big, ok, error);
    std::printf("  21 MiB image string: ok=%d, peak growth %.1f MiB (%.2fx the body)\n", ok, g,
                g / (big.size() / 1048576.0));
    CHECK(ok && g < 1.25 * (big.size() / 1048576.0), "big string: ok=%d growth %.1f MiB", ok, g);
}
#endif

}  // namespace

int main() {
    std::printf("flat objects\n");
    test_accepts_flat_objects();
    std::printf("rejections\n");
    test_rejections();
#ifdef __linux__
    std::printf("hostile bodies\n");
    test_hostile_bodies_stay_small();
#endif
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

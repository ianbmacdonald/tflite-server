// Standalone tests for src/image_preprocess.cpp and the budgeted stb decoder.
// Usage: test_image_preprocess <tests/data dir>
// Fixtures come from tools/make_preprocess_goldens.py (Pillow is the reference).

#include "image_preprocess.h"
#include "png_bomb.h"

#include <pthread.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_checks = 0;
std::string g_data;

#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        ++g_checks;                                                        \
        if (!(cond)) {                                                     \
            ++g_failures;                                                  \
            std::fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); \
            std::fprintf(stderr, __VA_ARGS__);                             \
            std::fprintf(stderr, "\n");                                    \
        }                                                                  \
    } while (0)

std::string read_file(const std::string& name) {
    std::ifstream f(g_data + "/" + name, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open fixture %s/%s\n", g_data.c_str(), name.c_str());
        std::exit(2);
    }
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

struct Diff {
    double mean = 0;
    double max = 0;
};

template <typename A, typename B>
Diff diff(const A* a, const B* b, size_t n) {
    Diff d;
    for (size_t i = 0; i < n; ++i) {
        const double e = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        d.mean += e;
        if (e > d.max) d.max = e;
    }
    d.mean /= static_cast<double>(n);
    return d;
}

imgproc::DecodeResult decode(const std::string& bytes, imgproc::DecodeLimits lim = {}) {
    return imgproc::decode_rgb8(bytes, lim);
}

void test_sniff_and_base64() {
    using imgproc::ImageFormat;
    CHECK(imgproc::sniff_image_format(read_file("small_baseline.jpg")) == ImageFormat::Jpeg, "jpeg");
    CHECK(imgproc::sniff_image_format(read_file("small_rgb.png")) == ImageFormat::Png, "png");
    CHECK(imgproc::sniff_image_format("GIF89a......") == ImageFormat::Unknown, "gif");
    CHECK(imgproc::sniff_image_format("") == ImageFormat::Unknown, "empty");
    CHECK(imgproc::sniff_image_format("\xFF\xD8") == ImageFormat::Unknown, "short jpeg magic");

    std::string out;
    CHECK(imgproc::strict_base64_decode("aGVsbG8=", out) && out == "hello", "pad1 '%s'", out.c_str());
    CHECK(imgproc::strict_base64_decode("aGVsbA==", out) && out == "hell", "pad2 '%s'", out.c_str());
    CHECK(imgproc::strict_base64_decode("aGVs", out) && out == "hel", "nopad");
    CHECK(imgproc::strict_base64_decode("aGVs\r\nbG8=\n", out) && out == "hello", "whitespace");
    CHECK(!imgproc::strict_base64_decode("aGVsbG8", out), "length not multiple of 4");
    CHECK(!imgproc::strict_base64_decode("aGV*bG8=", out), "bad alphabet");
    CHECK(!imgproc::strict_base64_decode("aG=sbG8=", out), "padding in the middle");
    CHECK(!imgproc::strict_base64_decode("aGVs=G8=", out), "padding before the last group");
    CHECK(!imgproc::strict_base64_decode("a===", out), "three pad chars");
    CHECK(!imgproc::strict_base64_decode("", out), "empty");
    CHECK(!imgproc::strict_base64_decode("aGVsbG8=aGVs", out), "data after padding");
    CHECK(!imgproc::strict_base64_decode("aGVsbA===", out), "extra pad char after a full group");
    CHECK(!imgproc::strict_base64_decode("aGVsbA=", out), "short final group");
    CHECK(!imgproc::strict_base64_decode("  \n ", out), "whitespace only");
    CHECK(imgproc::strict_base64_decode(" a G V s b A = = ", out) && out == "hell", "spaces inside the padding");
}

void test_decode_variants() {
    struct Case {
        const char* file;
        const char* golden;
        double max_mean;
        double max_abs;
    };
    // PNG decodes are lossless, so they must match Pillow exactly. JPEG decoders
    // differ in IDCT and chroma upsampling, so those get a tolerance.
    const Case cases[] = {
        {"small_rgb.png", "small_pil.rgb", 0.0, 0.0},
        {"small_rgba.png", "small_pil.rgb", 0.0, 0.0},
        {"small_gray.png", "small_gray_pil.rgb", 0.0, 0.0},
        {"small_rgba16.png", "small_pil.rgb", 0.0, 0.0},
        {"small_baseline.jpg", "small_baseline_pil.rgb", 0.5, 8.0},
        {"small_progressive.jpg", "small_progressive_pil.rgb", 0.5, 8.0},
        {"small_cmyk.jpg", "small_cmyk_pil.rgb", 0.5, 8.0},
    };
    for (const auto& c : cases) {
        auto r = decode(read_file(c.file));
        CHECK(r.error == imgproc::DecodeError::None, "%s: %s", c.file, r.message.c_str());
        if (r.error != imgproc::DecodeError::None) continue;
        CHECK(r.image.width == 64 && r.image.height == 48, "%s: %dx%d", c.file, r.image.width,
              r.image.height);
        const std::string golden = read_file(c.golden);
        CHECK(golden.size() == 64u * 48u * 3u, "%s golden size", c.golden);
        const Diff d = diff(r.image.pixels.get(),
                            reinterpret_cast<const unsigned char*>(golden.data()), golden.size());
        std::printf("  decode %-24s vs Pillow: mean %.3f max %.0f, budget peak %llu of %llu\n",
                    c.file, d.mean, d.max, (unsigned long long)r.budget_peak,
                    (unsigned long long)r.budget_limit);
        CHECK(d.mean <= c.max_mean && d.max <= c.max_abs, "%s: mean %.3f max %.0f", c.file, d.mean,
              d.max);
        CHECK(r.budget_peak <= r.budget_limit, "%s peak", c.file);
    }
}

// Patch the SOF0 segment of a baseline JPEG.
std::string patch_sof0(std::string jpg, int marker, int precision) {
    for (size_t i = 2; i + 4 < jpg.size(); ++i) {
        if (static_cast<unsigned char>(jpg[i]) == 0xFF && static_cast<unsigned char>(jpg[i + 1]) == 0xC0) {
            if (marker >= 0) jpg[i + 1] = static_cast<char>(marker);
            if (precision >= 0) jpg[i + 4] = static_cast<char>(precision);
            return jpg;
        }
    }
    std::fprintf(stderr, "no SOF0 in fixture\n");
    std::exit(2);
}

void test_rejects() {
    const std::string jpg = read_file("small_baseline.jpg");
    const std::string png = read_file("small_rgb.png");

    auto twelve = decode(patch_sof0(jpg, -1, 12));
    CHECK(twelve.error == imgproc::DecodeError::Corrupt, "12-bit JPEG: %s", twelve.message.c_str());
    std::printf("  12-bit JPEG -> %s\n", twelve.message.c_str());
    auto arith = decode(patch_sof0(jpg, 0xC9, -1));
    CHECK(arith.error == imgproc::DecodeError::Corrupt, "arithmetic JPEG: %s", arith.message.c_str());
    std::printf("  arithmetic JPEG -> %s\n", arith.message.c_str());

    auto gif = decode("GIF89a\x01\x00\x01\x00\x00\x00\x00;");
    CHECK(gif.error == imgproc::DecodeError::NotAnImage, "gif");
    auto garbage = decode(std::string("\xFF\xD8\xFF") + std::string(4096, '\x5A'));
    CHECK(garbage.error != imgproc::DecodeError::None, "garbage after JPEG magic");
    auto garbage_png = decode(png.substr(0, 8) + std::string(4096, '\x00'));
    CHECK(garbage_png.error != imgproc::DecodeError::None, "garbage after PNG magic");

    // Every truncation must fail cleanly or decode; none may crash. stb pads a
    // truncated JPEG scan with zeros, so late cuts can legitimately succeed.
    int failed = 0, total = 0;
    for (const std::string* src : {&jpg, &png}) {
        for (size_t n = 0; n < src->size(); n += 3) {
            auto r = decode(src->substr(0, n));
            ++total;
            if (r.error != imgproc::DecodeError::None) ++failed;
            CHECK(r.error != imgproc::DecodeError::None || r.image.pixels, "truncation %zu", n);
        }
    }
    std::printf("  truncations: %d of %d rejected, none crashed\n", failed, total);
    auto half_png = decode(png.substr(0, png.size() / 2));
    CHECK(half_png.error == imgproc::DecodeError::Corrupt, "half a PNG: %s", half_png.message.c_str());

    imgproc::DecodeLimits tiny;
    tiny.max_pixels = 100;
    auto cap = decode(png, tiny);
    CHECK(cap.error == imgproc::DecodeError::TooManyPixels, "pixel cap: %s", cap.message.c_str());
    CHECK(cap.budget_limit == 0 && cap.budget_peak == 0, "pixel cap must refuse before decode");
    std::printf("  pixel cap -> %s\n", cap.message.c_str());
}

void test_zip_bomb() {
    const uint64_t inflated = 272ull << 20;
    const std::string bomb = png_bomb::make(64, 64, inflated);
    std::printf("  zip bomb: %zu-byte PNG, 64x64 IHDR, IDAT inflates to %llu MiB\n", bomb.size(),
                (unsigned long long)(inflated >> 20));
    auto r = decode(bomb);
    CHECK(r.error == imgproc::DecodeError::BudgetExceeded, "bomb: %d %s", int(r.error),
          r.message.c_str());
    CHECK(r.message == "image decode exceeded memory budget", "bomb message '%s'", r.message.c_str());
    CHECK(r.budget_peak <= r.budget_limit, "bomb peak %llu > limit %llu",
          (unsigned long long)r.budget_peak, (unsigned long long)r.budget_limit);
    std::printf("  zip bomb -> %s (peak %llu of budget %llu bytes)\n", r.message.c_str(),
                (unsigned long long)r.budget_peak, (unsigned long long)r.budget_limit);

    // The same generator with a truthful stream length must decode, so the
    // failure above is the budget and not a malformed stream.
    auto ok = decode(png_bomb::make(64, 64, 64 * (64 * 3 + 1)));
    CHECK(ok.error == imgproc::DecodeError::None, "well-formed generator PNG: %s", ok.message.c_str());
}

void test_budget_ceiling() {
    imgproc::DecodeLimits lim;
    lim.budget_factor = 64;
    lim.max_budget = 256 * 1024;
    auto r = imgproc::decode_rgb8(read_file("synth_600x512.png"), lim);
    CHECK(r.budget_limit == lim.max_budget, "limit %llu, want the ceiling %llu",
          (unsigned long long)r.budget_limit, (unsigned long long)lim.max_budget);
    CHECK(r.error == imgproc::DecodeError::BudgetExceeded, "600x512 under a 256 KiB ceiling: %s",
          r.message.c_str());
    CHECK(r.budget_peak <= lim.max_budget, "peak %llu over the ceiling", (unsigned long long)r.budget_peak);
    auto ok = imgproc::decode_rgb8(read_file("synth_600x512.png"), imgproc::DecodeLimits{});
    CHECK(ok.error == imgproc::DecodeError::None && ok.budget_limit < imgproc::DecodeLimits{}.max_budget,
          "default ceiling must not bind on 600x512: %s", ok.message.c_str());
}

void test_budget_fits_worst_case_jpeg() {
    imgproc::DecodeLimits lim;
    lim.budget_slack = 64 * 1024;  // so the per-pixel factor alone has to cover the decode
    for (const char* f : {"cmyk_progressive_640x480.jpg", "ycc444_progressive_640x480.jpg"}) {
        auto r = decode(read_file(f), lim);
        CHECK(r.error == imgproc::DecodeError::None, "%s: %s", f, r.message.c_str());
        std::printf("  %-32s peak %.2f bytes/pixel (factor %llu)\n", f,
                    double(r.budget_peak) / (640.0 * 480.0), (unsigned long long)lim.budget_factor);
    }
}

void test_resampler_invariants() {
    std::vector<unsigned char> flat(97 * 61 * 3, 173);
    for (auto [w, h] : {std::pair{224, 224}, std::pair{13, 200}, std::pair{1, 1}, std::pair{300, 7}}) {
        auto out = imgproc::resample_triangle(flat.data(), 97, 61, 3, w, h);
        double worst = 0;
        for (float v : out) worst = std::max(worst, std::fabs(v - 173.0));
        CHECK(out.size() == size_t(w) * h * 3 && worst < 1e-3, "constant %dx%d drift %g", w, h, worst);
    }
    // A linear ramp is symmetric about its center, so the 1x1 triangle-filtered
    // value is the ramp's mean.
    const int rw = 255, rh = 9;
    std::vector<unsigned char> ramp(rw * rh);
    for (int y = 0; y < rh; ++y)
        for (int x = 0; x < rw; ++x) ramp[y * rw + x] = static_cast<unsigned char>(x);
    auto one = imgproc::resample_triangle(ramp.data(), rw, rh, 1, 1, 1);
    CHECK(std::fabs(one[0] - 127.0) < 1e-3, "ramp 1x1 = %f, want 127", one[0]);
    std::vector<unsigned char> img(31 * 17 * 3);
    for (size_t i = 0; i < img.size(); ++i) img[i] = static_cast<unsigned char>((i * 37) & 0xFF);
    auto same = imgproc::resample_triangle(img.data(), 31, 17, 3, 31, 17);
    const Diff d = diff(same.data(), img.data(), img.size());
    CHECK(d.max < 1e-4, "same-size resize must be the identity (max %g)", d.max);
}

void check_golden(const std::string& src, int w, int h, const char* golden, double max_mean,
                  double max_abs) {
    auto r = decode(read_file(src));
    CHECK(r.error == imgproc::DecodeError::None, "%s: %s", src.c_str(), r.message.c_str());
    if (r.error != imgproc::DecodeError::None) return;
    auto out = imgproc::resample_triangle(r.image.pixels.get(), r.image.width, r.image.height, 3, w, h);
    const std::string g = read_file(golden);
    CHECK(g.size() == out.size(), "%s size", golden);
    const Diff d = diff(out.data(), reinterpret_cast<const unsigned char*>(g.data()), out.size());
    std::printf("  %s %dx%d -> %dx%d vs Pillow BILINEAR: mean %.3f max %.3f\n", src.c_str(),
                r.image.width, r.image.height, w, h, d.mean, d.max);
    CHECK(d.mean <= max_mean && d.max <= max_abs, "%s: mean %.3f max %.3f", golden, d.mean, d.max);
}

void test_pil_goldens() {
    // Lossless source: the resampler emulates Pillow's fixed-point path, so the
    // result is byte identical.
    check_golden("synth_600x512.png", 224, 224, "synth_pil224.rgb", 0.0, 0.0);
    check_golden("synth_600x512.png", 331, 97, "synth_pil331x97.rgb", 0.0, 0.0);
    // JPEG source: only the decoders differ (stb vs libjpeg-turbo IDCT and
    // chroma upsampling), so the mean is held to 1.0 and a single value to 4.0.
    check_golden("grace_hopper.jpg", 224, 224, "grace_hopper_pil224.rgb", 1.0, 4.0);
}

struct StackRun {
    std::string bytes;
    std::vector<float> tensor;
    bool ok = false;
};

void* run_pipeline(void* arg) {
    auto* s = static_cast<StackRun*>(arg);
    auto r = imgproc::decode_rgb8(s->bytes, {});
    if (r.error != imgproc::DecodeError::None) return nullptr;
    s->tensor = imgproc::resample_triangle(r.image.pixels.get(), r.image.width, r.image.height, 3,
                                           224, 224);
    imgproc::normalize_in_place(s->tensor, {127.5f, 127.5f, 127.5f}, {127.5f, 127.5f, 127.5f});
    s->ok = true;
    return nullptr;
}

void test_small_stack() {
    for (const char* f : {"grace_hopper.jpg", "synth_600x512.png", "cmyk_progressive_640x480.jpg"}) {
        StackRun main_run{read_file(f)}, small_run{read_file(f)};
        run_pipeline(&main_run);
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, 128 * 1024);
        pthread_t t;
        const int rc = pthread_create(&t, &attr, run_pipeline, &small_run);
        pthread_attr_destroy(&attr);
        CHECK(rc == 0, "pthread_create: %d", rc);
        if (rc == 0) pthread_join(t, nullptr);
        CHECK(main_run.ok && small_run.ok && main_run.tensor == small_run.tensor,
              "%s on a 128 KiB stack", f);
        std::printf("  %s: decode+resize+normalize on a 128 KiB-stack thread: %s\n", f,
                    small_run.ok && main_run.tensor == small_run.tensor ? "ok" : "FAILED");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_image_preprocess <tests/data dir>\n");
        return 2;
    }
    g_data = argv[1];
    std::printf("sniff and base64\n");
    test_sniff_and_base64();
    std::printf("decode variants\n");
    test_decode_variants();
    std::printf("rejects\n");
    test_rejects();
    std::printf("zip bomb\n");
    test_zip_bomb();
    std::printf("budget ceiling\n");
    test_budget_ceiling();
    std::printf("budget on worst-case JPEGs\n");
    test_budget_fits_worst_case_jpeg();
    std::printf("resampler invariants\n");
    test_resampler_invariants();
    std::printf("Pillow goldens\n");
    test_pil_goldens();
    std::printf("small stack\n");
    test_small_stack();
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

// Mutation smoke for decode_rgb8, meant to run under -fsanitize=address,undefined.
// Usage: fuzz_image_decode <tests/data dir> [iterations] [seed]
// Seeds are the fixture images plus a generated PNG zip bomb, so memory growth
// paths are mutated too, not just the parsers' error paths.

#include "image_preprocess.h"
#include "png_bomb.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: fuzz_image_decode <tests/data dir> [iterations] [seed]\n");
        return 2;
    }
    const std::string dir = argv[1];
    const long iterations = argc > 2 ? std::atol(argv[2]) : 10000;
    const unsigned seed = argc > 3 ? static_cast<unsigned>(std::atol(argv[3])) : 1;

    std::vector<std::string> seeds;
    for (const char* f : {"small_baseline.jpg", "small_progressive.jpg", "small_cmyk.jpg",
                          "small_rgb.png", "small_rgba.png", "small_gray.png", "small_rgba16.png",
                          "ycc444_progressive_640x480.jpg", "grace_hopper.jpg"}) {
        std::ifstream in(dir + "/" + f, std::ios::binary);
        if (!in) {
            std::fprintf(stderr, "cannot open %s/%s\n", dir.c_str(), f);
            return 2;
        }
        seeds.emplace_back(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }
    seeds.push_back(png_bomb::make(64, 64, 32u << 20));

    std::mt19937 rng(seed);
    auto pick = [&](size_t n) { return std::uniform_int_distribution<size_t>(0, n - 1)(rng); };
    long ok = 0, rejected = 0, budget = 0;
    for (long i = 0; i < iterations; ++i) {
        std::string s = seeds[pick(seeds.size())];
        const size_t edits = 1 + pick(8);
        for (size_t e = 0; e < edits && !s.empty(); ++e) {
            switch (pick(5)) {
                case 0:  // flip bits somewhere past the magic, most of the time
                    s[pick(s.size())] ^= static_cast<char>(1u << pick(8));
                    break;
                case 1:  // random byte
                    s[pick(s.size())] = static_cast<char>(pick(256));
                    break;
                case 2:  // truncate
                    s.resize(pick(s.size()) + 1);
                    break;
                case 3: {  // duplicate a slice
                    const size_t a = pick(s.size()), n = std::min<size_t>(pick(512) + 1, s.size() - a);
                    s.insert(pick(s.size()), s.substr(a, n));
                    break;
                }
                default: {  // header-area edit: dimensions, segment lengths, chunk sizes
                    const size_t lim = std::min<size_t>(s.size(), 64);
                    s[pick(lim)] = static_cast<char>(pick(256));
                    break;
                }
            }
        }
        auto r = imgproc::decode_rgb8(s, {});
        if (r.error == imgproc::DecodeError::None) {
            ++ok;
            auto t = imgproc::resample_triangle(r.image.pixels.get(), r.image.width, r.image.height, 3,
                                                224, 224);
            if (t.size() != 224u * 224u * 3u) return 1;
        } else if (r.error == imgproc::DecodeError::BudgetExceeded) {
            ++budget;
        } else {
            ++rejected;
        }
        if (r.budget_peak > r.budget_limit) {
            std::fprintf(stderr, "iteration %ld: peak %llu over budget %llu\n", i,
                         (unsigned long long)r.budget_peak, (unsigned long long)r.budget_limit);
            return 1;
        }
    }
    std::printf("fuzz_image_decode: %ld iterations (seed %u): %ld decoded, %ld rejected, %ld over budget\n",
                iterations, seed, ok, rejected, budget);
    return 0;
}

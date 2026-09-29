#pragma once

// Builds an 8-bit RGB PNG whose IDAT zlib stream inflates to `inflated` zero
// bytes regardless of the IHDR size. Fixed-Huffman deflate: one literal 0, then
// (length 258, distance 1) copies, 13 bits per 258 bytes, so a 272 MiB stream
// is about 1.8 MB and is generated in memory instead of committed.

#include <cstdint>
#include <string>

namespace png_bomb {

inline uint32_t crc32(const std::string& s) {
    uint32_t c = 0xFFFFFFFFu;
    for (unsigned char b : s) {
        c ^= b;
        for (int k = 0; k < 8; ++k) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return ~c;
}

inline void be32(std::string& s, uint32_t v) {
    for (int sh = 24; sh >= 0; sh -= 8) s.push_back(static_cast<char>((v >> sh) & 0xFF));
}

inline void chunk(std::string& png, const char* type, const std::string& data) {
    be32(png, static_cast<uint32_t>(data.size()));
    std::string body = std::string(type, 4) + data;
    png += body;
    be32(png, crc32(body));
}

class BitWriter {
public:
    void bits(uint32_t v, int n) {  // LSB first, as deflate packs header and extra bits
        for (int i = 0; i < n; ++i) put((v >> i) & 1u);
    }
    void code(uint32_t v, int n) {  // Huffman codes go MSB first
        for (int i = n - 1; i >= 0; --i) put((v >> i) & 1u);
    }
    std::string finish() {
        if (nbits_) out_.push_back(static_cast<char>(cur_));
        return out_;
    }

private:
    void put(uint32_t b) {
        cur_ |= b << nbits_;
        if (++nbits_ == 8) {
            out_.push_back(static_cast<char>(cur_));
            cur_ = 0;
            nbits_ = 0;
        }
    }
    std::string out_;
    uint32_t cur_ = 0;
    int nbits_ = 0;
};

inline std::string make(uint32_t width, uint32_t height, uint64_t inflated) {
    if (inflated == 0) inflated = 1;
    const uint64_t copies = (inflated - 1) / 258;
    const uint64_t tail = (inflated - 1) % 258;
    BitWriter w;
    w.bits(1, 1);         // BFINAL
    w.bits(1, 2);         // BTYPE = fixed Huffman
    w.code(0x30, 8);      // literal 0
    for (uint64_t i = 0; i < copies; ++i) {
        w.code(0xC5, 8);  // length symbol 285 = 258, no extra bits
        w.code(0, 5);     // distance code 0 = 1
    }
    for (uint64_t i = 0; i < tail; ++i) w.code(0x30, 8);
    w.code(0, 7);         // end of block
    std::string z = "\x78\x01";
    z += w.finish();
    const uint32_t a = 1, b = static_cast<uint32_t>(inflated % 65521);  // adler32 of zeros
    be32(z, (b << 16) | a);

    std::string png = "\x89PNG\r\n\x1a\n";
    std::string ihdr;
    be32(ihdr, width);
    be32(ihdr, height);
    ihdr += std::string("\x08\x02\x00\x00\x00", 5);  // 8-bit RGB, no interlace
    chunk(png, "IHDR", ihdr);
    chunk(png, "IDAT", z);
    chunk(png, "IEND", "");
    return png;
}

}  // namespace png_bomb

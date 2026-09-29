#!/usr/bin/env python3
"""Write the fixtures tests/test_image_preprocess.cpp reads (tests/data).

Usage: make_preprocess_goldens.py <grace_hopper.jpg> [out_dir]

grace_hopper.jpg is the US Navy photo TensorFlow's examples use (public domain):
https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg
sha256 a8ca6d734765703b09728ab47fe59f473d93ae3967fc24c7c0288c3c7adb7130.

The goldens are Pillow's own output, which is the reference the classifier
parity numbers come from: Image.open(...).convert("RGB").resize(size, BILINEAR).
Raw goldens are interleaved uint8 RGB, row major, with no header.
"""

import hashlib
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

GRACE_HOPPER_SHA256 = "a8ca6d734765703b09728ab47fe59f473d93ae3967fc24c7c0288c3c7adb7130"


def synthetic(w, h):
    """Gradients, hard edges and a fine checkerboard: exercises every tap."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 255 * x / (w - 1)
    g = 255 * y / (h - 1)
    b = 127.5 + 127.5 * np.sin(x / 7.0) * np.cos(y / 11.0)
    img = np.stack([r, g, b], axis=-1)
    img[(x // 32 + y // 32) % 2 == 0] *= 0.5
    img[(h // 3 < y) & (y < h // 3 + 40)] = [250, 10, 10]
    checker = ((x.astype(int) + y.astype(int)) % 2 == 0) & (x > w - 64) & (y > h - 64)
    img[checker] = 255
    return Image.fromarray(np.clip(np.rint(img), 0, 255).astype(np.uint8), "RGB")


def png_chunk(kind, data):
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def write_png16(path, arr, color_type):
    """16-bit PNG writer (Pillow cannot write 16-bit RGB/RGBA)."""
    h, w = arr.shape[:2]
    rows = arr.astype(">u2").reshape(h, -1).tobytes()
    stride = len(rows) // h
    raw = b"".join(b"\x00" + rows[i * stride : (i + 1) * stride] for i in range(h))
    ihdr = struct.pack(">IIBBBBB", w, h, 16, color_type, 0, 0, 0)
    Path(path).write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(raw, 9))
        + png_chunk(b"IEND", b"")
    )


def raw(path, img):
    Path(path).write_bytes(np.asarray(img.convert("RGB"), dtype=np.uint8).tobytes())


def main():
    src = Path(sys.argv[1])
    out = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).parent.parent / "tests" / "data")
    out.mkdir(parents=True, exist_ok=True)
    blob = src.read_bytes()
    if hashlib.sha256(blob).hexdigest() != GRACE_HOPPER_SHA256:
        sys.exit(f"{src}: unexpected sha256")
    (out / "grace_hopper.jpg").write_bytes(blob)
    gh = Image.open(src).convert("RGB")
    raw(out / "grace_hopper_pil224.rgb", gh.resize((224, 224), Image.BILINEAR))

    syn = synthetic(600, 512)
    syn.save(out / "synth_600x512.png", optimize=True)
    raw(out / "synth_pil224.rgb", syn.resize((224, 224), Image.BILINEAR))
    raw(out / "synth_pil331x97.rgb", syn.resize((331, 97), Image.BILINEAR))

    small = syn.resize((64, 48), Image.BILINEAR)
    raw(out / "small_pil.rgb", small)
    small.save(out / "small_rgb.png")
    small.convert("RGBA").save(out / "small_rgba.png")
    small.convert("L").save(out / "small_gray.png")
    raw(out / "small_gray_pil.rgb", small.convert("L"))
    rgba16 = np.concatenate(
        [np.asarray(small, dtype=np.uint16) * 257, np.full((48, 64, 1), 65535, np.uint16)], axis=-1
    )
    write_png16(out / "small_rgba16.png", rgba16, 6)
    small.save(out / "small_baseline.jpg", quality=92)
    raw(out / "small_baseline_pil.rgb", Image.open(out / "small_baseline.jpg"))
    small.save(out / "small_progressive.jpg", quality=92, progressive=True)
    raw(out / "small_progressive_pil.rgb", Image.open(out / "small_progressive.jpg"))
    small.convert("CMYK").save(out / "small_cmyk.jpg", quality=92)
    raw(out / "small_cmyk_pil.rgb", Image.open(out / "small_cmyk.jpg"))

    # Worst case for the decode budget: 4 components, progressive, no chroma
    # subsampling, so stb holds a full coefficient plane per component.
    mid = syn.resize((640, 480), Image.BILINEAR)
    mid.convert("CMYK").save(out / "cmyk_progressive_640x480.jpg", quality=90, progressive=True)
    mid.save(out / "ycc444_progressive_640x480.jpg", quality=90, progressive=True, subsampling=0)

    for p in sorted(out.iterdir()):
        print(f"{p.stat().st_size:>8}  {p.name}")


if __name__ == "__main__":
    main()

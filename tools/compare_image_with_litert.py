#!/usr/bin/env python3
"""Compare tflite-server /classify/image with ai_edge_litert + Pillow.

Usage: compare_image_with_litert.py <model_dir> <port> <image> [<image> ...]

The reference preprocesses as the manifest says (Pillow BILINEAR stretch,
(x - mean) / std) and runs the same model.tflite in the ai_edge_litert
Interpreter. Pass: the same top-1 index for every image and a maximum
|score difference| below 0.02 over all classes.
"""

import http.client
import json
import sys
import uuid
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image

TOLERANCE = 0.02


def post_image(port, path, top_k):
    boundary = uuid.uuid4().hex
    body = (
        (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"top_k\"\r\n\r\n{top_k}\r\n"
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; "
            f'filename="{Path(path).name}"\r\nContent-Type: application/octet-stream\r\n\r\n'
        ).encode()
        + Path(path).read_bytes()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    c.request("POST", "/classify/image", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    r = c.getresponse()
    data = r.read()
    if r.status != 200:
        raise RuntimeError(f"{path}: HTTP {r.status}: {data[:300]!r}")
    return json.loads(data)


def main():
    model_dir, port, images = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
    manifest = json.loads((model_dir / "manifest.json").read_text())
    pre = manifest.get("preprocess", {})
    mean = np.asarray(pre.get("mean", 127.5), dtype=np.float32)
    std = np.asarray(pre.get("std", 127.5), dtype=np.float32)
    labels = (model_dir / manifest.get("labels_file", "labels.txt")).read_text().splitlines()
    it = Interpreter(model_path=str(model_dir / "model.tflite"))
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    h, w = int(inp["shape"][1]), int(inp["shape"][2])

    worst, mismatches = 0.0, 0
    for path in images:
        img = Image.open(path).convert("RGB")
        x = (np.asarray(img.resize((w, h), Image.BILINEAR), dtype=np.float32) - mean) / std
        it.set_tensor(inp["index"], x[None].astype(np.float32))
        it.invoke()
        ref = it.get_tensor(out["index"])[0]
        got = post_image(port, path, len(labels))
        srv = np.zeros_like(ref)
        for p in got["predictions"]:
            srv[p["index"]] = p["score"]
        delta = float(np.max(np.abs(srv - ref)))
        worst = max(worst, delta)
        same = int(np.argmax(ref)) == got["predictions"][0]["index"]
        mismatches += not same
        print(
            f"{'ok ' if same and delta < TOLERANCE else 'BAD'} {Path(path).name:40s} "
            f"{img.size[0]:>4}x{img.size[1]:<4} ref {int(np.argmax(ref)):>4} {float(ref.max()):.6f}  "
            f"srv {got['predictions'][0]['index']:>4} {got['predictions'][0]['score']:.6f}  "
            f"max|d| {delta:.6f}  {labels[int(np.argmax(ref))]}"
        )
    print(f"{len(images)} images: {mismatches} top-1 mismatches, max |score difference| {worst:.6f}")
    sys.exit(0 if mismatches == 0 and worst < TOLERANCE else 1)


if __name__ == "__main__":
    main()

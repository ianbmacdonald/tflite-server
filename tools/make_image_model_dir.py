#!/usr/bin/env python3
"""Build a tflite-server image-classification model directory for TF MobileNetV2 1.0 224.

Usage:
  make_image_model_dir.py <out_dir> [--tgz FILE] [--labels FILE] [--validate-image FILE]

Without --tgz / --labels the two files are downloaded from download.tensorflow.org.
With them, nothing is fetched (offline staging). Both are verified by sha256.

Writes model.tflite, labels.txt, manifest.json, LICENSE, README.md and, when
ai_edge_litert and Pillow are importable and --validate-image is given,
validation.json with the reference top-5 (Pillow BILINEAR stretch, x/127.5-1).
"""

import argparse
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

TGZ_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz"
TGZ_MEMBER = "mobilenet_v2_1.0_224.tflite"
MODEL_SHA256 = "9f3bc29e38e90842a852bfed957dbf5e36f2d97a91dd17736b1e5c0aca8d3303"
MODEL_SIZE = 13978596
LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
LABELS_SHA256 = "536feacc519de3d418de26b2effb4d75694a8c4c0063e36499a46fa8061e2da9"

MANIFEST = {
    "task": "image-classification",
    "labels_file": "labels.txt",
    "score_normalization": "none",
    "preprocess": {
        "resize": "stretch",
        "mean": [127.5, 127.5, 127.5],
        "std": [127.5, 127.5, 127.5],
        "channel_order": "RGB",
        "layout": "auto",
    },
    "top_k_default": 5,
}


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def fetch(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def reference_top5(model, labels, image):
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter
    from PIL import Image

    it = Interpreter(model_path=str(model))
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    h, w = int(inp["shape"][1]), int(inp["shape"][2])
    img = Image.open(image).convert("RGB")
    x = np.asarray(img.resize((w, h), Image.BILINEAR), dtype=np.float32) / 127.5 - 1.0
    it.set_tensor(inp["index"], x[None])
    it.invoke()
    p = it.get_tensor(out["index"])[0]
    top = np.argsort(-p, kind="stable")[:5]
    return {
        "image": Path(image).name,
        "image_sha256": sha256(Path(image).read_bytes()),
        "image_size": list(img.size),
        "reference": "ai_edge_litert Interpreter + Pillow BILINEAR stretch, x/127.5-1",
        "top5": [{"index": int(i), "label": labels[i], "score": round(float(p[i]), 6)} for i in top],
    }


README = """# MobileNetV2 1.0 224 (TFLite, fp32), tflite-server model directory

ImageNet classifier for tflite-server's `POST /classify/image`.

- `model.tflite`: `mobilenet_v2_1.0_224.tflite` from the TensorFlow hosted-models archive
  {tgz_url}
  ({size} bytes, sha256 `{model_sha}`). Input float32 `[1,224,224,3]` NHWC RGB on the -1..1
  scale; output float32 `[1,1001]`, already softmax. It has no SignatureDefs; LiteRT serves it
  through its default signature.
- `labels.txt`: {labels_url} (sha256 `{labels_sha}`), 1001 lines. Index 0 is `background`.
  Some names repeat (`crane`, `maillot`), so clients should key on the index.
- `manifest.json`: stretch resize to 224x224 (Pillow BILINEAR semantics), then (x - 127.5) / 127.5.

License: the weights are from tensorflow/models research/slim, Apache-2.0 (see `LICENSE`). The
ImageNet dataset they were trained on has its own, non-commercial research terms.

Expected top-5 for TensorFlow's `grace_hopper.jpg` (see `validation.json` when present):
653 military uniform 0.803491, 440 bearskin 0.037449, 668 mortarboard 0.017009.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--tgz")
    ap.add_argument("--labels")
    ap.add_argument("--validate-image")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tgz = Path(a.tgz).read_bytes() if a.tgz else fetch(TGZ_URL)
    with tarfile.open(fileobj=io.BytesIO(tgz)) as t:
        names = [m for m in t.getnames() if m.lstrip("./") == TGZ_MEMBER]
        if not names:
            sys.exit(f"{TGZ_MEMBER} not in the archive")
        model = t.extractfile(names[0]).read()
    if len(model) != MODEL_SIZE or sha256(model) != MODEL_SHA256:
        sys.exit("model.tflite: unexpected size or sha256")
    labels_blob = Path(a.labels).read_bytes() if a.labels else fetch(LABELS_URL)
    if sha256(labels_blob) != LABELS_SHA256:
        sys.exit("labels: unexpected sha256")

    (out / "model.tflite").write_bytes(model)
    (out / "labels.txt").write_bytes(labels_blob)
    (out / "manifest.json").write_text(json.dumps(MANIFEST, indent=2) + "\n")
    license_src = Path(__file__).resolve().parent.parent / "LICENSE"
    (out / "LICENSE").write_bytes(license_src.read_bytes())
    (out / "README.md").write_text(
        README.format(
            tgz_url=TGZ_URL,
            size=MODEL_SIZE,
            model_sha=MODEL_SHA256,
            labels_url=LABELS_URL,
            labels_sha=LABELS_SHA256,
        )
    )
    if a.validate_image:
        labels = labels_blob.decode().splitlines()
        v = reference_top5(out / "model.tflite", labels, a.validate_image)
        (out / "validation.json").write_text(json.dumps(v, indent=2) + "\n")
        print(json.dumps(v["top5"], indent=1))
    for p in sorted(out.iterdir()):
        print(f"{p.stat().st_size:>10}  {p.name}")


if __name__ == "__main__":
    main()

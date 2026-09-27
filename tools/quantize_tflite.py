"""Quantize an export_tflite.py model directory and validate it against PyTorch.

    python quantize_tflite.py <hf_model_id> <fp32_dir> <out_dir> [--recipe dynamic_wi8_afp32]

Copies the tokenizer/config/manifest files, writes the quantized model.tflite and
a validation.json computed exactly like export_tflite.py's, plus top-label
agreement, so a quantized directory is checked against the same bar.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from ai_edge_litert.interpreter import Interpreter
from ai_edge_quantizer import quantizer, recipe
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from export_tflite import FIXTURES, Logits, encode, softmax

EXTRA = [
    "Your parcel could not be delivered. Pay the 1.99 EUR redelivery fee here: http://dhl-redelivery.example/pay",
    "Hi team, attached is the Q3 budget spreadsheet. Please review before Friday's meeting.",
    "ok",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("fp32_dir")
    ap.add_argument("out")
    ap.add_argument("--recipe", default="dynamic_wi8_afp32")
    args = ap.parse_args()
    src, out = Path(args.fp32_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    q = quantizer.Quantizer(str(src / "model.tflite"), getattr(recipe, args.recipe)())
    result = q.quantize()
    (out / "model.tflite").write_bytes(result.quantized_model)
    for f in src.iterdir():
        if f.name not in ("model.tflite", "validation.json") and f.is_file() and "xnnpack" not in f.name:
            shutil.copy(f, out / f.name)

    tok = AutoTokenizer.from_pretrained(args.model_id)
    wrapped = Logits(AutoModelForSequenceClassification.from_pretrained(args.model_id).eval()).eval()
    manifest = json.loads((out / "manifest.json").read_text())
    interp = Interpreter(model_path=str(out / "model.tflite"))
    seq_lens = sorted(int(k.split("_")[1]) for k in interp.get_signature_list())
    max_delta, agree, total, per = 0.0, 0, 0, []
    for n in seq_lens:
        runner = interp.get_signature_runner(f"seq_{n}")
        for text in FIXTURES + EXTRA:
            ids, mask = encode(tok, text, n)
            with torch.no_grad():
                ref = softmax(wrapped(torch.from_numpy(ids), torch.from_numpy(mask)).numpy())[0]
            got = softmax(next(iter(runner(input_ids=ids, attention_mask=mask).values())))[0]
            delta = float(np.abs(ref - got).max())
            max_delta = max(max_delta, delta)
            same = int(ref.argmax() == got.argmax())
            agree += same
            total += 1
            per.append({"signature": f"seq_{n}", "text": text, "reference": ref.round(6).tolist(),
                        "tflite": got.round(6).tolist(), "max_delta": delta, "same_top_label": bool(same)})
    (out / "validation.json").write_text(json.dumps({
        "compared_against": args.model_id,
        "engine": "ai_edge_litert Interpreter (LiteRT 2.2.0)",
        "quantization": args.recipe,
        "signatures": [f"seq_{n}" for n in seq_lens],
        "fixtures": total,
        "max_score_delta": max_delta,
        "top_label_agreement": f"{agree}/{total}",
        "per_fixture": per,
    }, indent=2) + "\n")
    manifest["quantization"] = args.recipe
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{args.recipe}: model.tflite {(out / 'model.tflite').stat().st_size} bytes; "
          f"max_score_delta={max_delta:.2e}; top-label agreement {agree}/{total}")


if __name__ == "__main__":
    main()

"""Export a HuggingFace single-sequence text classifier to LiteRT (.tflite) for a
tflite-server, and validate it against the original PyTorch model.

Mirrors the lemonade-sdk ONNX classifier export (export.py in the *-ONNX repos):
same fixtures, same manifest.json and validation.json outputs, so the two engines
are held to the same parity bar.

    python export_tflite.py <hf_model_id> <out_dir> [--seq-len 512]

Outputs into <out_dir>: model.tflite, the tokenizer files (incl. tokenizer.json),
config.json, manifest.json and validation.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import litert_torch
from ai_edge_litert.interpreter import Interpreter
from transformers import AutoModelForSequenceClassification, AutoTokenizer

FIXTURES = [
    "My name is John Smith and my SSN is 123-45-6789.",
    "URGENT: verify your account at http://secure-login.example to avoid suspension.",
    "Thanks for the notes from today's standup, talk tomorrow.",
]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class Logits(torch.nn.Module):
    """input_ids + attention_mask -> logits, with a fixed sequence length."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def encode(tok, text: str, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    enc = tok(text, truncation=True, max_length=seq_len, padding="max_length", return_tensors="np")
    return enc["input_ids"].astype(np.int32), enc["attention_mask"].astype(np.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("out")
    ap.add_argument("--seq-len", type=int, default=512)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id).eval()
    wrapped = Logits(model).eval()

    ids, mask = encode(tok, FIXTURES[0], args.seq_len)
    # Keyword samples name the signature inputs input_ids / attention_mask, so a
    # server can bind them by name the way ort-server binds ONNX inputs.
    edge = litert_torch.convert(wrapped, sample_kwargs={
        "input_ids": torch.from_numpy(ids), "attention_mask": torch.from_numpy(mask)})
    edge.export(str(out / "model.tflite"))

    tok.save_pretrained(out)
    model.config.save_pretrained(out)
    id2label = {str(k): v for k, v in model.config.id2label.items()}
    (out / "manifest.json").write_text(json.dumps({
        "task": "text-classification",
        "id2label": id2label,
        "score_normalization": "softmax",
        "token_aggregation": None,
        "max_sequence_length": args.seq_len,
    }, indent=2) + "\n")

    interp = Interpreter(model_path=str(out / "model.tflite"))
    runner = interp.get_signature_runner()
    input_names = sorted(runner.get_input_details().keys())
    assert input_names == ["attention_mask", "input_ids"], input_names
    max_delta = 0.0
    per_fixture = []
    for text in FIXTURES:
        ids, mask = encode(tok, text, args.seq_len)
        with torch.no_grad():
            ref = softmax(wrapped(torch.from_numpy(ids), torch.from_numpy(mask)).numpy())[0]
        feeds = {"input_ids": ids, "attention_mask": mask}
        got = softmax(next(iter(runner(**feeds).values())))[0]
        delta = float(np.abs(ref - got).max())
        max_delta = max(max_delta, delta)
        per_fixture.append({"text": text, "reference": ref.round(6).tolist(),
                            "tflite": got.round(6).tolist(), "max_delta": delta})
    (out / "validation.json").write_text(json.dumps({
        "compared_against": args.model_id,
        "engine": "ai_edge_litert Interpreter (LiteRT 2.2.0)",
        "sequence_length": args.seq_len,
        "fixtures": len(FIXTURES),
        "max_score_delta": max_delta,
        "per_fixture": per_fixture,
    }, indent=2) + "\n")
    print(f"model.tflite {(out / 'model.tflite').stat().st_size} bytes; max_score_delta={max_delta:.2e}")


if __name__ == "__main__":
    main()

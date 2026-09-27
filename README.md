# tflite-server

A LiteRT (TFLite) text-classification server for [Lemonade](https://github.com/lemonade-sdk/lemonade).
It is [ort-server](https://github.com/lemonade-sdk/ort-server) with the inference engine swapped: the
same `/classify` contract, manifest rules, tokenizer handling and model-family allowlist, with a LiteRT
`CompiledModel` (CPU, XNNPACK) in place of the ONNX Runtime session. A client, or Lemonade's
`/v1/classify`, gets the same request and response shape from either engine.

It targets small hosts. It builds against LiteRT's CMake static library and runs musl-native, for
example on a prplOS gateway, with no glibc bundle.

> Status: **experimental.**

## Model directory

```
model-dir/
  model.tflite           # signature inputs input_ids / attention_mask [/ token_type_ids], int32, [1, L]
  tokenizer.json         # the model's HuggingFace tokenizer
  config.json            # stock HF config (model_type is checked against the allowlist)
  tokenizer_config.json  # model_max_length
  manifest.json          # OPTIONAL explicit contract, same schema as ort-server
```

Each signature is the same graph at one fixed sequence length. The exporter writes `seq_64`,
`seq_128`, `seq_256` and `seq_512`, which share the weights. A request is tokenized, truncated to
`min(manifest max_length, longest L)`, and run on the smallest signature that holds it, padded with an
attention mask of zero.

`tools/export_tflite.py <hf_model_id> <out_dir> [--seq-lens 64,128,256,512]` produces such a directory.
It uses the same fixtures as the lemonade-sdk ONNX exports and writes a `validation.json` with the
maximum score difference against the original PyTorch model, for every signature.

## Run

```bash
tflite-server --model-path <model-dir> --port <n> [--threads N] [--weight-cache FILE] [--verbose]
```

- `--threads N`: number of CPU threads (default: one per hardware thread).
- `--weight-cache FILE`: XNNPACK packed-weight cache file. **Use it on small hosts.** Without it,
  XNNPACK packs the weights into anonymous memory once per signature. With it, one packed copy is
  mapped from disk and shared by all signatures. The file is written on first start (about 170 MB
  for DistilBERT) and must live on persistent, writable storage.

- `GET /health` returns `{"status":"ok","engine":"litert"}`.
- `POST /classify {"input": "...", "top_k": 2}` returns `{"labels": {"LABEL_1": 0.98, ...}}`.

It binds 127.0.0.1 only. Lemonade reaches it as a local subprocess.

## Measured

These figures are for the DistilBERT phishing classifier (`cybersectony/phishing-email-detection-distilbert_v2.4.1`,
fp32). The musl binary ran through the prplOS 5.1 loader, and both servers were pinned to the same
two cores of a Ryzen AI Max+ 395:

| | RSS | anonymous | 64-token request | 512-token request |
|---|---|---|---|---|
| ort-server 0.3.7 (ONNX) | 418 MiB | 394 MiB | 9-12 ms | |
| tflite-server, 4 signatures, no cache | 888 MiB | 693 MiB | 15-18 ms | |
| tflite-server, 4 signatures, `--weight-cache` | 230-239 MiB | 35-44 MiB | 14-17 ms | 91 ms |

Scores match ort-server to within 1.94e-6 on 8 texts, including one truncated at 512 tokens, with the
same top label on all of them (`tools/compare_with_ort.py`).

## License

Apache-2.0. See `LICENSE` and `NOTICE`; this is a derivative of ort-server.

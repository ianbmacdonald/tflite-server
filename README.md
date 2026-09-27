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

The export fixes the sequence length `L`. Requests are tokenized, truncated to
`min(manifest max_length, L)`, and padded to `L` with an attention mask of zero on the padding.

`tools/export_tflite.py <hf_model_id> <out_dir> [--seq-len 512]` produces such a directory. It uses
the same fixtures as the lemonade-sdk ONNX exports and writes a `validation.json` with the maximum score
difference against the original PyTorch model.

## Run

```bash
tflite-server --model-path <model-dir> --port <n> [--verbose]
```

- `GET /health` returns `{"status":"ok","engine":"litert"}`.
- `POST /classify {"input": "...", "top_k": 2}` returns `{"labels": {"LABEL_1": 0.98, ...}}`.

It binds 127.0.0.1 only. Lemonade reaches it as a local subprocess.

## License

Apache-2.0. See `LICENSE` and `NOTICE`; this is a derivative of ort-server.

# tflite-server

A LiteRT (TFLite) classification server for [Lemonade](https://github.com/lemonade-sdk/lemonade).
It serves text classifiers (`POST /classify`) and, since v0.2.0, image classifiers
(`POST /classify/image`). One process serves one model directory.

The text path is [ort-server](https://github.com/lemonade-sdk/ort-server) with the inference engine swapped: the
same `/classify` contract, manifest rules, tokenizer handling and model-family allowlist, with a LiteRT
`CompiledModel` (CPU, XNNPACK) in place of the ONNX Runtime session. A client, or Lemonade's
`/v1/classify`, gets the same request and response shape from either engine.

It targets small hosts. It builds against LiteRT's CMake static library and runs musl-native, for
example on a prplOS gateway, with no glibc bundle.

> Status: **experimental.**

## Text model directory

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

## Image model directory

A `manifest.json` with `"task": "image-classification"` selects the image path.

```
model-dir/
  model.tflite    # one float32 input [1, H, W, 3] (NHWC RGB), one float32 output with N scores
  labels.txt      # N lines; line i names output index i
  manifest.json   # required, see below
```

```json
{
  "task": "image-classification",
  "labels_file": "labels.txt",
  "score_normalization": "none",
  "preprocess": {
    "resize": "stretch",
    "mean": [127.5, 127.5, 127.5],
    "std": [127.5, 127.5, 127.5],
    "channel_order": "RGB",
    "layout": "auto"
  },
  "top_k_default": 5
}
```

The image is decoded to RGB, stretched to the model's H x W exactly as Pillow's
`Image.resize((W, H), Image.BILINEAR)` does it (antialiased triangle filter, fixed-point
weights, 8-bit rounding between passes), then each channel becomes `(x - mean) / std` with
`x` on the 0..255 scale. `mean` and `std` are a number or a 3-vector.

This build accepts only the values shown for `resize`, `score_normalization`,
`channel_order` and `layout` (`layout` may also be `NHWC`). Reserved values are refused
at load time by name, for example `manifest.preprocess.resize 'center_crop' not supported
in this build`: `center_crop` and `resize_shorter`, `NCHW`, `BGR`, and `softmax` /
`sigmoid`. Unknown keys are refused too. With `"score_normalization": "none"` the model
must already output probabilities; a score outside [0, 1] is a 500. Quantized-input
models are refused at load.

`tools/make_image_model_dir.py <out_dir> [--tgz FILE --labels FILE] [--validate-image FILE]`
builds the directory for TF MobileNetV2 1.0 224 fp32 (`mobilenet_v2_1.0_224.tflite`, sha256
`9f3bc29e38e90842a852bfed957dbf5e36f2d97a91dd17736b1e5c0aca8d3303`, Apache-2.0) with the TensorFlow `ImageNetLabels.txt` (1001 lines, index
0 is `background`, sha256 `536feacc519de3d418de26b2effb4d75694a8c4c0063e36499a46fa8061e2da9`). With local files it fetches nothing.

## Run

```bash
tflite-server --model-path <model-dir> --port <n> [--threads N] [--weight-cache FILE] [--verbose]
              [--max-image-bytes N] [--max-image-pixels N] [--max-concurrent-decodes 1..2]
              [--decode-budget-factor N] [--max-decode-bytes N] [--http-threads 2..16]
              [--oom-score-adj 0..1000]
```

- `--threads N`: number of CPU threads (default: one per hardware thread).
- `--weight-cache FILE`: XNNPACK packed-weight cache file. **Use it on small hosts.** Without it,
  XNNPACK packs the weights into anonymous memory once per signature. With it, one packed copy is
  mapped from disk and shared by all signatures. The file is written on first start (about 170 MB
  for DistilBERT) and must live on persistent, writable storage.
- `--max-image-bytes N` (default 16777216): a larger image is a 413. The HTTP body limit is
  set from it (N x 4/3 + 64 KiB, for base64, about 22.4 MB by default). A body whose
  Content-Length is over that limit is refused with 413 before it is read; an image part over N
  is discarded as it arrives and answered with 413.
- `--max-image-pixels N` (default 4000000): read from the image header before any decode.
- `--max-concurrent-decodes N` (default 1, clamped to 1..2): decode slots. A request waits up
  to 30 s for one, then gets 503. The decoded image is freed before inference.
- `--decode-budget-factor N` (default 16): the decoder may allocate at most
  N x width x height bytes, plus twice the input size for PNG, plus 4 MiB. The worst legitimate
  case, a progressive 4-component JPEG, peaks at about 15 bytes per pixel. A PNG whose zlib
  stream inflates past its header's size (a "zip bomb") fails with 400
  `image decode exceeded memory budget` instead of growing without bound. A realloc is charged
  as old + new size, because it may copy.
- `--max-decode-bytes N` (default 268435456, 16 MiB..4 GiB): a ceiling on that budget. At
  startup, a factor x pixel cap that exceeds it prints a warning.
- `--http-threads N` (default 4): HTTP worker threads. An image request reads its body only
  after it takes one of `max-concurrent-decodes + 1` request slots (waiting up to 30 s, then
  503), so extra threads hold waiting connections, not buffered bodies.
- `--oom-score-adj N` (default 0, unchanged): Linux only. Raising it needs no privilege and
  makes this process the OOM killer's first choice over the processes it shares a host with.

Recommended on a small gateway: `--max-image-pixels 4000000 --max-concurrent-decodes 1
--http-threads 4 --oom-score-adj 500`.

Endpoints:

- `GET /health` returns `{"status":"ok","engine":"litert","task":"...","version":"0.2.0"}`, where
  `task` is `image-classification`, `text-classification` or `token-classification`.
- `POST /classify {"input": "...", "top_k": 2}` returns `{"labels": {"LABEL_1": 0.98, ...}}`.
  On an image model it is a 400.
- `POST /classify/image` takes either multipart/form-data with exactly one file part named
  `image` or `file` (plus an optional `top_k` field), or JSON
  `{"image": "<base64>", "top_k": 3}`. The JSON `image` may also be a
  `data:image/jpeg;base64,` or `data:image/png;base64,` URL; `http(s)` URLs are refused.
  `top_k` is an integer from 1 to 1000000 (default: the manifest's `top_k_default`), clamped
  to the label count. On a text model it is a 400.

  ```bash
  curl -F image=@grace_hopper.jpg -F top_k=3 http://127.0.0.1:<n>/classify/image
  ```

  ```json
  {"predictions": [{"index": 653, "label": "military uniform", "score": 0.8054297},
                   {"index": 440, "label": "bearskin", "score": 0.0363316},
                   {"index": 668, "label": "mortarboard", "score": 0.0167730}],
   "labels": {"military uniform": 0.8054297, "bearskin": 0.0363316, "mortarboard": 0.0167730},
   "input": {"width": 512, "height": 600},
   "timings": {"decode_ms": 2.3, "preprocess_ms": 2.4, "inference_ms": 4.8}}
  ```

  `predictions` is sorted by score and is the authoritative result. `labels` is the
  `/classify` shape; ImageNet repeats some names (`crane`, `maillot`), so it can hold fewer
  entries than `predictions`. Errors are `{"error": "..."}` with 400 (not JPEG or PNG; corrupt
  or truncated; 12-bit or arithmetic-coded JPEG; over the pixel cap; decode budget exceeded;
  zero or several image parts; bad base64 or `top_k`), 413 (over `--max-image-bytes`) or 503
  (no decode slot within 30 s).

  JPEG (baseline and progressive, including Adobe CMYK/YCCK, which are converted to RGB) and
  PNG (all bit depths and color types; alpha is dropped) are supported. EXIF orientation is not
  applied, as in the Pillow reference.

It binds 127.0.0.1 only. Lemonade reaches it as a local subprocess.

Memory envelope for image requests, beyond the model: each admitted request holds at most
its body plus one decoded copy. That is about 2.7 x `--max-image-bytes` for base64 JSON (the
body, the image string, then the decoded bytes) and 1 x for multipart (the image part is
the only copy). With the defaults that is 2 slots x ~45 MB, plus one decode under a budget
of at most `--max-decode-bytes` (about 68 MB at the 4 MP default). JSON bodies must be one flat
object of at most 32 scalar keys; nesting is refused at the first nested bracket, so a
hostile body never becomes a DOM. `--verbose` prints the bound at startup.

## Measured

### Image classification

TF MobileNetV2 1.0 224 fp32, musl build under the prplOS 5.1 loader on a Ryzen AI Max+ 395:

- `grace_hopper.jpg`: top-1 653 `military uniform` 0.805430; the ai_edge_litert + Pillow
  reference is 0.803491. The whole difference is the JPEG decoder (stb vs libjpeg-turbo): the
  same image as PNG scores 0.803492 on both.
- `tools/compare_image_with_litert.py` over 25 images (flowers, photos, grayscale, CMYK,
  progressive, RGBA PNG, one 2300x1700 progressive JPEG): the same top-1 on all 25, maximum
  score difference 0.0061 over all 1001 classes.
- Pinned to two cores with `MemoryMax=512M`, 8 clients sending a 3.9 MP progressive JPEG and
  8 sending a 272 MiB PNG zip bomb, 50 requests each: every image answered correctly, every
  bomb got 400, `/health` kept answering, VmHWM 78 MiB. 100 sequential `grace_hopper`
  requests after that: inference 2.0 ms median, VmRSS flat at 46 MiB.
- Same limits, 12 clients at once sending bodies at the payload limit (a 16 MiB JPEG as
  multipart, the same as 21 MiB of base64 JSON, and 21 MiB of nested JSON arrays), 3 each:
  every image answered, every nested body got 400, VmHWM 119 MiB. The v0.2.0 draft, which
  buffered every body and parsed JSON into a DOM, was OOM-killed at `MemoryMax=2G` by the
  same run.

### Text classification

These figures are for the DistilBERT phishing classifier (`cybersectony/phishing-email-detection-distilbert_v2.4.1`,
fp32). The musl binary ran through the prplOS 5.1 loader, and both servers were pinned to the same
two cores of a Ryzen AI Max+ 395:

| | RSS | anonymous | 64-token request | 512-token request |
|---|---|---|---|---|
| ort-server 0.3.7 (ONNX) | 418 MiB | 394 MiB | 9-12 ms | |
| tflite-server, 4 signatures, no cache | 888 MiB | 693 MiB | 15-18 ms | |
| tflite-server, 4 signatures, `--weight-cache` | 230-239 MiB | 35-44 MiB | 14-17 ms | 91 ms |

**Quantization, measured on the same model.** `tools/quantize_tflite.py` applies an ai-edge-quantizer
recipe and re-validates against PyTorch:

| recipe | model.tflite | max score delta | same top label | tflite-server RSS / anonymous |
|---|---|---|---|---|
| fp32 (no quantization) | 270 MB | 1.2e-9 | 12/12 | 238 / 44 MiB (`--weight-cache`) |
| `weight_only_wi8_afp32` | 74 MB | 0.029 | 24/24 | 1225 / 972 MiB |
| `dynamic_wi8_afp32` (and per-channel) | 74 MB | 0.36 | 20/24 | not used: changes answers |

Weight-only int8 saves disk, but LiteRT expands the weights to fp32 in memory, so it costs RAM. On a
host where RAM is the constraint, serve the fp32 model with `--weight-cache`.

Scores match ort-server to within 1.94e-6 on 8 texts (1.21e-6 with v0.2.0), including one truncated at 512 tokens, with the
same top label on all of them (`tools/compare_with_ort.py`).

## Tests

The image decode, resize and manifest code builds without LiteRT:

```bash
cmake -S . -B build-tests -DTFLITE_SERVER_PREPROCESS_TESTS=ON && cmake --build build-tests
ctest --test-dir build-tests --output-on-failure
```

`tests/test_image_preprocess` checks decoding against Pillow (PNG exact, JPEG within a small
tolerance), byte-identical resizing against Pillow goldens, the 12-bit and arithmetic JPEG
rejections, the pixel cap, a generated PNG zip bomb, the decode budget on worst-case JPEGs,
and the whole pipeline on a thread with a 128 KiB stack (musl's default).
`tests/fuzz_image_decode <tests/data> 10000` is a mutation smoke; build it with
`-fsanitize=address,undefined`. `tests/test_flat_json` covers the request JSON parser and
checks that 21 MiB hostile bodies (nested arrays, a flat array, thousands of keys) add under
4 MiB of peak RSS. `tools/make_preprocess_goldens.py` regenerates `tests/data`.
Against a running server: `tests/e2e_image_server.py`, `tests/load_image_server.py` and
`tools/compare_image_with_litert.py`. The e2e script checks both 413 paths: a 20 MB image part
is under the derived body limit (about 22.4 MB) and gets the image-size 413 on a connection
that stays usable, and a 30 MB body is over it and gets 413 from its Content-Length.

## License

Apache-2.0. See `LICENSE` and `NOTICE`; the text path is a derivative of ort-server.
`third_party/stb/stb_image.h` is public domain or MIT.

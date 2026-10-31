# Patches for the musl (prplOS) build

Each directory holds `git format-patch` output for one third-party source, to apply with `git am`
(or `patch -p1`) on the pinned upstream commit below. `tools/feed/prepare-sources.sh` checks out
each commit and applies its patches on a local branch `musl`.

| Directory | Upstream | Base commit | Patches |
|---|---|---|---|
| `litert/` | https://github.com/google-ai-edge/LiteRT | `c54e491a1ea07abdb931233eadf2f432db656997` (tag `v2.3.2`, 2026-10-26) | `0001`-`0003` |
| `tokenizers-cpp/` | https://github.com/mlc-ai/tokenizers-cpp | `c586c52f93f7b060753bd2388eb96a105cb7374d` (main, 2026-10-31) | `0001` |

- `litert/0001-build-skip-the-vendor-NPU-dispatch-tree-when-LITERT_.patch`: with
  `-DLITERT_ENABLE_NPU=OFF`, `litert/vendors` is not configured, so the Qualcomm QAIRT, MediaTek
  NeuroPilot and Samsung AI LiteCore SDK headers are no longer downloaded and no vendor dispatch
  library is built. Without it, configuring v2.3.2 (and v2.2.0) downloads all three.
- `litert/0002-build-revert-the-v2.3.2-RuntimeProxy-ABI-check-which.patch`: `git revert` of
  `c54e491a1ea0`, the tag's own commit. Its `litert/cc/internal/litert_runtime_proxy.h` uses
  delegation-metrics types and `LiteRtRuntimeCApiStruct` members that exist only on LiteRT main
  (after `0dedf30fc`), so v2.3.2 as tagged does not compile (GCC 13: `'LiteRtDelegationMetricsConst'
  has not been declared`, 216 errors). The check matters only with a Google Play services runtime;
  this static CPU build always uses the builtin runtime. With 0002 the LiteRT tree equals v2.3.1
  plus `c368443e8` (a Kotlin JNI fix).
- `litert/0003-build-make-the-tensor-examples-optional-LITERT_TENSO.patch`: adds
  `LITERT_TENSOR_BUILD_EXAMPLES` (default ON, upstream behaviour); the build passes OFF. Without it,
  configuring LiteRT downloads a prebuilt `libLiteRtWebGpuAccelerator.so` from
  `storage.googleapis.com/litert/binaries/latest/` (unpinned) and fetches stb and sentencepiece,
  for `tensor/examples` that tflite-server does not use. Found by the offline build: with no
  network the configure fails there.
- `tokenizers-cpp/0001-build-add-rust-Cargo.lock-for-a-pinned-vendorable-sh.patch`: adds the
  `rust/Cargo.lock` the builds use, so `cargo build --locked` and `cargo vendor` work.

The musl port itself needs no source change in LiteRT. It needs these compiler flags
(`tools/feed/env.sh`):

- `-D_GNU_SOURCE`: XNNPACK uses `CLOCK_MONOTONIC`, which musl's headers hide under strict ISO.
- `-DFLATBUFFERS_LOCALE_INDEPENDENT=0`: musl has no `strtod_l` / `strtoull_l`.
- `-DCMAKE_SYSTEM_NAME=Linux` and `-DTFLITE_HOST_TOOLS_DIR=<dir with a host flatc 25.9.23>`: the
  cross build runs a host `flatc` instead of building one for the target.
- `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`: CMake 4 refuses neon2sse's old `cmake_minimum_required`.
- aarch64: `-march=armv8-a+crc -mtune=cortex-a53`, not `-mcpu=cortex-a53`, because a global
  `-mcpu` overrides the per-file `-march=armv8.2-a+...` of XNNPACK's and KleidiAI's
  runtime-dispatched kernels.

`-DLITERT_ENABLE_GPU=OFF` does not build on v2.3.2: `litert/runtime/open_cl_memory.cc` is compiled
anyway and needs `CL/cl.h`. The build keeps GPU on, which fetches only header repositories
(Khronos OpenCL, OpenGL, EGL, Vulkan; Apache-2.0 or MIT), and runs inference on the CPU.

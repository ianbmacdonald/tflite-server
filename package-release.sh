#!/bin/bash
# Package a tflite-server release tarball (run on the build host).
# Usage: package-release.sh <version> [x86_64|aarch64]
#   BLD             build tree (default build-prplos-<arch>)
#   LITERT_SRC      the LiteRT checkout (default $R/litert-v220)
#   LITERT_BUILD    the LiteRT tree the server was linked against (licence texts of what it vendors)
#   TOKENIZERS_SRC  the tokenizers-cpp checkout (default $R/tokenizers-cpp)
#   TOKDIR          the directory holding the target's libtokenizers_c.a
#   FLAVOUR         archive name infix, e.g. allops -> tflite-server-allops-musl-<arch>-<version>
#   OUT             output directory (default $R/bundle)
# Every component LiteRT's CMake fetched into LITERT_BUILD must have a licence text below, or the
# script fails: a new dependency cannot ship without one.
set -euo pipefail
VER=$1
ARCH=${2:-x86_64}
R=${R:-$HOME/build-litert}
SRC=$(cd "$(dirname "$0")" && pwd)
LS=${LITERT_SRC:-$R/litert-v220}
case "$ARCH" in
x86_64)
  SD=$R/prplos-5.1-staging; T=$SD/toolchain-x86_64_gcc-13.3.0_musl/bin/x86_64-openwrt-linux-musl-
  LB=${LITERT_BUILD:-$R/litert-v220-musl-x86_64}
  TOKDIR=${TOKDIR:-$R/tokenizers-cpp/rust/target/x86_64-unknown-linux-musl/release}
  HOST="prplOS 5.1 x86_64 (gcc 13.3.0 musl toolchain)" ;;
aarch64)
  SD=$R/prplos-5.1-staging-aarch64; T=$SD/toolchain-aarch64_cortex-a53_gcc-13.3.0_musl/bin/aarch64-openwrt-linux-musl-
  LB=${LITERT_BUILD:-$R/aarch64/litert-v220-musl-aarch64}
  TOKDIR=${TOKDIR:-$R/aarch64/tokenizers-target/aarch64-unknown-linux-musl/release}
  HOST="prplOS 5.1 aarch64 cortex-a53 (gcc 13.3.0 musl toolchain, -march=armv8-a+crc -mtune=cortex-a53)" ;;
*) echo "usage: $0 <version> [x86_64|aarch64]" >&2; exit 2 ;;
esac
export STAGING_DIR=$SD
BLD=${BLD:-$SRC/build-prplos-$ARCH}
TK=${TOKENIZERS_SRC:-$R/tokenizers-cpp}
NAME=tflite-server${FLAVOUR:+-$FLAVOUR}-musl-$ARCH-$VER
OUT=${OUT:-$R/bundle}
B=$OUT/$NAME
L=$B/licenses
need() {  # need <src> <dest-name>: licence texts are not optional
  [ -f "$1" ] || { echo "missing licence text $1" >&2; exit 1; }
  cp "$1" "$L/$2"
}

rm -rf "$B" && mkdir -p "$B/bin" "$L"
cp "$BLD/tflite-server.stripped" "$B/bin/tflite-server"
cp "$SRC/LICENSE" "$SRC/NOTICE" "$SRC/README.md" "$B/"
# LiteRT and what its CMake build vendors and compiles in. lic <component> <file> <dest-name>: the
# component name is the FetchContent name (LITERT_BUILD/_deps/<name>-subbuild) or, for XNNPACK's own
# downloads, the <name>-source directory.
covered=" "
lic() { need "$LB/$2" "$3"; covered="$covered$1 "; }
need "$LS/LICENSE" LiteRT-LICENSE
lic tensorflow tflite_build/tensorflow-src/LICENSE TensorFlow-LICENSE
lic xnnpack xnnpack/LICENSE XNNPACK-LICENSE
lic cpuinfo cpuinfo/LICENSE cpuinfo-LICENSE
lic pthreadpool-source pthreadpool-source/LICENSE pthreadpool-LICENSE
lic FP16-source FP16-source/LICENSE FP16-LICENSE
lic FXdiv-source FXdiv-source/LICENSE FXdiv-LICENSE
lic abseil-cpp abseil-cpp/LICENSE abseil-cpp-LICENSE
lic ruy ruy/LICENSE ruy-LICENSE
lic gemmlowp gemmlowp/LICENSE gemmlowp-LICENSE
lic farmhash farmhash/COPYING farmhash-COPYING
lic fft2d fft2d/readme2d.txt fft2d-readme2d.txt
lic flatbuffers flatbuffers/LICENSE flatbuffers-LICENSE
lic ml_dtypes ml_dtypes/LICENSE ml_dtypes-LICENSE
for f in MPL2 BSD MINPACK APACHE README; do lic eigen "eigen/COPYING.$f" "Eigen-COPYING.$f"; done
[ -d "$LB/safeint" ] && lic safeint safeint/LICENSE SafeInt-LICENSE
# Fetched by LiteRT v2.3.2; protobuf is not linked, the Khronos trees are headers.
[ -d "$LB/protobuf" ] && lic protobuf protobuf/LICENSE protobuf-LICENSE
[ -d "$LB/opencl_headers" ] && lic opencl_headers opencl_headers/LICENSE OpenCL-Headers-LICENSE
[ -d "$LB/vulkan_headers" ] && lic vulkan_headers vulkan_headers/LICENSE.txt Vulkan-Headers-LICENSE
[ -d "$LB/egl_headers" ] && lic egl_headers egl_headers/egl_headers_LICENSE.txt EGL-Registry-LICENSE
[ -d "$LB/opengl_headers" ] && lic opengl_headers opengl_headers/opengl_headers_LICENSE.txt OpenGL-Registry-LICENSE
if [ "$ARCH" = aarch64 ]; then
  lic kleidiai-source kleidiai-source/LICENSES/Apache-2.0.txt KleidiAI-LICENSE
  lic kleidiai-source kleidiai-source/LICENSES/BSD-3-Clause.txt KleidiAI-BSD-3-Clause.txt
else
  lic neon2sse neon2sse/LICENSE NEON_2_SSE-LICENSE
fi
fetched=$( { ls -d "$LB"/_deps/*-subbuild 2>/dev/null | sed 's|.*/||; s|-subbuild$||'
             ls -d "$LB"/*-source 2>/dev/null | sed 's|.*/||'; } | sort -u)
for c in $fetched; do
  case "$covered" in *" $c "*) ;; *) echo "fetched component $c has no licence text in $0" >&2; exit 1 ;; esac
done
need "$TK/LICENSE" tokenizers-cpp-LICENSE
need "$BLD/_deps/httplib-src/LICENSE" cpp-httplib-LICENSE
need "$BLD/_deps/json-src/LICENSE.MIT" nlohmann-json-LICENSE
need "$SRC/third_party/stb/LICENSE" stb-LICENSE
# The Rust crates of the tokenizer shim (incl. HuggingFace tokenizers, onig, Oniguruma), per target.
python3 "$SRC/tools/rust_licenses.py" "$TK/rust" "$ARCH-unknown-linux-musl" "$L" tflite-server \
  --cargo "$HOME/.cargo/bin/cargo"
RUSTC=$(strings "$TOKDIR/libtokenizers_c.a" | grep -o -m1 'rustc version [0-9][^ )]*' || true)
RD=$(ls -d "$HOME"/.rustup/toolchains/stable-*/share/doc/rust/licenses | head -1)
mkdir -p "$L/rust-std"
for f in Apache-2.0 MIT LLVM-exception Unicode-3.0; do need "$RD/$f.txt" "rust-std/$f.txt"; done
echo "The Rust standard library (${RUSTC:-rustc}) linked into the tokenizer shim: MIT OR Apache-2.0; its bundled libunwind: Apache-2.0 WITH LLVM-exception; Unicode data: Unicode-3.0." > "$L/rust-std/README"
OPS=$(grep -E '^TFLITE_SERVER_OPS:' "$BLD/CMakeCache.txt" | cut -d= -f2-)
{
  echo "tflite-server $VER, musl-native for $HOST."
  echo "Source: https://github.com/ianbmacdonald/tflite-server @ $(git -C "$SRC" rev-parse --short HEAD)"
  echo "LiteRT: $(git -C "$LS" rev-parse --short HEAD) ($(git -C "$LS" describe --tags)), static, $(basename "$LB")."
  if [ -n "$OPS" ] && [ "$OPS" != all ]; then
    echo "Operators: curated, $(basename "$OPS") ($(grep -v '^#' "$OPS" | grep -c .) TFLite builtin ops: the union over the six use cases"
    echo "  of the gateway study - text classification, sentence embeddings, image classification, object detection,"
    echo "  audio, time series; 17 models incl. DistilBERT and MobileNetV2 - plus the zero-shot and decision models"
    echo "  GLiClass-Edge v3.0, gliclass-modern-base-v3.0, Julia-1 and Laya):"
    grep -v '^#' "$OPS" | grep . | paste -sd' ' | fold -s -w 100 | sed 's/^/  /'
    echo "  A model with another op fails to load (Didn't find op for builtin opcode). Build with"
    echo "  -DTFLITE_SERVER_OPS=all for every builtin op."
  else
    echo "Operators: general build, every TFLite builtin op."
  fi
  echo "Linked with --gc-sections (TFLITE_SERVER_GC_SECTIONS=$(grep -E '^TFLITE_SERVER_GC_SECTIONS:' "$BLD/CMakeCache.txt" | cut -d= -f2-))."
  echo "Runtime libraries (from the prplOS image, not bundled):"
  "${T}readelf" -d "$B/bin/tflite-server" | grep -o 'NEEDED.*\[.*\]' | grep -o '\[[^]]*\]' | sed 's/^/  /'
} > "$B/BUILD-INFO.txt"
chmod -R go-w "$B"

cd "$OUT"
tar czf "$NAME.tar.gz" "$NAME"
sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256"
echo "$NAME.tar.gz $(stat -c %s "$NAME.tar.gz") bytes sha256 $(cut -c1-64 "$NAME.tar.gz.sha256")"
cat "$B/BUILD-INFO.txt"
ls "$L"

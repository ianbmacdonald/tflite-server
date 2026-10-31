#!/bin/bash
# offline-check.sh <arch>: configure LiteRT and tflite-server with NO network (unshare -rn), taking
# every source from the online build's fetched trees, then build both. Proves DEPENDENCIES.md's list.
# F: the work area (LiteRT/tokenizers checkouts, build/<arch> trees of the online build).
set -euo pipefail
ARCH=$1; F=${F:-/mnt/ai/lemonade/feed/tflite-server}; O=$F/build/$ARCH/litert; FS=$(dirname "$(readlink -f "$0")")
export ARCH JOBS=${JOBS:-12}
# LiteRT's CMake hard-codes ${CMAKE_BINARY_DIR}/<name> for several of these trees (opencl_headers,
# xnnpack/include, safeint, flatbuffers, FP16-source), so FETCHCONTENT_SOURCE_DIR_<NAME> elsewhere
# does not work: unpack each source at its default place in the new build directory instead (as a
# tarball would be, without .git), and tell FetchContent and XNNPACK not to fetch.
B=$F/build/$ARCH/offline
rm -rf "$B"; mkdir -p "$B/litert/tflite_build"
DIRS="abseil-cpp cpuinfo eigen farmhash fft2d flatbuffers gemmlowp ml_dtypes neon2sse protobuf ruy safeint xnnpack
      egl_headers opencl_headers opengl_headers vulkan_headers tflite_build/tensorflow-src
      FP16-source pthreadpool-source FXdiv-source kleidiai-source"
for d in $DIRS; do
  [ -d "$O/$d" ] || continue
  mkdir -p "$B/litert/$d"; tar -C "$O/$d" --exclude=.git -cf - . | tar -C "$B/litert/$d" -xf -
done
SRCS="-DFETCHCONTENT_FULLY_DISCONNECTED=ON -DPTHREADPOOL_SOURCE_DIR=$B/litert/pthreadpool-source"
SRCS="$SRCS -DFXDIV_SOURCE_DIR=$B/litert/FXdiv-source -DFP16_SOURCE_DIR=$B/litert/FP16-source"
[ -d "$B/litert/kleidiai-source" ] && SRCS="$SRCS -DKLEIDIAI_SOURCE_DIR=$B/litert/kleidiai-source"
echo "offline: $SRCS"
if unshare -rn curl -s -m 5 -o /dev/null https://github.com; then echo 'NETWORK STILL UP'; exit 1; fi; echo 'network: none (curl to github.com fails inside unshare -rn)'
unshare -rn env B="$B" LITERT_CMAKE_EXTRA="$SRCS" "$FS/build-litert.sh"
grep -a -i -E "download|cloning|fetch" "$B/litert/cfg.log" | grep -v -i "define .* to avoid it" || true
# Rust crates: vendor online (the download step), then build the shim with no network.
( cd "$F/tokenizers-cpp/rust" && "$HOME/.cargo/bin/cargo" vendor --locked "$F/cargo-vendor" > /dev/null )
unshare -rn env B="$B" CARGO_NET_OFFLINE=true \
  CARGO_EXTRA="--config source.crates-io.replace-with='vendored' --config source.vendored.directory='$F/cargo-vendor'" \
  "$FS/build-tokenizers.sh"
TB=$F/build/$ARCH/tflite-server/_deps
unshare -rn env B="$B" BLD="$B/tflite-server" \
  TFLITE_SERVER_CMAKE_EXTRA="-DFETCHCONTENT_FULLY_DISCONNECTED=ON -DFETCHCONTENT_SOURCE_DIR_HTTPLIB=$TB/httplib-src -DFETCHCONTENT_SOURCE_DIR_JSON=$TB/json-src" \
  "$FS/build-tflite-server.sh"
echo "offline-check $ARCH done"

#!/bin/bash
# Build tflite-server musl-native for prplOS 5.1 x86_64 on the ai4 build host layout.
# LITERT_BUILD: a LiteRT tree compiled with -ffunction-sections -fdata-sections lets --gc-sections drop
# the unused kernels (the releases use one). TFLITE_SERVER_OPS=all: the general build.
set -euo pipefail
R=${R:-$HOME/build-litert}
SD=$R/prplos-5.1-staging
T=$SD/toolchain-x86_64_gcc-13.3.0_musl/bin
export STAGING_DIR=$SD
SRC=$(cd "$(dirname "$0")" && pwd)
BLD=$SRC/build-prplos-x86_64
rm -rf "$BLD"
cmake -S "$SRC" -B "$BLD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=x86_64 \
  -DCMAKE_C_COMPILER=$T/x86_64-openwrt-linux-musl-gcc -DCMAKE_CXX_COMPILER=$T/x86_64-openwrt-linux-musl-g++ \
  -DCMAKE_CXX_FLAGS="-D_GNU_SOURCE -DFLATBUFFERS_LOCALE_INDEPENDENT=0" \
  -DCMAKE_SYSROOT=$SD/target-x86_64_musl \
  -DLITERT_SRC=$R/litert-v220 -DLITERT_BUILD=${LITERT_BUILD:-$R/litert-v220-musl-x86_64} \
  -DTFLITE_SERVER_OPS="${TFLITE_SERVER_OPS:-$SRC/ops/usecase-all.txt}" \
  -DTOKENIZERS_CPP_SRC=$R/tokenizers-cpp \
  -DTOKENIZERS_C_LIB=$R/tokenizers-cpp/rust/target/x86_64-unknown-linux-musl/release/libtokenizers_c.a
nice -n 19 cmake --build "$BLD" -j"${JOBS:-8}"
"$T/x86_64-openwrt-linux-musl-strip" -o "$BLD/tflite-server.stripped" "$BLD/tflite-server"
file "$BLD/tflite-server"
ls -la "$BLD/tflite-server" "$BLD/tflite-server.stripped" | awk '{print $5, $9}'
"$T/x86_64-openwrt-linux-musl-readelf" -d "$BLD/tflite-server" | grep NEEDED

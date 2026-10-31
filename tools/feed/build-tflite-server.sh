#!/bin/bash
# tflite-server for prplOS 5.1 musl against the trees build-litert.sh and build-tokenizers.sh wrote.
# ARCH=x86_64|aarch64 build-tflite-server.sh. Output: $B/tflite-server/tflite-server{,.stripped}.
set -euo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
SRC=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
BLD=${BLD:-$B/tflite-server}
rm -rf "$BLD"
cmake -S "$SRC" -B "$BLD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=$ARCH \
  -DCMAKE_C_COMPILER=$T/$X-gcc -DCMAKE_CXX_COMPILER=$T/$X-g++ \
  -DCMAKE_C_FLAGS="$CPUFLAGS" \
  -DCMAKE_CXX_FLAGS="$CPUFLAGS $MUSL_FLAGS" \
  -DCMAKE_SYSROOT=$S \
  -DLITERT_SRC="$LITERT_SRC" -DLITERT_BUILD="$B/litert" \
  -DTFLITE_SERVER_OPS="${TFLITE_SERVER_OPS:-$SRC/ops/usecase-all.txt}" \
  -DTOKENIZERS_CPP_SRC="$TOKENIZERS_SRC" \
  -DTOKENIZERS_C_LIB="$B/tokenizers-target/$RUST_TARGET/release/libtokenizers_c.a" \
  ${TFLITE_SERVER_CMAKE_EXTRA:-}
cmake --build "$BLD" -j"$JOBS" --target tflite-server
if grep -q "^TFLITE_SERVER_BUILD_TFLITE_RUN:BOOL=ON" "$BLD/CMakeCache.txt"; then
  cmake --build "$BLD" -j"$JOBS" --target tflite-run
  "$T/$X-strip" -o "$BLD/tflite-run.stripped" "$BLD/tflite-run"
fi
"$T/$X-strip" -o "$BLD/tflite-server.stripped" "$BLD/tflite-server"
file "$BLD/tflite-server.stripped"
ls -la "$BLD/tflite-server" "$BLD/tflite-server.stripped"
"$T/$X-readelf" -d "$BLD/tflite-server" | grep NEEDED

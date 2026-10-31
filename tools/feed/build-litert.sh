#!/bin/bash
# LiteRT (static libraries, CMake) for prplOS 5.1 musl. ARCH=x86_64|aarch64 build-litert.sh
# LITERT_SRC must be a checkout of LITERT_COMMIT with patches/musl/litert/*.patch applied
# (prepare-sources.sh does that). Compiled with -ffunction-sections -fdata-sections so the
# tflite-server link can drop the kernels it does not register. NPU off (no vendor SDK), tensor
# examples off (no prebuilt accelerator download). GPU stays on: litert_runtime
# does not compile without the OpenCL headers (measured on v2.3.2); inference runs on the CPU.
set -euo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
git -C "$LITERT_SRC" merge-base --is-ancestor "$LITERT_COMMIT" HEAD ||
  { echo "ABORT: $LITERT_SRC is not $LITERT_COMMIT (+ patches)"; exit 1; }
[ -x "$HOST_FLATC_DIR/flatc" ] || { echo "ABORT: no host flatc in $HOST_FLATC_DIR"; exit 1; }
echo "start $(date -u +%FT%TZ) arch=$ARCH litert=$(git -C "$LITERT_SRC" describe --tags --always) host flatc: $("$HOST_FLATC_DIR/flatc" --version)"
L=$B/litert
mkdir -p "$L"
cmake -S "$LITERT_SRC/litert" -B "$L" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=$ARCH \
  -DCMAKE_C_COMPILER=$T/$X-gcc -DCMAKE_CXX_COMPILER=$T/$X-g++ \
  -DCMAKE_C_FLAGS="$CPUFLAGS $MUSL_FLAGS -ffunction-sections -fdata-sections" \
  -DCMAKE_CXX_FLAGS="$CPUFLAGS $MUSL_FLAGS -ffunction-sections -fdata-sections" \
  -DCMAKE_SYSROOT=$S -DCMAKE_FIND_ROOT_PATH=$S \
  -DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
  -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DTFLITE_HOST_TOOLS_DIR="$HOST_FLATC_DIR" \
  -DLITERT_ENABLE_NPU=OFF -DLITERT_TENSOR_BUILD_EXAMPLES=OFF \
  ${LITERT_CMAKE_EXTRA:-} > "$L/cfg.log" 2>&1
if grep -q -a "fetching FlatBuffers to build flatc" "$L/cfg.log"; then
  echo "ABORT: flatc would be built as a target binary"; exit 1
fi
# CPU only: no vendor NPU SDK may be fetched (patches/musl/litert/0001).
for d in qnn_headers neuro_pilot-src litecore_headers; do
  [ ! -e "$L/_deps/$d" ] || { echo "ABORT: a vendor NPU SDK was fetched ($d)"; exit 1; }
done
if grep -a -E "QAIRT|qnn_headers|NeuroPilot|LiteCore" "$L/cfg.log"; then
  echo "ABORT: the configure log names a vendor NPU SDK"; exit 1
fi
# No prebuilt binary either (patches/musl/litert/0003).
if grep -a "Downloading LiteRT GPU accelerator" "$L/cfg.log"; then
  echo "ABORT: a prebuilt accelerator library was downloaded"; exit 1
fi
ninja -C "$L" -j"$JOBS" litert_c_api run_model > "$L/build.log" 2>&1
find "$L" \( -name liblitert_c_api.a -o -name run_model \) -type f -exec ls -la {} \;
echo "done $(date -u +%FT%TZ)"

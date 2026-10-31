#!/bin/bash
# tokenizers-cpp's Rust C shim (libtokenizers_c.a) for prplOS 5.1 musl. ARCH=x86_64|aarch64.
# TOKENIZERS_SRC: a checkout of TOKENIZERS_COMMIT with patches/musl/tokenizers-cpp/*.patch applied
# (the Cargo.lock). Offline: cargo vendor once, then CARGO_NET_OFFLINE=true and CARGO_EXTRA="--config
# source.crates-io.replace-with='v' --config source.v.directory='<vendor dir>'" (DEPENDENCIES.md).
set -euo pipefail
. "$(dirname "$(readlink -f "$0")")/env.sh"
git -C "$TOKENIZERS_SRC" merge-base --is-ancestor "$TOKENIZERS_COMMIT" HEAD ||
  { echo "ABORT: $TOKENIZERS_SRC is not $TOKENIZERS_COMMIT (+ patches)"; exit 1; }
[ -f "$TOKENIZERS_SRC/rust/Cargo.lock" ] || { echo "ABORT: no rust/Cargo.lock"; exit 1; }
CARGO=${CARGO:-$HOME/.cargo/bin/cargo}
echo "tokenizers-cpp $(git -C "$TOKENIZERS_SRC" rev-parse --short HEAD) $("$CARGO" --version)"
export CARGO_TARGET_DIR=$B/tokenizers-target
U=$(echo "$RUST_TARGET" | tr a-z- A-Z_)
export CC_${RUST_TARGET//-/_}=$T/$X-gcc
export CXX_${RUST_TARGET//-/_}=$T/$X-g++
export AR_${RUST_TARGET//-/_}=$T/$X-gcc-ar
export CFLAGS_${RUST_TARGET//-/_}="$CPUFLAGS"
export CARGO_TARGET_${U}_LINKER=$T/$X-gcc
[ -n "$RUST_CPU" ] && export CARGO_TARGET_${U}_RUSTFLAGS="$RUST_CPU"
cd "$TOKENIZERS_SRC/rust"
"$CARGO" build --release --locked --target "$RUST_TARGET" -j"$JOBS" ${CARGO_EXTRA:-}
ls -la "$CARGO_TARGET_DIR/$RUST_TARGET/release/libtokenizers_c.a"

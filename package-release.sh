#!/bin/bash
# Package a tflite-server release tarball from build-prplos-x86_64 (run on the build host).
# Usage: package-release.sh <version>   (OUT=<dir> overrides the output directory, default $R/bundle)
set -euo pipefail
VER=$1
R=${R:-$HOME/build-litert}
SRC=$(cd "$(dirname "$0")" && pwd)
BLD=$SRC/build-prplos-x86_64
NAME=tflite-server-musl-x86_64-$VER
OUT=${OUT:-$R/bundle}
B=$OUT/$NAME
T=$R/prplos-5.1-staging/toolchain-x86_64_gcc-13.3.0_musl/bin
export STAGING_DIR=$R/prplos-5.1-staging

rm -rf "$B" && mkdir -p "$B/bin" "$B/licenses"
cp "$BLD/tflite-server.stripped" "$B/bin/tflite-server"
cp "$SRC/LICENSE" "$SRC/NOTICE" "$SRC/README.md" "$B/"
cp "$R/litert-v220/LICENSE" "$B/licenses/LiteRT-LICENSE"
cp "$R/tokenizers-cpp/LICENSE" "$B/licenses/tokenizers-cpp-LICENSE"
cp "$BLD/_deps/httplib-src/LICENSE" "$B/licenses/cpp-httplib-LICENSE"
cp "$BLD/_deps/json-src/LICENSE.MIT" "$B/licenses/nlohmann-json-LICENSE"
cp "$SRC/third_party/stb/LICENSE" "$B/licenses/stb-LICENSE"
ONIG=$(find "$HOME/.cargo/registry/src" -maxdepth 4 -path '*onig_sys*' -name COPYING | head -1)
[ -n "$ONIG" ] && cp "$ONIG" "$B/licenses/oniguruma-COPYING"
( cd "$R/tokenizers-cpp/rust" && "$HOME/.cargo/bin/cargo" metadata --format-version 1 \
    --filter-platform x86_64-unknown-linux-musl 2>/dev/null ) | python3 -c '
import json, sys
d = json.load(sys.stdin)
res = {n["id"] for n in d["resolve"]["nodes"]}
print("Rust crates statically linked into bin/tflite-server (via the HuggingFace tokenizers C shim):")
for name, ver, lic in sorted({(p["name"], p["version"], p.get("license") or "see crate") for p in d["packages"] if p["id"] in res}):
    print(f"  {name} {ver}  {lic}")
' > "$B/licenses/rust-crates.txt"
{
  echo "tflite-server $VER, musl-native for prplOS 5.1 x86_64 (gcc 13.3.0 musl toolchain)."
  echo "Source: https://github.com/ianbmacdonald/tflite-server @ $(git -C "$SRC" rev-parse --short HEAD)"
  echo "LiteRT: $(git -C "$R/litert-v220" rev-parse --short HEAD) (v2.2.0), static."
  echo "Runtime libraries (from the prplOS image, not bundled):"
  "$T/x86_64-openwrt-linux-musl-readelf" -d "$B/bin/tflite-server" | grep -o 'NEEDED.*\[.*\]' | grep -o '\[[^]]*\]' | sed 's/^/  /'
} > "$B/BUILD-INFO.txt"

cd "$OUT"
tar czf "$NAME.tar.gz" "$NAME"
sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256"
echo "$NAME.tar.gz $(stat -c %s "$NAME.tar.gz") bytes sha256 $(cut -c1-64 "$NAME.tar.gz.sha256")"
cat "$B/BUILD-INFO.txt"
ls "$B/licenses"

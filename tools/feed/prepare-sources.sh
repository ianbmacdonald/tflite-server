#!/bin/bash
# Check out LiteRT and tokenizers-cpp at their pinned commits and apply patches/musl/<name>/*.patch
# on a local branch "musl". Needs network only for the clones; with existing checkouts it fetches nothing.
set -euo pipefail
ARCH=${ARCH:-x86_64}
. "$(dirname "$(readlink -f "$0")")/env.sh"
P=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)/patches/musl
prep() {  # dir url commit patchdir
    [ -d "$1/.git" ] || git clone -q "$2" "$1"
    git -C "$1" checkout -q -B musl "$3"
    if compgen -G "$4/*.patch" > /dev/null; then
        git -C "$1" -c user.name=feed -c user.email=feed@localhost am -q --committer-date-is-author-date "$4"/*.patch
    fi
    echo "$1: $(git -C "$1" rev-parse HEAD) on $3"
}
prep "$LITERT_SRC" https://github.com/google-ai-edge/LiteRT.git "$LITERT_COMMIT" "$P/litert"
prep "$TOKENIZERS_SRC" https://github.com/mlc-ai/tokenizers-cpp.git "$TOKENIZERS_COMMIT" "$P/tokenizers-cpp"

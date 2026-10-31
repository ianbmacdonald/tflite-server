# Shared settings for the musl feed builds (sourced). ARCH=x86_64|aarch64.
# Paths default to the ai4 layout; override any of them from the environment.
: "${ARCH:?ARCH=x86_64|aarch64}"
F=${F:-/mnt/ai/lemonade/feed/tflite-server}
R=${R:-/mnt/ai/lemonade/build-litert}
LITERT_SRC=${LITERT_SRC:-$F/litert-src}
LITERT_COMMIT=c54e491a1ea07abdb931233eadf2f432db656997   # v2.3.2
TOKENIZERS_SRC=${TOKENIZERS_SRC:-$F/tokenizers-cpp}
TOKENIZERS_COMMIT=c586c52f93f7b060753bd2388eb96a105cb7374d
HOST_FLATC_DIR=${HOST_FLATC_DIR:-$R/host-flatc}         # a HOST flatc 25.9.23
B=${B:-$F/build/$ARCH}
JOBS=${JOBS:-12}
case "$ARCH" in
  x86_64)
    SD=${SD:-$R/prplos-5.1-staging}
    T=$SD/toolchain-x86_64_gcc-13.3.0_musl/bin
    S=$SD/target-x86_64_musl
    X=x86_64-openwrt-linux-musl
    CPUFLAGS=""
    RUST_TARGET=x86_64-unknown-linux-musl
    RUST_CPU=""
    ;;
  aarch64)
    SD=${SD:-$R/prplos-5.1-staging-aarch64}
    T=$SD/toolchain-aarch64_cortex-a53_gcc-13.3.0_musl/bin
    S=$SD/target-aarch64_cortex-a53_musl
    X=aarch64-openwrt-linux-musl
    # -march/-mtune, not -mcpu: a global -mcpu overrides the per-file -march=armv8.2-a+...
    # that XNNPACK and KleidiAI put on their runtime-dispatched kernels.
    CPUFLAGS="-march=armv8-a+crc -mtune=cortex-a53"
    RUST_TARGET=aarch64-unknown-linux-musl
    RUST_CPU="-C target-cpu=cortex-a53"
    ;;
  *) echo "ARCH must be x86_64 or aarch64" >&2; exit 2 ;;
esac
export STAGING_DIR=$SD
# musl: XNNPACK needs CLOCK_MONOTONIC (_GNU_SOURCE); flatbuffers must not use the *_l locale functions.
MUSL_FLAGS="-D_GNU_SOURCE -DFLATBUFFERS_LOCALE_INDEPENDENT=0"

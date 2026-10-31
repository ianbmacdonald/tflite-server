# Dependencies

Every external source the musl build of tflite-server fetches, as the LiteRT v2.3.2 CMake actually
resolved them on ai4 (read from the fetched trees with `git rev-parse HEAD` and from each
`_deps/*-subbuild/CMakeLists.txt`, not from documentation). The build scripts are `tools/feed/`.
`patches/musl/README.md` names the patches and their base commits.

Default build: CPU only. The vendor NPU SDKs (Qualcomm QAIRT, MediaTek NeuroPilot, Samsung AI
LiteCore), which unpatched LiteRT downloads at configure time, are not fetched
(`patches/musl/litert/0001` with `-DLITERT_ENABLE_NPU=OFF`; `build-litert.sh` aborts if one appears).

## Top-level sources

| Name | URL | Pinned commit / hash | Licence |
|---|---|---|---|
| LiteRT | https://github.com/google-ai-edge/LiteRT | `c54e491a1ea07abdb931233eadf2f432db656997` (v2.3.2) + `patches/musl/litert` | Apache-2.0 |
| tokenizers-cpp | https://github.com/mlc-ai/tokenizers-cpp | `c586c52f93f7b060753bd2388eb96a105cb7374d` + `patches/musl/tokenizers-cpp` | Apache-2.0 |
| cpp-httplib | https://github.com/yhirose/cpp-httplib | `a7bc00e3307fecdb4d67545e93be7b88cfb1e186` (v0.18.3) | MIT |
| nlohmann/json | https://github.com/nlohmann/json | `9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03` (v3.11.3) | MIT |
| stb_image | vendored in `third_party/stb` | (in this repo) | MIT or public domain |
| host flatc | https://github.com/google/flatbuffers | `187240970746d00bbd26b0f5873ed54d2477f9f3` (v25.9.23), built for the HOST | Apache-2.0 |

tokenizers-cpp's git submodules (msgpack, sentencepiece) are not needed: the build uses only its
`include/tokenizers_c.h` and the Rust shim in `rust/`.

## Fetched by the LiteRT v2.3.2 CMake configure

`B` is the LiteRT build directory. "Offline" is where the unpacked source goes for a build with no
network: LiteRT's CMake declares each FetchContent source at a fixed `SOURCE_DIR` under `B` and also
hard-codes several of those paths as include directories (`opencl_headers`, `xnnpack/include`,
`safeint`, `flatbuffers`, `FP16-source`), so `FETCHCONTENT_SOURCE_DIR_<NAME>` pointing elsewhere
configures but does not compile (measured). Unpack each source there (no `.git` needed) and
configure with `-DFETCHCONTENT_FULLY_DISCONNECTED=ON` plus the `-D` variables named in the column,
which XNNPACK and pthreadpool read instead of downloading.

| Name | URL | Commit / hash as resolved | Licence | Offline |
|---|---|---|---|---|
| abseil-cpp | https://github.com/abseil/abseil-cpp | `5650e9cf76d3be4318d5fa3af38ee483ddfd5e4a` (tag 20260526.0) | Apache-2.0 | `B/abseil-cpp` |
| TensorFlow (TFLite sources, `B/tflite_build/tensorflow-src`) | https://github.com/tensorflow/tensorflow | `a344c313f75bbe28d00abe63f5d49cd6f233b6d0` (tag v2.21.0-rc0) | Apache-2.0 | `B/tflite_build/tensorflow-src` |
| XNNPACK | https://github.com/google/XNNPACK | `d89ef6669a14db203b3b7935b1b3862cb63fb6df` | BSD-3-Clause | `B/xnnpack` |
| pthreadpool (XNNPACK) | https://github.com/google/pthreadpool/archive/15a6644ba1c45f1acc16ac1e883efc3e56c6bed2.zip | sha256 `9b9fb1179b71021c0c048504eab636424c58e9c8374404754ece6d7cd90f26d4` | BSD-2-Clause | `B/pthreadpool-source` + `-DPTHREADPOOL_SOURCE_DIR` |
| FXdiv (pthreadpool) | https://github.com/Maratyszcza/FXdiv | `63058eff77e11aa15bf531df5dd34395ec3017c8` (pthreadpool asks for `master`; the feed scripts pin this commit) | MIT | `B/FXdiv-source` + `-DFXDIV_SOURCE_DIR` |
| FP16 | https://github.com/Maratyszcza/FP16/archive/0a92994d729ff76a58f692d3028ca1b64b145d91.zip | sha256 `e66e65515fa09927b348d3d584c68be4215cfe664100d01c9dbc7655a5716d70` | MIT | `B/FP16-source` + `-DFP16_SOURCE_DIR` |
| cpuinfo | https://github.com/pytorch/cpuinfo | `8ce83db858065145192c97af90cb668ad72a12e9` | BSD-2-Clause | `B/cpuinfo` |
| eigen | https://gitlab.com/libeigen/eigen | `ea13a98decd497a8c5588fb5de71b57bcf10d864` | MPL-2.0 (built with `EIGEN_MPL2_ONLY`) | `B/eigen` |
| farmhash | https://github.com/google/farmhash | `0d859a811870d10f53a594927d0d0b97573ad06d` | MIT | `B/farmhash` |
| fft2d (Ooura FFT) | https://storage.googleapis.com/mirror.tensorflow.org/github.com/petewarden/OouraFFT/archive/v1.0.tar.gz | sha256 `5f4dabc2ae21e1f537425d58a49cdca1c49ea11db0d6271e2a4b27e9697548eb` | permissive (Ooura's notice, `readme2d.txt`) | `B/fft2d` |
| flatbuffers (library) | https://github.com/google/flatbuffers | `187240970746d00bbd26b0f5873ed54d2477f9f3` (v25.9.23) | Apache-2.0 | `B/flatbuffers` |
| gemmlowp | https://github.com/google/gemmlowp | `16e8662c34917be0065110bfcd9cc27d30f52fdf` | Apache-2.0 | `B/gemmlowp` |
| ruy | https://github.com/google/ruy | `3286a34cc8de6149ac6844107dfdffac91531e72` | Apache-2.0 | `B/ruy` |
| ml_dtypes | https://github.com/jax-ml/ml_dtypes | `00d98cd92ade342fef589c0470379abb27baebe9` | Apache-2.0 | `B/ml_dtypes` |
| protobuf | https://github.com/protocolbuffers/protobuf | `90b73ac3f0b10320315c2ca0d03a5a9b095d2f66` | BSD-3-Clause | `B/protobuf` |
| SafeInt | https://github.com/dcleblanc/SafeInt | `72f0745e26091af6d1186fd91426377f16688ba7` | MIT | `B/safeint` |
| NEON_2_SSE (x86_64 only) | https://storage.googleapis.com/mirror.tensorflow.org/github.com/intel/ARM_NEON_2_x86_SSE/archive/a15b489e1222b2087007546b4912e21293ea86ff.tar.gz | sha256 `019fbc7ec25860070a1d90e12686fc160cfb33e22aa063c80f52b363f1361e9d` | BSD-2-Clause | `B/neon2sse` |
| EGL-Registry (headers) | https://github.com/KhronosGroup/EGL-Registry | `649981109e263b737e7735933c90626c29a306f2` | Apache-2.0 / MIT | `B/egl_headers` |
| OpenCL-Headers | https://github.com/KhronosGroup/OpenCL-Headers | `dcd5bede6859d26833cd85f0d6bbcee7382dc9b3` | Apache-2.0 | `B/opencl_headers` |
| OpenGL-Registry (headers) | https://github.com/KhronosGroup/OpenGL-Registry | `0cb0880d91581d34f96899c86fc1bf35627b4b81` | Apache-2.0 / MIT | `B/opengl_headers` |
| Vulkan-Headers | https://github.com/KhronosGroup/Vulkan-Headers | `32c07c0c5334aea069e518206d75e002ccd85389` | Apache-2.0 | `B/vulkan_headers` |
| KleidiAI (XNNPACK, aarch64 only) | https://gitlab.arm.com/kleidi/kleidiai/-/archive/4bc7bd457930de119f8b826ecfdebfbd9c08cf8c/kleidiai-4bc7bd457930de119f8b826ecfdebfbd9c08cf8c.zip | sha256 `652ee046ebf99bee5ed78aaf8961ea1df6ac09b6ebbd3a33106d9ce255efba69` | Apache-2.0 and BSD-3-Clause | `B/kleidiai-source` + `-DKLEIDIAI_SOURCE_DIR` |

Both architectures resolve the same commits (checked tree by tree on ai4); NEON_2_SSE is x86_64 only and
KleidiAI aarch64 only.

The four Khronos header repositories come with LiteRT's GPU option, which stays on because
v2.3.2's `litert_runtime` does not compile without `CL/cl.h` (`-DLITERT_ENABLE_GPU=OFF` fails at
`litert/runtime/open_cl_memory.cc`). They are headers only; nothing runs on a GPU.
Unpatched, LiteRT's configure also fetches stb (`c0c982601f40183e74d84a61237e968dca08380e`) and
sentencepiece v0.2.0 and downloads a prebuilt `libLiteRtWebGpuAccelerator.so` from
`https://storage.googleapis.com/litert/binaries/latest/` (no pin, no hash) for `tensor/examples`;
`patches/musl/litert/0003` with `-DLITERT_TENSOR_BUILD_EXAMPLES=OFF` drops all three.

## Rust crates (tokenizers C shim)

`patches/musl/tokenizers-cpp/0001` adds `rust/Cargo.lock`; with it, `cargo vendor` in
`tokenizers-cpp/rust` writes every crate below into `vendor/` plus the `[source]` stanza for
`.cargo/config.toml`, and `CARGO_NET_OFFLINE=true cargo build --release --locked --target
<arch>-unknown-linux-musl` then builds with no network. Checksums are the crates.io sha256 from
the lock file. `onig_sys` bundles the Oniguruma C sources (BSD-2-Clause) and compiles them with the
target C compiler. The Rust standard library (MIT OR Apache-2.0) comes with the rustup target
(`rustup target add <arch>-unknown-linux-musl`; built with rustc 1.98.1).

| Crate | Version | Licence | sha256 |
|---|---|---|---|
| ahash | 0.8.12 | MIT OR Apache-2.0 | `5a15f179cd60c4584b8a8c596927aadc462e27f2ca70c04e0071964a73ba7a75` |
| aho-corasick | 1.1.5 | Unlicense OR MIT | `c982642fa9e8606056828ee9a8505737230110bb1099153c79efe865c59d12ba` |
| base64 | 0.13.1 | MIT/Apache-2.0 | `9e1b586273c5702936fe7b7d6896644d8be71e6314cfe09d3167c95f712589e8` |
| bitflags | 2.13.2 | MIT OR Apache-2.0 | `3ded4057c258ba199e2d26386d3af3780957ecaee6c4ef4041c6b4b8b97c0b06` |
| castaway | 0.2.4 | MIT | `dec551ab6e7578819132c713a93c022a05d60159dc86e7a7050223577484c55a` |
| cc | 1.5.1 | MIT OR Apache-2.0 | `f360145194ee8e21db5ee7f3fcd4fe52210864c75c985dae33218202c8bbe040` |
| cfg-if | 1.0.5 | MIT OR Apache-2.0 | `4e7648175b45a9a48536d676f68d918270699102aa8dab5496df06904c914600` |
| compact_str | 0.9.1 | MIT | `9dfdd1c2274d9aa354115b09dc9a901d6c5576818cdf70d14cae2bdb47df00ab` |
| crossbeam-deque | 0.8.8 | MIT OR Apache-2.0 | `622f3fc73690be383c7214310406f28a90e6edeadc3cea882f9d71e495b9711a` |
| crossbeam-epoch | 0.9.21 | MIT OR Apache-2.0 | `dc74980687109a3b14c72fd458107bf0baa1da1a1a805e178d15501ba9b86d9d` |
| crossbeam-utils | 0.8.23 | MIT OR Apache-2.0 | `a31eee39dddec8330830986fcd7625edb5a24ec90ea038215273bbc3adb08ac6` |
| darling | 0.20.11 | MIT | `fc7f46116c46ff9ab3eb1597a45688b6715c6e628b5c133e288e709a29bcb4ee` |
| darling_core | 0.20.11 | MIT | `0d00b9596d185e565c2207a0b01f8bd1a135483d02d9b7b0a54b11da8d53412e` |
| darling_macro | 0.20.11 | MIT | `fc34b93ccb385b40dc71c6fceac4b2ad23662c7eeb248cf10d529b7e055b6ead` |
| dary_heap | 0.3.9 | MIT OR Apache-2.0 | `8b1e3a325bc115f096c8b77bbf027a7c2592230e70be2d985be950d3d5e60ebe` |
| derive_builder | 0.20.2 | MIT OR Apache-2.0 | `507dfb09ea8b7fa618fcf76e953f4f5e192547945816d5358edffe39f6f94947` |
| derive_builder_core | 0.20.2 | MIT OR Apache-2.0 | `2d5bcf7b024d6835cfb3d473887cd966994907effbe9227e8c8219824d06c4e8` |
| derive_builder_macro | 0.20.2 | MIT OR Apache-2.0 | `ab63b0e2bf4d5928aff72e83a7dace85d7bba5fe12dcc3c5a572d78caffd3f3c` |
| either | 1.18.0 | MIT OR Apache-2.0 | `252afb9ae5eaa683babdc6a068b3f5726eb19e05070c731f9b2a23a7c3e8ed34` |
| esaxx-rs | 0.1.10 | Apache-2.0 | `d817e038c30374a4bcb22f94d0a8a0e216958d4c3dcde369b1439fec4bdda6e6` |
| find-msvc-tools | 0.1.14 | MIT OR Apache-2.0 | `aedcfb3409746eddb02b9e19ebda1c3394f759a152e48ee875a0844d1b955484` |
| fnv | 1.0.7 | Apache-2.0 / MIT | `3f9eec918d3f24069decb9af1554cad7c880e2da24a9afd88aca000531ab82c1` |
| getrandom | 0.3.4 | MIT OR Apache-2.0 | `899def5c37c4fd7b2664648c28120ecec138e4d395b459e5ca34f9cce2dd77fd` |
| ident_case | 1.0.1 | MIT/Apache-2.0 | `b9e0384b61958566e926dc50660321d12159025e767c18e043daf26b70104c39` |
| itertools | 0.14.0 | MIT OR Apache-2.0 | `2b192c782037fadd9cfa75548310488aabdbf3d2da73885b31bd0abd03351285` |
| itoa | 1.0.18 | MIT OR Apache-2.0 | `8f42a60cbdf9a97f5d2305f08a87dc4e09308d1276d28c869c684d7777685682` |
| libc | 0.2.189 | MIT OR Apache-2.0 | `3eaf3ede3fee6db1a4c2ee091bf8a8b4dccdc6d17f656fb07896ee72867612f2` |
| log | 0.4.34 | MIT OR Apache-2.0 | `f9f8bd3e56ce4dfc153cf470fffbfa98c7620958b312ca5c3a4b8d5181fd13c6` |
| macro_rules_attribute | 0.2.3 | Apache-2.0 OR MIT OR Zlib | `b3ae8f6d608c795738406608304d30a2dfbdc8e58e44f7ba43236da5208ded3c` |
| macro_rules_attribute-proc_macro | 0.2.3 | Apache-2.0 OR MIT OR Zlib | `fc04a4c58212d57930a24bf47d3fa87485264a3a054e9c10e042eb373573ad3c` |
| memchr | 2.8.3 | Unlicense OR MIT | `cf8baf1c55e62ffcace7a9f06f4bd9cd3f0c4beb022d3b367256b91b87513d98` |
| minimal-lexical | 0.2.1 | MIT/Apache-2.0 | `68354c5c6bd36d73ff3feceb05efa59b6acb7626617f4962be322a825e61f79a` |
| monostate | 0.1.18 | MIT OR Apache-2.0 | `3341a273f6c9d5bef1908f17b7267bbab0e95c9bf69a0d4dcf8e9e1b2c76ef67` |
| monostate-impl | 0.1.18 | MIT OR Apache-2.0 | `e4db6d5580af57bf992f59068d4ea26fd518574ff48d7639b255a36f9de6e7e9` |
| nom | 7.1.3 | MIT | `d273983c5a657a70a3e8f2a01329822f3b8c8172b73826411a55751e404a0a4a` |
| once_cell | 1.21.4 | MIT OR Apache-2.0 | `9f7c3e4beb33f85d45ae3e3a1792185706c8e16d043238c593331cc7cd313b50` |
| onig | 6.5.3 | MIT | `0cc3cbf698f9438986c11a880c90a6d04b9de27575afd28bbf45b154b6c709e2` |
| onig_sys | 69.9.3 | MIT | `1e68317604e77e53b85896388e1a803c1d21b74c899ec9e5e1112db90735edd7` |
| paste | 1.0.15 | MIT OR Apache-2.0 | `57c0d7b74b563b49d38dae00a0c37d4d6de9b432382b2892f0574ddcae73fd0a` |
| pastey | 0.2.3 | MIT OR Apache-2.0 | `2ee67f1008b1ba2321834326597b8e186293b049a023cdef258527550b9935b4` |
| pkg-config | 0.3.34 | MIT OR Apache-2.0 | `f6b464fbc74e149a392436b17d523f769e057cb6877f6a5c4618bc6f11800548` |
| ppv-lite86 | 0.2.21 | MIT OR Apache-2.0 | `85eae3c4ed2f50dcfe72643da4befc30deadb458a9b590d720cde2f2b1e97da9` |
| proc-macro2 | 1.0.107 | MIT OR Apache-2.0 | `985e7ec9bb745e6ce6535b544d84d6cd6f7ad8bd711c398938ae983b91a766d9` |
| quote | 1.0.47 | MIT OR Apache-2.0 | `1fbf4db142a473a8d80c26bbf18454ed458bf8d26c8219c331daecfdbd079001` |
| r-efi | 5.3.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later (UEFI only; not compiled for Linux) | `69cdb34c158ceb288df11e18b4bd39de994f6657d83847bdffdbd7f346754b0f` |
| rand | 0.9.5 | MIT OR Apache-2.0 | `b9ef1d0d795eb7d84685bca4f72f3649f064e6641543d3a8c415898726a57b41` |
| rand_chacha | 0.9.0 | MIT OR Apache-2.0 | `d3022b5f1df60f26e1ffddd6c66e8aa15de382ae63b3a0c1bfc0e4d3e3f325cb` |
| rand_core | 0.9.5 | MIT OR Apache-2.0 | `76afc826de14238e6e8c374ddcc1fa19e374fd8dd986b0d2af0d02377261d83c` |
| rayon | 1.12.0 | MIT OR Apache-2.0 | `fb39b166781f92d482534ef4b4b1b2568f42613b53e5b6c160e24cfbfa30926d` |
| rayon-cond | 0.4.0 | Apache-2.0/MIT | `2964d0cf57a3e7a06e8183d14a8b527195c706b7983549cd5462d5aa3747438f` |
| rayon-core | 1.13.0 | MIT OR Apache-2.0 | `22e18b0f0062d30d4230b2e85ff77fdfe4326feb054b9783a3460d8435c8ab91` |
| regex | 1.13.1 | MIT OR Apache-2.0 | `f020237b6c8eed93db2e2cb53c00c60a8e1bc73da7d073199a1180401450218d` |
| regex-automata | 0.4.18 | MIT OR Apache-2.0 | `ad8553b9b26413251cbf30e620595c7a41b3887f03da04579c0e6b0d6a06b4b2` |
| regex-syntax | 0.8.11 | MIT OR Apache-2.0 | `d6f6ff9a378485b298a5286656da665ba74413d36db0979633275d2e708145d4` |
| rustversion | 1.0.23 | MIT OR Apache-2.0 | `cf54715a573b99ac80df0bc206da022bcd442c974952c7b9720069370852e21f` |
| ryu | 1.0.23 | Apache-2.0 OR BSL-1.0 | `9774ba4a74de5f7b1c1451ed6cd5285a32eddb5cccb8cc655a4e50009e06477f` |
| serde | 1.0.229 | MIT OR Apache-2.0 | `4148590afebada386688f18773da617792bf2ef03ffc1e4cbd2b1d45b023e0ba` |
| serde_core | 1.0.229 | MIT OR Apache-2.0 | `67dca2c9c51e58a4791a4b1ed58308b39c64224d349a935ab5039aa360942a48` |
| serde_derive | 1.0.229 | MIT OR Apache-2.0 | `e7a5d71263a5a7d47b41f6b3f06ba276f10cc18b0931f1799f710578e2309348` |
| serde_json | 1.0.151 | MIT OR Apache-2.0 | `c841b55ecdae098c80dcae9cf767f6f8a0c2cdb3416bbef72181df4d0fe73f14` |
| shlex | 2.0.1 | MIT OR Apache-2.0 | `f8fadd59c855ef2080decdef8ff161eb6661b86933c9d82e5ba29dc602a55aba` |
| smallvec | 1.16.2 | MIT OR Apache-2.0 | `f9395f0f0eee849a9b707b2f06bb92a6a422090e2123bb2ef8e87a0e61892a8e` |
| spm_precompiled | 0.1.4 | Apache-2.0 | `5851699c4033c63636f7ea4cf7b7c1f1bf06d0cc03cfb42e711de5a5c46cf326` |
| static_assertions | 1.1.0 | MIT OR Apache-2.0 | `a2eb9349b6444b326872e140eb1cf5e7c522154d69e7a0ffb0fb81c06b37543f` |
| strsim | 0.11.1 | MIT | `7da8b5736845d9f2fcb837ea5d9e2628564b3b043a70948a3f0b778838c5fb4f` |
| syn | 2.0.119 | MIT OR Apache-2.0 | `872831b642d1a07999a962a351ed35b955ea2cfc8f3862091e2a240a84f17297` |
| syn | 3.0.6 | MIT OR Apache-2.0 | `8593e8e72159ed2257d083c7a454a85cbf854f37a0966d8d483aff8c8a3ebcee` |
| thiserror | 2.0.21 | MIT OR Apache-2.0 | `09e52cb86a36cede5cb101bf8908837b3e4c6e5e59fe7fd85c23fb56200d189e` |
| thiserror-impl | 2.0.21 | MIT OR Apache-2.0 | `fe5197923287db20a58125f0bc85c062f7f2c892de97b18c356f9efb14b28524` |
| tokenizers | 0.21.4 | Apache-2.0 | `a620b996116a59e184c2fa2dfd8251ea34a36d0a514758c6f966386bd2e03476` |
| unicode-ident | 1.0.26 | (MIT OR Apache-2.0) AND Unicode-3.0 | `d245f478577f809a851594d02313b640fb437e0bb33866753cff937863096954` |
| unicode-normalization-alignments | 0.1.12 | MIT/Apache-2.0 | `43f613e4fa046e69818dd287fdc4bc78175ff20331479dab6e1b0f98d57062de` |
| unicode-segmentation | 1.13.3 | MIT OR Apache-2.0 | `c6f5d3c3b1bf09027a88a6bc961fc00497d651009560b5463668dc81b0fa87a8` |
| unicode_categories | 0.1.1 | MIT OR Apache-2.0 | `39ec24b3121d976906ece63c9daad25b85969647682eee313cb5779fdd69e14e` |
| version_check | 0.9.5 | MIT/Apache-2.0 | `0b928f33d975fc6ad9f86c8f283853ad26bdd5b10b7f1542aa2fa15e2289105a` |
| wasip2 | 1.0.4+wasi-0.2.12 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT (WASI only; not compiled for Linux) | `b67efb37e106e55ce722a510d6b5f9c17f083e5fc79afc2badeb12cc313d9487` |
| wit-bindgen | 0.57.1 | Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT (WASI only; not compiled for Linux) | `1ebf944e87a7c253233ad6766e082e3cd714b5d03812acc24c318f549614536e` |
| zerocopy | 0.8.59 | BSD-2-Clause OR Apache-2.0 OR MIT | `6df92bf3d9227be3d53173901ddbffac2babc27ae50f397776ffd6dc33f800cb` |
| zerocopy-derive | 0.8.59 | BSD-2-Clause OR Apache-2.0 OR MIT | `ac4f328cf2f05d084e496c3e9c3f33ed0a183656a16e1fcec4d464d8373aec82` |
| zmij | 1.0.23 | MIT | `29666d0abbfad1e3dc4dcf6144730dd3a3ab225bbbdac83319345b1b44ccfc1b` |

## Offline build

OpenWrt-style: fetch everything in a download step, then build with no network.

1. Download: LiteRT at the pinned commit, tokenizers-cpp at the pinned commit, every row of the
   LiteRT table (git commit or archive with the sha256 above), cpp-httplib and nlohmann/json, and
   `cargo vendor` of `tokenizers-cpp/rust` after applying `patches/musl/tokenizers-cpp`.
2. Apply `patches/musl/litert/*.patch` and `patches/musl/tokenizers-cpp/*.patch` (`git am` or
   `patch -p1`).
3. Build a host `flatc` from the flatbuffers v25.9.23 source (host compiler, target `flatc` only)
   and pass its directory as `-DTFLITE_HOST_TOOLS_DIR`.
4. Unpack every LiteRT-table source at its `B/...` path and configure LiteRT with
   `-DFETCHCONTENT_FULLY_DISCONNECTED=ON -DPTHREADPOOL_SOURCE_DIR=B/pthreadpool-source
   -DFXDIV_SOURCE_DIR=B/FXdiv-source -DFP16_SOURCE_DIR=B/FP16-source` (aarch64 also
   `-DKLEIDIAI_SOURCE_DIR=B/kleidiai-source`). `tools/feed/build-litert.sh` passes anything in
   `LITERT_CMAKE_EXTRA` through; its LiteRT build directory is `$B/litert`.
5. Configure tflite-server with `-DFETCHCONTENT_FULLY_DISCONNECTED=ON
   -DFETCHCONTENT_SOURCE_DIR_HTTPLIB=<dir> -DFETCHCONTENT_SOURCE_DIR_JSON=<dir>`
   (`TFLITE_SERVER_CMAKE_EXTRA` in `tools/feed/build-tflite-server.sh`).
6. Build the tokenizer shim with `CARGO_NET_OFFLINE=true` and the vendored sources.

Measured on ai4, 2026-10-31 (x86_64): `tools/feed/offline-check.sh x86_64` ran steps 4-6 inside
`unshare -rn` (no network: a `curl` to github.com fails inside it), with every source unpacked
from the online build's trees without `.git`. LiteRT, the shim (from `cargo vendor`) and
tflite-server built, and the offline binary gives the online binary's `/classify` answer. The
configure still prints "Downloading TensorFlow repository..." (a status line printed before
FetchContent runs disconnected); nothing is fetched. The stripped binary is 4 KiB smaller than the
online one (10,891,128 vs 10,895,224 bytes): embedded source paths differ.


## Licence flags

- No dependency of the default build blocks publishing its source in a public feed: all are
  Apache-2.0, BSD, MIT, MPL-2.0 (Eigen, MPL2-only mode), Zlib, BSL-1.0, Unicode-3.0 or public domain.
- Blocked from the feed and NOT fetched by this build: Qualcomm QAIRT SDK headers
  (softwarecenter.qualcomm.com), MediaTek NeuroPilot headers (s3 mediatek.neuropilot.com), Samsung
  AI LiteCore (soc-developer.semiconductor.samsung.com, with prebuilt `.so` compilers). Unpatched
  LiteRT v2.3.2 and v2.2.0 download all three at configure time; the v0.3.0 release build did too
  (the v2.2.0 trees on ai4 hold `_deps/qnn_headers` and `_deps/litecore_headers`; nothing linked them).
- FXdiv: pthreadpool's CMake asks for `master`. `tools/feed/prepare-sources.sh` checks out the commit
  above and `tools/feed/build-litert.sh` passes it as `FXDIV_SOURCE_DIR`, refusing any other commit.
- Unpinned prebuilt binary, NOT fetched by this build: `libLiteRtWebGpuAccelerator.so` from a
  "latest" URL (unpatched LiteRT downloads it at configure time; patch 0003).
- Fetched but not in the binary: protobuf (no protobuf symbol or string in the linked x86_64
  binary). SafeInt is header-only and compiled in.

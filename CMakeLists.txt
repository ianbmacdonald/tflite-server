cmake_minimum_required(VERSION 3.24)
project(tflite-server VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TFLITE_SERVER_PREPROCESS_TESTS
  "Build the image preprocessing tests (they need neither LiteRT nor tokenizers-cpp)" OFF)
set(TFLITE_SERVER_OPS "" CACHE FILEPATH
  "Register only the TFLite builtin ops listed in this file (one per line; tools/tflite_ops.py writes one from models). Empty: all builtin ops")
option(TFLITE_SERVER_GC_SECTIONS
  "Compile with -ffunction-sections -fdata-sections and link with --gc-sections" OFF)

include(FetchContent)
set(HTTPLIB_USE_OPENSSL_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_ZLIB_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
set(HTTPLIB_USE_BROTLI_IF_AVAILABLE OFF CACHE BOOL "" FORCE)
FetchContent_Declare(httplib
  GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
  GIT_TAG a7bc00e3307fecdb4d67545e93be7b88cfb1e186)  # v0.18.3
FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG 9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03)  # v3.11.3
FetchContent_MakeAvailable(json)

find_package(Threads REQUIRED)

# Decode, resize and manifest parsing: no LiteRT, so the tests build anywhere.
add_library(tflite_image STATIC
  src/flat_json.cpp
  src/image_manifest.cpp
  src/image_preprocess.cpp
  src/stb_image_impl.cpp)
target_include_directories(tflite_image PUBLIC src third_party/stb)
target_link_libraries(tflite_image PUBLIC nlohmann_json::nlohmann_json)

if(TFLITE_SERVER_PREPROCESS_TESTS)
  enable_testing()
  add_executable(test_image_preprocess tests/test_image_preprocess.cpp)
  target_link_libraries(test_image_preprocess PRIVATE tflite_image Threads::Threads)
  add_executable(test_image_manifest tests/test_image_manifest.cpp)
  target_link_libraries(test_image_manifest PRIVATE tflite_image)
  add_executable(test_flat_json tests/test_flat_json.cpp)
  target_link_libraries(test_flat_json PRIVATE tflite_image)
  add_executable(fuzz_image_decode tests/fuzz_image_decode.cpp)
  target_link_libraries(fuzz_image_decode PRIVATE tflite_image)
  add_test(NAME image_preprocess
    COMMAND test_image_preprocess "${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
  add_test(NAME image_manifest
    COMMAND test_image_manifest "${CMAKE_CURRENT_SOURCE_DIR}/tests/data")
  add_test(NAME flat_json COMMAND test_flat_json)
  add_test(NAME image_decode_fuzz_smoke
    COMMAND fuzz_image_decode "${CMAKE_CURRENT_SOURCE_DIR}/tests/data" 2000)
endif()

# LiteRT is consumed as a CMake build tree of the LiteRT repo (static libraries),
# e.g. built with prpl-harness build/litert/litert-v2.2.0-musl-build.sh.
set(LITERT_SRC "" CACHE PATH "LiteRT source checkout (the repo root that contains litert/ and tflite/)")
set(LITERT_BUILD "" CACHE PATH "LiteRT CMake build tree (configured from <LITERT_SRC>/litert)")
# tokenizers-cpp's Rust C shim, built for the target (cargo build --release --target ...).
set(TOKENIZERS_CPP_SRC "" CACHE PATH "mlc-ai/tokenizers-cpp checkout (for include/tokenizers_c.h)")
set(TOKENIZERS_C_LIB "" CACHE FILEPATH "libtokenizers_c.a built for the target")
set(_missing "")
foreach(v LITERT_SRC LITERT_BUILD TOKENIZERS_CPP_SRC TOKENIZERS_C_LIB)
  if(NOT ${v})
    list(APPEND _missing ${v})
  endif()
endforeach()
if(_missing)
  if(TFLITE_SERVER_PREPROCESS_TESTS)
    message(STATUS "tflite-server: tests-only build (${_missing} not set); the server is not built")
    return()
  endif()
  message(FATAL_ERROR "${_missing} required")
endif()

FetchContent_MakeAvailable(httplib)

# Every static archive of the LiteRT build except its CLI tools and test/bench
# harnesses, linked as one group: the dependency order inside LiteRT's tree is
# long and cyclic, and the linker keeps only the objects actually referenced.
file(GLOB_RECURSE _litert_archives "${LITERT_BUILD}/*.a")
list(FILTER _litert_archives EXCLUDE REGEX "/(tools|gtest|gmock|googletest|benchmark)[^/]*/")
list(FILTER _litert_archives EXCLUDE REGEX "(gtest|gmock|benchmark)[^/]*\\.a$")
list(FILTER _litert_archives EXCLUDE REGEX "/testdata/|-src/")
list(LENGTH _litert_archives _n)
message(STATUS "tflite-server: linking ${_n} LiteRT archives from ${LITERT_BUILD}")

add_executable(tflite-server
  src/main.cpp
  src/image_model.cpp
  src/litert_engine.cpp
  src/text_model.cpp)
target_compile_definitions(tflite-server PRIVATE TFLITE_SERVER_VERSION="${PROJECT_VERSION}")
if(TFLITE_SERVER_OPS)
  include(cmake/SelectedOps.cmake)
  tflite_server_selected_ops("${TFLITE_SERVER_OPS}" "${LITERT_SRC}" _selected_ops_srcs)
  target_sources(tflite-server PRIVATE ${_selected_ops_srcs})
endif()
if(TFLITE_SERVER_GC_SECTIONS)
  # Only drops LiteRT code at section granularity if LiteRT was built with the same flags.
  target_compile_options(tflite-server PRIVATE -ffunction-sections -fdata-sections)
  target_compile_options(tflite_image PRIVATE -ffunction-sections -fdata-sections)
  target_link_options(tflite-server PRIVATE -Wl,--gc-sections)
endif()
# LiteRT's C++ headers reuse type names as member names (Span, ElementType), which
# GCC rejects by default; LiteRT's own CMake build compiles with -fpermissive too.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(tflite-server PRIVATE -fpermissive)
endif()
target_include_directories(tflite-server PRIVATE
  "${LITERT_SRC}"
  "${LITERT_SRC}/tflite/.."
  "${LITERT_BUILD}/abseil-cpp"
  "${LITERT_BUILD}/flatbuffers/include"
  "${LITERT_BUILD}/include"
  "${LITERT_BUILD}/tflite_build"
  "${LITERT_BUILD}/tflite_build/tensorflow-src"
  "${TOKENIZERS_CPP_SRC}/include")
target_link_libraries(tflite-server PRIVATE
  tflite_image
  httplib::httplib nlohmann_json::nlohmann_json
  -Wl,--start-group ${_litert_archives} -Wl,--end-group
  "${TOKENIZERS_C_LIB}"
  ${CMAKE_DL_LIBS} Threads::Threads m)

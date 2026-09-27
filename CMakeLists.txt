cmake_minimum_required(VERSION 3.24)
project(tflite-server CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# LiteRT is consumed as a CMake build tree of the LiteRT repo (static libraries),
# e.g. built with prpl-harness build/litert/litert-v2.2.0-musl-build.sh.
set(LITERT_SRC "" CACHE PATH "LiteRT source checkout (the repo root that contains litert/ and tflite/)")
set(LITERT_BUILD "" CACHE PATH "LiteRT CMake build tree (configured from <LITERT_SRC>/litert)")
# tokenizers-cpp's Rust C shim, built for the target (cargo build --release --target ...).
set(TOKENIZERS_CPP_SRC "" CACHE PATH "mlc-ai/tokenizers-cpp checkout (for include/tokenizers_c.h)")
set(TOKENIZERS_C_LIB "" CACHE FILEPATH "libtokenizers_c.a built for the target")
foreach(v LITERT_SRC LITERT_BUILD TOKENIZERS_CPP_SRC TOKENIZERS_C_LIB)
  if(NOT ${v})
    message(FATAL_ERROR "${v} is required")
  endif()
endforeach()

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
FetchContent_MakeAvailable(httplib json)

# Every static archive of the LiteRT build except its CLI tools and test/bench
# harnesses, linked as one group: the dependency order inside LiteRT's tree is
# long and cyclic, and the linker keeps only the objects actually referenced.
file(GLOB_RECURSE _litert_archives "${LITERT_BUILD}/*.a")
list(FILTER _litert_archives EXCLUDE REGEX "/(tools|gtest|gmock|googletest|benchmark)[^/]*/")
list(FILTER _litert_archives EXCLUDE REGEX "(gtest|gmock|benchmark)[^/]*\\.a$")
list(FILTER _litert_archives EXCLUDE REGEX "/testdata/|-src/")
list(LENGTH _litert_archives _n)
message(STATUS "tflite-server: linking ${_n} LiteRT archives from ${LITERT_BUILD}")

add_executable(tflite-server src/main.cpp)
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
  httplib::httplib nlohmann_json::nlohmann_json
  -Wl,--start-group ${_litert_archives} -Wl,--end-group
  "${TOKENIZERS_C_LIB}"
  ${CMAKE_DL_LIBS} pthread m)

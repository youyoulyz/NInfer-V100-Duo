#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/../.." && pwd)

readonly deps_dir="${repo_dir}/build/_deps"
readonly build_dir="${repo_dir}/build-v100-duo"

NINFER_DEPS_DIR="${deps_dir}" "${script_dir}/build_dependencies.sh"

# glibc >= 2.41 and CUDA 12.8 disagree on the C23 cospi/sinpi family, which fails CMake's CUDA
# compiler check before any project file is compiled. Every translation unit already receives the
# toolkit include directory as a system include; giving the check the same flag is all it needs.
# Preserve any caller-supplied CUDA flags.
cuda_flags="-isystem /usr/local/cuda-12.8/targets/x86_64-linux/include"
if [[ -n "${CMAKE_CUDA_FLAGS:-}" ]]; then cuda_flags="${cuda_flags} ${CMAKE_CUDA_FLAGS}"; fi

PKG_CONFIG_PATH="${deps_dir}/install/lib/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}" \
    cmake -S "${repo_dir}" -B "${build_dir}" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES=70 \
        -DCMAKE_CUDA_FLAGS="${cuda_flags}" \
        -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF
cmake --build "${build_dir}" --target ninfer ninfer-serve -j

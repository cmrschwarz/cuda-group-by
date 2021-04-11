#!/bin/bash
# creates a ./build folder if not already present and runs cmake .. (Release config) in it
# with the necessary cuda env fixes applied

script_dir="$(dirname "$(readlink -f "$0")")"

build_dir="$script_dir/build"

mkdir -p "$build_dir"
cd "$build_dir"
#rm -rf *

export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc

cmake -DCMAKE_BUILD_TYPE=Release -DCUDA_GROUP_BY_USE_CUB_SUBMODULE=OFF ..

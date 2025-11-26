#!/bin/bash

pip install nixl-cu13

cd /sgl-workspace

pip uninstall -y deep_ep
cd DeepEP
TORCH_CUDA_ARCH_LIST='9.0;10.0;10.3' MAX_JOBS=$(nproc) pip install --no-build-isolation .
#!/bin/bash
# Install sglang from main branch

set -e

cd /sgl-workspace

rm -rf sglang

git clone https://github.com/sgl-project/sglang.git
cd sglang
git config --global --add safe.directory "*"
pip install -e "python"

python3 -m pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps
python3 -m pip install nvidia-cudnn-cu12==9.16.0.29 --force-reinstall --no-deps
python3 -m pip install nvidia-cutlass-dsl==4.3.0 --force-reinstall --no-deps

#!/bin/bash
# Install sglang from main branch

set -e

cd /sgl-workspace

rm -rf sglang

git clone https://github.com/sgl-project/sglang.git
cd sglang
git config --global --add safe.directory "*"
pip install -e "python"


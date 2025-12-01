#!/bin/bash

cd /sgl-workspace

rm -rf sglang

git clone --branch revert-11312-cheng/dev/skip-weight-load https://github.com/sgl-project/sglang.git

cd sglang
git config --global --add safe.directory "*"
pip install -e "python"
#!/bin/bash
BRANCH="baizhou/gb200-spec"

cd /sgl-workspace/sglang
git remote remove origin
git remote add origin https://github.com/sgl-project/sglang.git
git fetch origin
git checkout origin/${BRANCH}
pip install -e python
#!/bin/bash

DYNAMO_COMMIT="${DYNAMO_COMMIT:-main}"

# dynamo
cd /sgl-workspace
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo
git checkout "$DYNAMO_COMMIT"
cd lib/bindings/python

# rust 
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"

# install runtime
RUSTFLAGS="-C target-cpu=native" maturin build -o ./dist
pip install dist/*.whl

# install dynamo
cd /sgl-workspace/dynamo
pip install .

#!/usr/bin/env bash
set -euo pipefail
# Run from the mbdac repository. Reuse .venv/Torch/CUDA; no SDK installation.
benchmark_python="${MBDAC_PYTHON:-.venv/bin/python}"
benchmark_source="data/external/holosoma"
if [ ! -d "$benchmark_source/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/amazon-far/holosoma.git "$benchmark_source"
fi
git -C "$benchmark_source" checkout d18d6cc50f872c15e904a22ceac22313cec955c8
uv pip install --python "$benchmark_python" --no-deps -e "$benchmark_source/src/holosoma"
# Minimal dependencies imported by the original pure-Torch learner and configs.
# Full Holosoma metadata also lists simulator/deployment packages unused here;
# installing those would replace the matched MJLab environment unnecessarily.
uv pip install --python "$benchmark_python" loguru==0.7.3 omegaconf==2.4.0 easydict==1.13 \
  tyro==1.0.16 rich==14.3.4 termcolor==3.3.0 onnx==1.23.2 yourdfpy==0.0.60
"$benchmark_python" -c 'from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent; print(FastSACAgent)'

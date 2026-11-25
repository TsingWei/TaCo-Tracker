#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec torchrun --standalone --nproc_per_node="${NUM_GPUS:-4}" train.py --config configs/taco_s.yaml "$@"

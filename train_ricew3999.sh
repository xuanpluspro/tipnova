#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
TAG="${1:-ricew3999_tipfuse_ep50_bs32}"
DATA_PATH="${DATA_PATH:-$PWD/data/RiceW3999}"
WEIGHTS_PATH="${WEIGHTS_PATH:-$PWD/pretrained/vgg16_bn-6c64b313.pth}"
mkdir -p "exp/$TAG/train.log"
python -u train.py --data-path "$DATA_PATH" --pretrained "$WEIGHTS_PATH" \
  --batch-size "${BATCH_SIZE:-32}" --epochs "${EPOCHS:-50}" \
  --workers "${WORKERS:-4}" --eval-every "${VAL_EVERY:-1}" --tag "$TAG" \
  "${@:2}" 2>&1 | tee -a "exp/$TAG/train.log/running.log"

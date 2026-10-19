#!/usr/bin/env bash
# Canonical optimized_v1 launcher; output format follows optimized/run.sh.
set -eo pipefail
cd "$(dirname "$0")"

exec 9> .training.lock
flock -n 9 || { echo 'A training is already running here.' >&2; exit 2; }

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LLM_TEST_ENV=/SharePath/chenyupeng/cyp-testtest/llm_test_migrated
export PATH="$LLM_TEST_ENV/bin:/home/naie/.local/bin:$PATH"
export AIRBENCH_TORCHVISION_PATH="${AIRBENCH_TORCHVISION_PATH:-/SharePath/chenyupeng/diff_model/cifar10-airbench/airbench_npu/runtime_dependencies}"
export PYTHONPATH="$AIRBENCH_TORCHVISION_PATH:${PYTHONPATH:-}"
export AIRBENCH_DATA_PATH="${AIRBENCH_DATA_PATH:-/SharePath/chenyupeng/diff_model/cifar10-airbench/airbench_npu/data/cifar10}"
export AIRBENCH_RUNS="${AIRBENCH_RUNS:-20}"
export AIRBENCH_DEVICES="${AIRBENCH_DEVICES:-0,1}"
export AIRBENCH_ALLOW_INTERNAL_FORMAT="${AIRBENCH_ALLOW_INTERNAL_FORMAT:-1}"
export AIRBENCH_PROXY_FULL_COMPILE="${AIRBENCH_PROXY_FULL_COMPILE:-1}"
export AIRBENCH_CACHE_WHITENING="${AIRBENCH_CACHE_WHITENING:-1}"
export AIRBENCH_FINAL_REDUCE_POOL="${AIRBENCH_FINAL_REDUCE_POOL:-1}"
export AIRBENCH_EVAL_CHUNK="${AIRBENCH_EVAL_CHUNK:-10000}"
export AIRBENCH_EVAL_COMPILE="${AIRBENCH_EVAL_COMPILE:-1}"
export AIRBENCH_STREAM_PROXY_MASKS="${AIRBENCH_STREAM_PROXY_MASKS:-1}"
export AIRBENCH_PROXY_LOOKAHEAD_STEPS="${AIRBENCH_PROXY_LOOKAHEAD_STEPS:-4}"
export AIRBENCH_DDP_BUCKET_MB="${AIRBENCH_DDP_BUCKET_MB:-5}"
export AIRBENCH_DDP_BUCKET_VIEW="${AIRBENCH_DDP_BUCKET_VIEW:-1}"
export ASCEND_RT_VISIBLE_DEVICES="$AIRBENCH_DEVICES"
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export PYTHONUNBUFFERED=1
export TORCH_NPU_ENABLE_COMPILE_CACHE=1
export TORCH_NPU_COMPILE_CACHE_DIR="${TORCH_NPU_COMPILE_CACHE_DIR:-$HOME/.cache/airbench_optimized_v1}"

if [[ ! "$AIRBENCH_DEVICES" =~ ^[0-9]+,[0-9]+$ ]]; then
  echo 'AIRBENCH_DEVICES must contain exactly two comma-separated NPU IDs.' >&2
  exit 2
fi
IFS=, read -r device0 device1 <<< "$AIRBENCH_DEVICES"
if [[ "$device0" == "$device1" ]]; then
  echo 'AIRBENCH_DEVICES must contain two distinct NPU IDs.' >&2
  exit 2
fi

test -f "$AIRBENCH_DATA_PATH/train.pt" && test -f "$AIRBENCH_DATA_PATH/test.pt"
python -c 'import torch, torch_npu, torchvision'

mkdir -p logs
log="$PWD/logs/$(date +%Y%m%d_%H%M%S_%N).log"
started_ns=$(date +%s%N)
printf 'version=optimized_v1 user=%s python=%s mode=ddp syncbn=local physical_npus=%s npu_count_used=2 one_training_process=1 runs=%s\n' \
  "$(id -un)" "$(command -v python)" "$AIRBENCH_DEVICES" "$AIRBENCH_RUNS" | tee "$log"

set +e
python -m torch.distributed.run --standalone --nproc_per_node=2 localbn_ddp.py 2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}
set -e

ended_ns=$(date +%s%N)
{
  printf '%s\n' '=== SUMMARY ==='
  printf 'exit_code=%s process_wall_ns=%s\n' "$status" "$((ended_ns - started_ns))"
} | tee -a "$log"
printf '%s\n' "$log"
exit "$status"

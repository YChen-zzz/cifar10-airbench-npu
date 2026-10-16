#!/usr/bin/env bash
# 唯一产物：logs/<时间戳>.log （header + 完整训练输出，末尾自带 JSON summary）
set -eo pipefail
cd "$(dirname "$0")"
exec 9> .training.lock
flock -n 9 || { echo 'A training is already running here.' >&2; exit 2; }
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LLM_TEST_ENV=/SharePath/chenyupeng/cyp-testtest/llm_test_migrated
export PATH="$LLM_TEST_ENV/bin:/home/naie/.local/bin:$PATH"
export PYTHONPATH="/SharePath/chenyupeng/diff_model/cifar10-airbench/airbench_npu/runtime_dependencies:${PYTHONPATH:-}"
hash -r
export AIRBENCH_DATA_PATH="${AIRBENCH_DATA_PATH:-/SharePath/chenyupeng/diff_model/cifar10-airbench/airbench_npu/data/cifar10}"
export AIRBENCH_RUNS="${AIRBENCH_RUNS:-20}"
export AIRBENCH_SYNCBN="${AIRBENCH_SYNCBN:-local}"
export AIRBENCH_DEVICES="${AIRBENCH_DEVICES:-0,1}"
export AIRBENCH_EVAL_EVERY_EPOCH=0 TASK_QUEUE_ENABLE=2 PYTHONUNBUFFERED=1
export TORCH_NPU_ENABLE_COMPILE_CACHE=1
export TORCH_NPU_COMPILE_CACHE_DIR="${TORCH_NPU_COMPILE_CACHE_DIR:-$HOME/.cache/airbench_optimized}"
[[ "$AIRBENCH_DEVICES" =~ ^([0-9]+),([0-9]+)$ ]] || { echo 'optimized needs exactly two NPU IDs' >&2; exit 2; }
[[ "${BASH_REMATCH[1]}" != "${BASH_REMATCH[2]}" ]] || { echo 'NPU IDs must differ' >&2; exit 2; }
test -f "$AIRBENCH_DATA_PATH/train.pt" && test -f "$AIRBENCH_DATA_PATH/test.pt"
python -c 'import torch, torch_npu, torchvision' || { echo 'preflight failed' >&2; exit 3; }

mkdir -p logs
log="$PWD/logs/$(date +%Y%m%d_%H%M%S_%N).log"
started=$(date +%s%N)
echo "version=optimized user=$(id -un) python=$(command -v python) mode=ddp syncbn=$AIRBENCH_SYNCBN physical_npus=$AIRBENCH_DEVICES npu_count_used=2 one_training_process=1 runs=$AIRBENCH_RUNS" | tee "$log"
set +e
python -m torch.distributed.run --standalone --nproc_per_node=2 airbench96_faster_npu.py \
  --mode ddp --runs "$AIRBENCH_RUNS" --output "$(mktemp -u /tmp/airbench_metrics_XXXX.json)" 2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}
set -e
ended=$(date +%s%N)
printf '=== SUMMARY ===\nexit_code=%s process_wall_ns=%s\n' "$status" "$(( ended - started ))" | tee -a "$log"
echo "$log"
exit "$status"

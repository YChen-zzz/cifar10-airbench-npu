#!/usr/bin/env bash
# 唯一产物：logs/<时间戳>.log
#   header 在启动时立即写入（log 一秒内可见），跑完后追加 device 分段 + SUMMARY
set -eo pipefail
cd "$(dirname "$0")"
exec 9> .training.lock
flock -n 9 || { echo 'A training is already running here.' >&2; exit 2; }
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LLM_TEST_ENV=/SharePath/chenyupeng/cyp-testtest/llm_test_migrated
export PATH="$LLM_TEST_ENV/bin:/home/naie/.local/bin:$PATH"
export AIRBENCH_TORCHVISION_PATH="${AIRBENCH_TORCHVISION_PATH:-/SharePath/chenyupeng/diff_model/cifar10-airbench/airbench_npu/runtime_dependencies}"
export PYTHONPATH="$AIRBENCH_TORCHVISION_PATH:${PYTHONPATH:-}"
hash -r
export AIRBENCH_DATA_PATH="${AIRBENCH_DATA_PATH:-/SharePath/chenyupeng/diff_model/cifar10-airbench/airbench_npu/data/cifar10}"
export AIRBENCH_RUNS="${AIRBENCH_RUNS:-10}"          # 每 die 次数
export AIRBENCH_DEVICES="${AIRBENCH_DEVICES:-0,1}"
export AIRBENCH_EVAL_EVERY_EPOCH=1
export TASK_QUEUE_ENABLE=2 CPU_AFFINITY_CONF=1 PYTHONUNBUFFERED=1
export TORCH_NPU_ENABLE_COMPILE_CACHE=1
export TORCH_NPU_COMPILE_CACHE_DIR="${TORCH_NPU_COMPILE_CACHE_DIR:-$HOME/.cache/airbench_baseline}"
devices=(${AIRBENCH_DEVICES//,/ })
[[ ${#devices[@]} -eq 2 && "${devices[0]}" != "${devices[1]}" ]] || { echo 'baseline needs exactly two distinct NPU IDs' >&2; exit 2; }
test -f "$AIRBENCH_DATA_PATH/train.pt" && test -f "$AIRBENCH_DATA_PATH/test.pt"
python -c 'import torch, torch_npu, torchvision' || { echo 'preflight failed' >&2; exit 3; }

total=$(( AIRBENCH_RUNS * 2 ))
mkdir -p logs
log="$PWD/logs/$(date +%Y%m%d_%H%M%S_%N).log"
tmp="$(mktemp -d)"
pids=()
trap 'kill "${pids[@]}" 2>/dev/null || true; rm -rf "$tmp"' INT TERM EXIT
printf 'version=baseline user=%s python=%s independent_training_processes=2 physical_npus=%s npu_count_used=2 total_runs=%s runs_per_npu=%s workers_per_npu=1 eval_every_epoch=1\n' \
  "$(id -un)" "$(command -v python)" "$AIRBENCH_DEVICES" "$total" "$AIRBENCH_RUNS" | tee "$log"
started=$(date +%s%N)
for device in "${devices[@]}"; do
  ASCEND_RT_VISIBLE_DEVICES="$device" python airbench96_faster_npu.py > "$tmp/device${device}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
ended=$(date +%s%N)

{
  for device in "${devices[@]}"; do
    printf '=== device%s.log ===\n' "$device"
    cat "$tmp/device${device}.log"
  done
  printf '=== SUMMARY ===\n'
  python - "$tmp" "$total" "$(( ended - started ))" "$status" "$AIRBENCH_RUNS" <<'PY'
import sys, json, statistics
from pathlib import Path
tmp, expected, wall_ns, status, per_npu = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
accs, times = [], []
for path in sorted(tmp.glob('device*.log')):
    for line in path.read_text(errors='ignore').splitlines():
        if '|   eval  |' in line:
            f = [x.strip() for x in line.split('|')]
            accs.append(float(f[6])); times.append(float(f[7]))
summary = dict(version='baseline', independent_training_processes=2, runs_per_npu=per_npu,
               expected_runs=expected, completed_runs=len(accs), exit_codes=[status],
               npu_count_used=2, wall_elapsed_s=wall_ns / 1e9,
               wall_throughput_s_per_training=wall_ns / 1e9 / expected if expected else None,
               accuracy_mean=statistics.mean(accs) if accs else None,
               accuracy_std=statistics.stdev(accs) if len(accs) > 1 else None,
               individual_official_time_mean_s=statistics.mean(times) if times else None)
print(json.dumps(summary, indent=2))
PY
} >> "$log"

find logs -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + 2>/dev/null || true
echo "$log"
exit "$status"

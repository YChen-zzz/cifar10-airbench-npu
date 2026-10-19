"""Two-rank local-BN DDP baseline for the original Airbench task.

The original global loader, schedules, global top-k selection, summed loss,
optimizer, lookahead, whitening, validation and TTA remain unchanged.  Model
compute is split across two ranks.  BatchNorm is deliberately *not* converted
to SyncBatchNorm: each rank uses statistics from its local shard.
"""

import json
import os
import random
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import airbench96_faster_npu as original


RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
assert WORLD == 2, "This baseline is defined for exactly two NPU ranks"
torch.npu.set_device(LOCAL_RANK)
if os.environ.get("AIRBENCH_ALLOW_INTERNAL_FORMAT", "0") == "1":
    torch.npu.config.allow_internal_format = True
    _sgd_class = torch.optim.SGD

    def _internal_format_sgd(*args, **kwargs):
        # NPU foreach SGD rejects internal-format gradients.  The scalar path
        # is state-bit-exact to base-format scalar SGD in the 30-step Lab.
        kwargs["foreach"] = False
        return _sgd_class(*args, **kwargs)

    torch.optim.SGD = _internal_format_sgd
if (
    os.environ.get("AIRBENCH_SELECTIVE_COMPILE", "0") == "1"
    or os.environ.get("AIRBENCH_PROXY_FULL_COMPILE", "0") == "1"
    or os.environ.get("AIRBENCH_EVAL_COMPILE", "0") == "1"
):
    cache_root = os.environ.get("TORCH_NPU_COMPILE_CACHE_DIR", "/tmp/airbench_compile")
    os.environ["TORCH_NPU_COMPILE_CACHE_DIR"] = f"{cache_root}_rank{LOCAL_RANK}"
    import torch.utils._triton
    import torch._dynamo.utils
    import torch._dynamo.variables.builder

    # transfer_to_npu aliases CUDA availability to NPU, but CUDA/Triton
    # capability checks are irrelevant to backend='npu' and return None.
    torch.utils._triton.has_triton = lambda: False
    torch._dynamo.utils.has_triton = lambda: False
    torch.utils._triton.has_triton_tma = lambda: False
    torch._dynamo.variables.builder.has_triton_tma = lambda: False
dist.init_process_group("hccl")


class Gather(torch.autograd.Function):
    """All-gather the sharded model output while preserving global-sum grads."""

    @staticmethod
    def forward(ctx, local):
        ctx.local_size = local.shape[0]
        parts = [torch.empty_like(local) for _ in range(WORLD)]
        dist.all_gather(parts, local.contiguous())
        return torch.cat(parts)

    @staticmethod
    def backward(ctx, grad):
        # Every rank evaluates the same global summed loss.  DDP later averages
        # parameter gradients, so scale the local VJP by WORLD exactly once.
        start = RANK * ctx.local_size
        return grad.narrow(0, start, ctx.local_size).contiguous() * WORLD


class LocalBatchNormDDP(nn.Module):
    """Shard model compute but retain ordinary per-rank BatchNorm modules."""

    def __init__(self, module):
        super().__init__()
        self.core = module
        self.dp = DDP(
            self.core,
            device_ids=[LOCAL_RANK],
            broadcast_buffers=False,
            bucket_cap_mb=float(os.environ.get("AIRBENCH_DDP_BUCKET_MB", "25")),
            gradient_as_bucket_view=(
                os.environ.get("AIRBENCH_DDP_BUCKET_VIEW", "0") == "1"
            ),
            static_graph=(os.environ.get("AIRBENCH_DDP_STATIC_GRAPH", "0") == "1"),
        )
        self.eval_forward = (
            torch.compile(self.core.forward, backend="npu", dynamic=False)
            if os.environ.get("AIRBENCH_EVAL_COMPILE", "0") == "1"
            else None
        )

    def forward(self, inputs):
        assert len(inputs) % WORLD == 0
        local = inputs.chunk(WORLD)[RANK].contiguous()
        if not torch.is_grad_enabled() and self.eval_forward is not None:
            output = self.eval_forward(local)
        else:
            output = self.dp(local)
        if torch.is_grad_enabled() and output.requires_grad:
            return Gather.apply(output)
        parts = [torch.empty_like(output) for _ in range(WORLD)]
        dist.all_gather(parts, output.contiguous())
        return torch.cat(parts)

    def __getitem__(self, index):
        return self.core[index]

    def state_dict(self, *args, **kwargs):
        return self.core.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.core.load_state_dict(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self.core.named_parameters(*args, **kwargs)


class NpuExactGelu(nn.Module):
    def forward(self, inputs):
        return torch_npu.npu_gelu(inputs, approximate="none")


# The original script intentionally chooses fresh seeds for every run.  Make
# each choice on rank 0 and broadcast it so both ranks construct one global
# data stream and identical reinitialization before parameter broadcast.
_random_randint = random.randint


def global_randint(lower, upper):
    value = _random_randint(lower, upper) if RANK == 0 else 0
    value_tensor = torch.tensor([value], dtype=torch.int64, device="npu")
    dist.broadcast(value_tensor, 0)
    return int(value_tensor.item())


random.randint = global_randint

_original_reinit = original.reinit_net


def reinit_global(model):
    _original_reinit(model)
    for tensor in model.state_dict().values():
        dist.broadcast(tensor, 0)


original.reinit_net = reinit_global

if os.environ.get("AIRBENCH_CACHE_WHITENING", "0") == "1":
    _original_init_whitening_conv = original.init_whitening_conv
    _whitening_weights = {}

    def init_whitening_conv_cached(layer, train_set, eps=5e-4):
        # The proxy and main whitening kernels are a deterministic function of
        # the same fixed CIFAR-10 prefix.  Reuse the resulting weight, while
        # retaining every reinitialization, RNG draw and subsequent update.
        key = (tuple(layer.weight.shape), layer.weight.dtype, float(eps))
        cached = _whitening_weights.get(key)
        if cached is None:
            _original_init_whitening_conv(layer, train_set, eps)
            _whitening_weights[key] = layer.weight.detach().clone()
        else:
            layer.weight.data.copy_(cached)

    original.init_whitening_conv = init_whitening_conv_cached

if os.environ.get("AIRBENCH_FOREACH_LOOKAHEAD", "0") == "1":
    _lookahead_class = original.LookaheadState

    class ForeachLookaheadState(_lookahead_class):
        def __init__(self, net):
            super().__init__(net)
            self._ema_float_values = [
                value
                for value in self.net_ema.values()
                if value.dtype in (torch.half, torch.float)
            ]
            self._net_float_values = {}

        def update(self, net, decay):
            # Module storages remain stable across load_state_dict; cache their
            # references and fuse the exact lerp/copy sequence into foreach ops.
            key = id(net)
            net_values = self._net_float_values.get(key)
            if net_values is None:
                net_values = [
                    value
                    for value in net.state_dict().values()
                    if value.dtype in (torch.half, torch.float)
                ]
                self._net_float_values[key] = net_values
            torch._foreach_lerp_(self._ema_float_values, net_values, 1 - decay)
            torch._foreach_copy_(net_values, self._ema_float_values)

    original.LookaheadState = ForeachLookaheadState

_original_print_training_details = original.print_training_details


def print_rank0(variables, is_final_entry):
    if RANK == 0:
        _original_print_training_details(variables, is_final_entry)


original.print_training_details = print_rank0


if __name__ == "__main__":
    short_epochs = os.environ.get("AIRBENCH_SHORT_EPOCHS")
    if short_epochs is not None:
        original.hyp["opt"]["train_epochs"] = float(short_epochs)

    proxy = original.make_net(original.hyp["proxy"])
    proxy[0].bias.requires_grad = False
    trainbias = original.make_net(original.hyp["net"])
    freezebias = original.make_net(original.hyp["net"])
    freezebias[0].bias.requires_grad = False
    if os.environ.get("AIRBENCH_SELECTIVE_NPU_GELU", "0") == "1":
        for block in (proxy[2], proxy[3], proxy[4], trainbias[4], freezebias[4]):
            block.activ = NpuExactGelu()
    if os.environ.get("AIRBENCH_SELECTIVE_COMPILE", "0") == "1":
        # Cumulative L4 winner from real-shape F+B Lab: proxy blocks 2/3 and
        # main block 3 only.  Other ConvGroups were faster in eager L0.
        for block in (proxy[3], proxy[4], trainbias[4], freezebias[4]):
            block.forward = torch.compile(block.forward, backend="npu", dynamic=False)
    if os.environ.get("AIRBENCH_PROXY_FULL_COMPILE", "0") == "1":
        # Keep every proxy layer in one graph.  This avoids the FakeTensor leak
        # observed when a compiled ConvGroup hands off to an eager Linear, and
        # preserves the original module/state_dict interface required by DDP.
        proxy.forward = torch.compile(proxy.forward, backend="npu", dynamic=False)
    models = [LocalBatchNormDDP(model) for model in (proxy, trainbias, freezebias)]

    profiler = None
    profile_dir = None
    profile_level = os.environ.get("AIRBENCH_PROFILE_LEVEL")
    if profile_level:
        target = os.environ.get("AIRBENCH_PROFILE_TARGET", "main")
        run_id = os.environ["AIRBENCH_PROFILE_RUN_ID"]
        profile_root = Path.cwd() / "profiling" / f"{run_id}_{profile_level}_{target}"
        profile_dir = profile_root / f"rank_{RANK}"
        profile_dir.mkdir(parents=True, exist_ok=True)
        activities = [torch_npu.profiler.ProfilerActivity.NPU]
        kwargs = {}
        if profile_level.upper() == "L1":
            activities.insert(0, torch_npu.profiler.ProfilerActivity.CPU)
            kwargs.update(
                with_stack=True,
                record_shapes=True,
                profile_memory=True,
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                ),
            )
        profiler = torch_npu.profiler.profile(
            activities=activities,
            schedule=torch_npu.profiler.schedule(
                wait=1, warmup=1, active=3, repeat=1, skip_first=10
            ),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                str(profile_dir)
            ),
            **kwargs,
        )
        profiler.__enter__()

        def profile_step_hook(regime, step):
            if regime == target:
                profiler.step()

        original.PROFILE_STEP_HOOK = profile_step_hook

    results = []
    for run in range(int(os.environ.get("AIRBENCH_RUNS", "1"))):
        dist.barrier()
        torch.npu.synchronize()
        started = time.perf_counter()
        accuracy, official_s = original.main(run, original.hyp, *models)
        torch.npu.synchronize()
        dist.barrier()
        wall_s = time.perf_counter() - started
        result = {
            "run": run,
            "accuracy": accuracy,
            "single_training_wall_s": wall_s,
            "official_training_region_s": official_s,
        }
        results.append(result)
        if RANK == 0:
            print("SINGLE_TRAINING_RESULT " + json.dumps(result), flush=True)

    if RANK == 0:
        summary = {
            "version": "optimized_v1",
            "mode": "ddp",
            "independent_training_processes": 1,
            "physical_npus": os.environ.get("AIRBENCH_DEVICES", "0,1"),
            "batch_dp_devices": [0, 1],
            "distributed_ranks": WORLD,
            "syncbn_implementation": "local",
            "bn_semantics": "local BN per rank",
            "global_proxy_batch": original.hyp["opt"]["batch_size"],
            "local_proxy_batch": original.hyp["opt"]["batch_size"] // WORLD,
            "global_selected_batch": original.hyp["opt"]["batch_size_masked"],
            "local_selected_batch_nominal": original.hyp["opt"]["batch_size_masked"] // WORLD,
            "records": results,
            "accuracy_mean": statistics.mean(item["accuracy"] for item in results),
            "official_time_measured_runs": len(results),
            "official_time_mean_s": statistics.mean(
                item["official_training_region_s"] for item in results
            ),
        }
        print(json.dumps(summary, indent=2), flush=True)

    if profiler is not None:
        profiler.__exit__(None, None, None)
        original.PROFILE_STEP_HOOK = None
        dist.barrier()
        if RANK == 0:
            latest = Path.cwd() / "profiling" / "latest"
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(profile_dir.parent.name, target_is_directory=True)
            print(f"PROFILE_DIR {profile_dir.parent}", flush=True)

    dist.destroy_process_group()

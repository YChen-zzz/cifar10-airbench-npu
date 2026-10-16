# CIFAR-10 Airbench · Ascend NPU Leaderboard

> **This repository tracks NPU reproductions of the cifar10-airbench speedruns on one Ascend 910C NPU.**
>
> Forked from upstream [KellerJordan/cifar10-airbench](https://github.com/KellerJordan/cifar10-airbench) — the original project is the single-GPU CIFAR-10 speedrun: the fastest known training of a neural network to 94% / 96% accuracy on CIFAR-10 using one NVIDIA A100 GPU.
>
> In this fork the upstream scripts are ported and benchmarked on **one Ascend 910C NPU (two dies)**. The goal is a comparable NPU result with auditable source, launch commands, environment metadata, logs and metrics. All algorithmic ideas, records and contributor credit belong to the upstream open-source project and are preserved in full below.
>
> - The table below is the main result of this repository.
> - Implementation sources and successful run logs live under `airbench_npu/`.
> - Anything in the preserved upstream README that mentions CUDA / A100 / cuDNN / Triton / `torch.compile` / multi-GPU launch flags is kept for history and attribution only and does **not** describe the NPU runs.

---

## Runtime

The NPU runs use a common reference runtime with one Ascend 910C NPU:

| Field | Value |
|---|---|
| Python | `/SharePath/chenyupeng/cyp-testtest/llm_test_migrated/bin/python`, version `3.10.19` |
| torch | `2.7.1+cpu` |
| torch_npu | `2.7.1.post2` |
| CANN | `8.5.0` |
| torchvision | `0.22.1`, pinned by the launchers via `AIRBENCH_TORCHVISION_PATH` |
| Hardware | one Ascend 910C NPU (two dies) |
| Process count | two processes, one per die |
| Dataset | CIFAR-10, pre-cached as `train.pt` / `test.pt` under `airbench_npu/data/cifar10/` |

An Ascend 910C NPU contains two dies, so training uses two processes, one per die;
`--nproc_per_node=2` refers to that process count.

Some logs report a PyTorch version such as `2.7.1+cpu`; in the Ascend build the `+cpu`
suffix is the package build string and does not mean the run used CPU instead of NPU.

Both launchers pin the `torchvision` build the runs used by putting it first on
`PYTHONPATH`. The path is the `AIRBENCH_TORCHVISION_PATH` variable, defaulting to
`/SharePath/chenyupeng/diff_model/cifar10-airbench/airbench_npu/runtime_dependencies`
(`torchvision 0.22.1`); set it to use a different build.
`airbench_npu/baseline/run.sh` and `airbench_npu/optimized/run.sh` each write exactly
one log file, `logs/<timestamp>.log`, containing the complete training output.

Run the optimized build with:

```bash
cd airbench_npu/optimized
AIRBENCH_RUNS=20 bash run.sh
```

`AIRBENCH_RUNS` is the number of complete sequential trainings, and the default `20`
reproduces the leaderboard row. The launcher also accepts `AIRBENCH_DEVICES` (NPU device
IDs, default `0,1`), `AIRBENCH_SYNCBN` (BatchNorm mode, default `local`),
`AIRBENCH_DATA_PATH` and `AIRBENCH_TORCHVISION_PATH`. One run of this launcher writes a
single `logs/<timestamp>.log`.

## NPU leaderboard

The following table is the NPU reproduction leaderboard. For each row the source lives
in the corresponding `airbench_npu/<implementation>/` directory and the run log lives in
its `logs/` subdirectory.

| # | Time | Accuracy (mean) | Description | Log |
| - | - | - | - | - |
| 1 | 75.88 s | 96.046% | upstream port, training algorithm unchanged | [log](airbench_npu/baseline/logs/20261016_134816_607999689.log) |
| 2 | 32.66 s | 96.015% | **[optimized]** whole-batch BatchNorm after the layer-wise batch split, per-rank local loss, side-stream proxy overlap, single-reduction final max-pool | [log](airbench_npu/optimized/logs/20261016_180136_ddp.log) |

For reference, the upstream GPU records this port is based on:

| Script | Mean accuracy | Time | Hardware |
| - | - | - | - |
| upstream `airbench96_faster.py` | 96.00% | 27.3 s | 400W NVIDIA A100 |
| upstream `airbench94_muon.py` | 94.01% | 2.59 s | 400W NVIDIA A100 |

---

## Below is the original upstream README (preserved for full attribution)

# CIFAR-10 Airbench 💨

This repo contains the two fastest known algorithms ([edit: not anymore](https://github.com/hiverge/cifar10-speedrun/tree/main)) for training a neural network to 94% or 96% accuracy on CIFAR-10 using a single NVIDIA A100 GPU.

| Script | Mean accuracy | Time | PFLOPs |
| - | - | - | - |
| [airbench94_muon.py](./airbench94_muon.py) | 94.01% | 2.59s | 0.29 |
| [airbench96_faster.py](airbench96_faster.py) | 96.00% | 27.3s | 3.1 |

(Timings done using `torch==2.4.1` on a 400W NVIDIA A100)

For comparison, the standard training used in most studies on CIFAR-10 is much slower:

| Baseline | Mean accuracy | Time | PFLOPs |
| - | - | - | - |
| Standard ResNet-18 training | 96.0% | 7min | 32.3 |

## Quickstart

The current speedrun record for 94% on CIFAR-10 can be run via:
```
git clone https://github.com/KellerJordan/cifar10-airbench.git
cd cifar10-airbench
python airbench94_muon.py
```

(requires `torch` and `torchvision` to be installed)

---
---

## Methods

The set of methods used to obtain these training speeds are described in [the paper](https://arxiv.org/abs/2404.00498).

In addition, [airbench94_muon.py](airbench94_muon.py) uses the [Muon optimizer](https://kellerjordan.github.io/posts/muon/) and [airbench96_faster.py](airbench96_faster.py) uses a form of data filtering. These are both new records since the paper.
We have preserved the paper's old records as well in the `legacy` folder:

| Script | Mean accuracy | Time | PFLOPs |
| - | - | - | - |
| [airbench94_compiled.py](legacy/airbench94_compiled.py) | 94.01% | 3.09s | 0.36 |
| [airbench94.py](legacy/airbench94.py) | 94.01% | 3.83s | 0.36 |
| [airbench95.py](legacy/airbench95.py) | 95.01% | 10.4s | 1.4 |
| [airbench96.py](legacy/airbench96.py) | 96.03% | 34.7s | 4.9 |

![alt](img/alternating_flip.png)
![curve](img/airbench94_intro.png)

## Motivation

CIFAR-10 is one of the most widely used datasets in machine learning, facilitating [thousands of research projects per year](https://paperswithcode.com/dataset/cifar-10). 
This repo provides fast and stable training baselines for CIFAR-10 in order to help accelerate this research.
The trainings are provided as easily runnable dependency-free PyTorch scripts, and can replace classic baselines like training ResNet-20 or ResNet-18.

## Using the GPU-accelerated dataloader independently

For writing custom CIFAR-10 experiments or trainings, you may find it useful to use the GPU-accelerated dataloader independently.
```
import airbench
train_loader = airbench.CifarLoader('/tmp/cifar10', train=True, aug=dict(flip=True, translate=4, cutout=16), batch_size=500)
test_loader = airbench.CifarLoader('/tmp/cifar10', train=False, batch_size=1000)

for epoch in range(200):
    for inputs, labels in train_loader:
        # outputs = model(inputs)
        # loss = F.cross_entropy(outputs, labels)
        ...
```

If you wish to modify the data in the loader, it can be done like so:
```
import airbench
train_loader = airbench.CifarLoader('/tmp/cifar10', train=True, aug=dict(flip=True, translate=4, cutout=16), batch_size=500)
mask = (train_loader.labels < 6) # (this is just an example, the mask can be anything)
train_loader.images = train_loader.images[mask]
train_loader.labels = train_loader.labels[mask]
print(len(train_loader)) # The loader now contains 30,000 images and has batch size 500, so this prints 60.
```

## Example data-selection experiment

Airbench can be used as a platform for experiments in data selection and active learning.
The following is an example experiment which demonstrates the classic result that low-confidence examples provide more training signal than random examples.
It runs in <20 seconds on an A100.

```
import torch
from airbench import train94, infer, evaluate, CifarLoader

net = train94(label_smoothing=0) # train this network without label smoothing to get a better confidence signal

loader = CifarLoader('cifar10', train=True, batch_size=1000)
logits = infer(net, loader)
conf = logits.log_softmax(1).amax(1) # confidence

train_loader = CifarLoader('cifar10', train=True, batch_size=1024, aug=dict(flip=True, translate=2))
mask = (torch.rand(len(train_loader.labels)) < 0.6)
print('Training on %d images selected randomly' % mask.sum())
train_loader.images = train_loader.images[mask]
train_loader.labels = train_loader.labels[mask]
train94(train_loader, epochs=16) # yields around 93% accuracy

train_loader = CifarLoader('cifar10', train=True, batch_size=1024, aug=dict(flip=True, translate=2))
mask = (conf < conf.float().quantile(0.6))
print('Training on %d images selected based on minimum confidence' % mask.sum())
train_loader.images = train_loader.images[mask]
train_loader.labels = train_loader.labels[mask]
train94(train_loader, epochs=16) # yields around 94% accuracy => low-confidence sampling is better than random.
```

## Prior work

This project builds on the excellent previous record https://github.com/tysam-code/hlb-CIFAR10 (6.3 A100-seconds to 94%).

Which itself builds on the amazing series https://myrtle.ai/learn/how-to-train-your-resnet/ (26 V100-seconds to 94%, which is >=8 A100-seconds)


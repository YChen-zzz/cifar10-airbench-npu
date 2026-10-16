# CIFAR10 Airbench 原始 NPU 基线

原始训练入口取自旧工程 Git `5b58c71`。迁移前 20 次完整训练的最终 Acc 为 `0.9602 ± 0.0012`，历史官方计时约 75–76 秒/次；这不是 2-NPU 聚合吞吐。

运行：在本目录执行 `bash run.sh`。默认共 20 次完整独立训练，每张 NPU 一个持久进程，使用两张 NPU；可设置偶数 `AIRBENCH_TOTAL_RUNS`。原有逐 epoch 验证、45 epochs、增强、优化器和最终 TTA 均保留。训练源码只增加数据路径设置。

交付前提：benchmark 默认镜像 `docker.cnb.cool/nilpotenter/docker/codeserver-mindspeed:v1.0.5` 中已配置兼容的 Python、torch 2.7.1、torch_npu 2.7.1.post2、torchvision 0.22.1 与 CANN；数据默认位于本目录上一级的 `data/cifar10/train.pt` 与 `test.pt`。若仓库数据位于其他目录，运行 `AIRBENCH_DATA_PATH=/绝对路径/cifar10 bash run.sh`。

入口不自动下载、安装依赖、编译或切换到主机专属环境。本项目无自定义 .so。迁移主机测试使用 `../verify_migrated.sh`，它配置迁移 Python 和 `../runtime_dependencies` 中隔离的 torchvision；此主机测试不等于已在干净 benchmark 镜像验证。

实际运行文件仅 `airbench96_faster_npu.py` 和 `run.sh`，无需 launch.py 或 worker.py。完整复跑主日志在 `logs/`；每次运行的子日志、源码快照和 metrics 在 `archive/evidence/<时间戳>/`，不会覆盖旧日志。历史原件与原训练日志在 `archive/source`、`archive/historical`。迁移差异记录在 `archive/source/migration.diff`；源码改动仅数据路径覆盖与运行次数设置。

# optimized — 两卡 DDP + local BN（32.66 s / 96.015%）

单次完整训练延迟 32.663 s（20 次验证：warm 2-20 均值 32.663 s，median 32.663，
min 32.543，max 32.806，std 0.065），TTA 精度均值 0.96015。
对照：同机 baseline（每卡独立单卡训练）单次 75.9-76.6 s、精度均值 0.9605。

用法（产物只有 logs/<时间戳>.log 一个文件）：

```bash
AIRBENCH_RUNS=20 bash run.sh              # 20 次顺序完整训练
AIRBENCH_DEVICES=0,1 AIRBENCH_RUNS=5 bash run.sh
```

实现要点（都在 airbench96_faster_npu.py 内，逐项实测）：

- 去掉每步对 inputs/labels 的冗余 `dist.broadcast`（两卡增强流本身逐位一致，
  由 `AIRBENCH_CHECK_INPUT=1` 的主机端逐元素校验证明，2160 步全通过）
- 局部 loss 取代 logits all-gather：各 rank 只算自己 256 样本的 logits，
  loss 本地求和后乘 world_size 补偿 DDP 的梯度平均
- proxy 生成器化并放到侧流、8 步预取，2160 步 proxy 不再阻塞主循环
- 广播 rank 0 的 512 个 hard-example 索引（4 KB）而非整批图像（3 MB），
  两卡 mask 严格一致（两卡各自的 proxy 会在 512/513 名边界差一个样本）
- FinalMaxPool 由两次 max 折叠为 flatten 后一次 max（输出与输入梯度逐位一致，
  含并列最大值；0.66 → 0.20 ms/次）
- `allow_internal_format` 全程开启；`gradient_as_bucket_view`；loader 空 cat 快路径；
  `_foreach_` lookahead；proxy 精简

已实测否决：MaxPool→reshape+amax（慢 30 倍）、FinalMaxPool→`max_pool2d(x,3)`（更慢）、
`allow_internal_format=False`（+33%）、`torch.compile(inductor/npugraph_ex)`（无法捕获
aclop conv，绕行方案即 +33%）、DDP `static_graph`（更慢）。

下界分析（见原目录 evidence_db/）：主循环 12.65 ms/step 中 gap A（kernel 效率）占 83%，
host 开销仅 4.2% 且已被异步流水线掩盖；本容器只有 2 张卡，17 s 目标在 Python 层不可达。

依赖：父目录的 `data/` 与 `runtime_dependencies/` 两个软链（torchvision 从后者来）。

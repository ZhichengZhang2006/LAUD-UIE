# 1.0 效率验收

工作簿 Sheet4 的 MambaCue 数值与 `efficiency_benchmarks/results/MambaCue.json` 的历史记录一致。历史效率环境为 school-gpu 的 NVIDIA A100-PCIE-40GB，HVI，PyTorch 2.4.0+cu121。该次测量加载 UIEB 权重，FP32、TF32 关闭、batch 1、RGB 256×256。先预热 10 次，再进行 30 次 CUDA 同步的前向计时，使用中位数，范围为模型前向。历史记录 SHA256 为 `77323621f1665d5bee3c1a8d13650d1b2243061ff54d8775517bf4e43a3d6c39`。

| 工作簿单元格 | 指标 | 数值 |
| --- | --- | ---: |
| Sheet4!B11 | THOP 统计参数量 | 1,866,193 |
| Sheet4!C11 | THOP 原始运算计数 | 18,334,791,200 |
| Sheet4!D11 | GPU 前向中位延迟 | 58.85527 ms |

Sheet1 另记 1.855M、164.49G 和 28.86 ms。现有档案已证实 Sheet4 的测量条件，正式效率表采用 Sheet4。成本栏沿用历史 THOP 原计数，另列候选实际参数量；THOP 对自定义扫描内核的覆盖可能不全。计时报告保存全部 30 次观测及测量前后的 GPU 占用。共享负载及硬件差异影响可比性，报告应说明实际测量环境。

UIEB 候选须通过完整训练预算及数据审计，之后由 `python -m luad_uie.scripts.profile_fusion_workbook RUN ARM UIEB_AUDIT WORKBOOK_REFERENCE OUTPUT_JSON` 评分。入口只加载候选权重，基线数值取自工作簿。

### 四方向批量推理

四个方向共享同一 Mamba 参数，且每条序列的递归状态各自独立。推理可将方向沿 batch 维合并为两次或一次 Mamba 调用，再恢复各方向坐标并求平均。状态字典、训练前向、梯度路径和默认推理路径均沿用现有实现。批量执行可能增加显存占用。

在 school-gpu2 A100 的 HVI 环境，加载 UIEB 主干权重并给 RGB 头设置非零参数。使用 256×256 单图及 61×65 双图核验，特征的最大绝对差为 1.91e-6，RGB 输出的最大绝对差为 1.49e-7。该测试验证批量实现的浮点一致性。正式权重可分别用 `--scan-batch 1`、`--scan-batch 2`、`--scan-batch 4` 测量延迟及前向额外峰值显存。正式测量入口会先检查当前权重的两条路径输出接近。

### 已训练权重的交错测量

使用七组实验中已完成的自由残差组第 100 轮检查点，三种模式共享同一模型实例和权重。用固定的 128 张 LSUI 验证图核对逐图输出。每种模式预热 10 次，以六种顺序循环交错计时，每种模式测量 60 次。测试设备为正在运行训练任务的 school-gpu2 A100，使用 FP32、关闭 TF32、batch 1、256×256 输入。

| 每次调用合并方向数 | 中位延迟 ms | P90 ms | 前向额外峰值显存 MiB | 128 图 RGB 最大绝对差 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 86.35 | 90.59 | 153.4 | 0 |
| 2 | 71.34 | 82.37 | 315.1 | 5.96e-7 |
| 4 | 68.72 | 73.38 | 539.6 | 6.56e-7 |

两方向和四方向合批的中位延迟分别下降 17.4% 和 20.4%。该比较衡量同一检查点的推理实现。按工作簿口径核验与 MambaCue 的速度差，仍需完成 UIEB 权重和匹配的测量环境。原始计时、顺序和检查点指纹见 [fusion_scan_epoch100.json](fusion_scan_epoch100.json)。

复现命令为 `python -m luad_uie.scripts.benchmark_fusion_scans FORMAL100_ROOT OUTPUT_JSON`。

### 两方向合批的峰值显存

两方向模式先计算和合并行扫描，再计算列扫描。合并后的行分量替代两条行输出在后续步骤中继续驻留。使用相同的第 100 轮权重、验证图和交错计时设置，对比提交 `658c3c2` 引入的原两方向实现。每种模式 60 次测量。

| 实现 | 中位延迟 ms | 前向额外峰值显存 MiB |
| --- | ---: | ---: |
| 逐方向 | 86.04 | 153.4 |
| 原两方向合批 | 69.93 | 314.1 |
| 分阶段两方向合批 | 70.58 | 283.1 |

分阶段计算较原合批降低 9.9% 峰值显存，中位延迟增加约 0.65 ms。它在本次 128 张图上与逐方向实现的 RGB 输出逐元素相等，浮点一致性结论适用于本次模型、图像和软件环境。见 [原始测量](fusion_scan_pair_memory.json)。

### CUDA Graph 固定形状试验

使用相同的第 100 轮权重和两方向合批，在共享 school-gpu2 A100 上分别测量普通前向和 CUDA Graph 重放。重放计时包括向固定 GPU 输入缓冲区拷贝图像。双方每种方式各 60 次，交替交换测量顺序。128 张验证图的输出逐元素一致。

普通前向的中位和 P90 延迟为 70.51 和 77.11 ms。图重放为 71.35 和 80.87 ms。逐轮配对时，图重放在 60 轮中有 33 轮较快，配对差的中位节省 0.80 ms。捕获用时 0.16 秒，PyTorch 额外保留约 318 MiB 显存。这次共享 GPU 试验未观察到稳定的延迟改善，正式测量使用普通合批实现。见 [原始测量](fusion_cuda_graph_epoch100.json)。

复现命令为 `python -m luad_uie.scripts.benchmark_fusion_cuda_graph FORMAL100_ROOT OUTPUT_JSON`。

### BF16 混合精度部署试验

相同第 100 轮 LSUI 权重、两方向合批、128 张验证图。比较 FP32、整网 BF16 自动混合精度和仅主干 BF16。后者以 FP32 执行 RGB 残差头。输入、输出质量计算均为 FP32。共享 A100 每种模式预热 10 次，交错计时 60 次。

| 模式 | 中位延迟 ms | 前向额外峰值 MiB | 平均 PSNR 差 dB | 最差单图 PSNR 差 dB | 平均 SSIM 差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 69.07 | 283.1 | 0 | 0 | 0 |
| 整网 BF16 | 55.91 | 161.7 | -0.00194 | -0.36570 | -0.0000944 |
| 主干 BF16 + FP32 RGB 头 | 55.99 | 161.7 | -0.00185 | -0.35229 | -0.0000771 |

仅主干 BF16 的运行时间较 FP32 低 18.9%，额外峰值显存低 42.9%。FP32 输出头只小幅减轻最差样本的 PSNR 下降。混合精度是可选部署配置，正式质量与工作簿效率比较采用 FP32。见 [逐图结果与原始计时](fusion_precision_epoch100.json)。

复现命令为 `python -m luad_uie.scripts.benchmark_fusion_precision FORMAL100_ROOT OUTPUT_JSON`。

在支持 BF16 的 CUDA 设备上，可用候选模型的常规推理入口导出图像。输出的 `metrics.json` 记录权重指纹、精度和扫描合批配置。第 100 轮权重已在 school-gpu2 上用该入口完成两张 256×256 PNG 的导出检查。

```sh
python -m luad_uie.scripts.evaluate --checkpoint CHECKPOINT --manifest INPUT_CSV \
    --output OUTPUT --inference-only --save-images --precision bf16-backbone --scan-batch 2
```

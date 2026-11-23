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

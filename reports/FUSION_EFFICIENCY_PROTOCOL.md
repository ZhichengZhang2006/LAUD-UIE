# 1.0 效率验收

工作簿 Sheet4 的 MambaCue 数值与 `efficiency_benchmarks/results/MambaCue.json` 的历史记录一致。历史效率环境为 school-gpu 的 NVIDIA A100-PCIE-40GB，HVI，PyTorch 2.4.0+cu121。该次测量加载 UIEB 权重，FP32、TF32 关闭、batch 1、RGB 256×256。先预热 10 次，再进行 30 次 CUDA 同步的前向计时，使用中位数，范围为模型前向。历史记录 SHA256 为 `77323621f1665d5bee3c1a8d13650d1b2243061ff54d8775517bf4e43a3d6c39`。

| 工作簿单元格 | 指标 | 数值 |
| --- | --- | ---: |
| Sheet4!B11 | THOP 统计参数量 | 1,866,193 |
| Sheet4!C11 | THOP 原始运算计数 | 18,334,791,200 |
| Sheet4!D11 | GPU 前向中位延迟 | 58.85527 ms |

Sheet1 另记 1.855M、164.49G 和 28.86 ms。现有档案已证实 Sheet4 的测量条件，正式效率表采用 Sheet4。成本栏沿用历史 THOP 原计数，另列候选实际参数量；THOP 对自定义扫描内核的覆盖可能不全。计时报告保存全部 30 次观测及测量前后的 GPU 占用。共享负载及硬件差异影响可比性，报告应说明实际测量环境。

UIEB 候选须通过完整训练预算及数据审计，之后由 `python -m luad_uie.scripts.profile_fusion_workbook RUN ARM UIEB_AUDIT WORKBOOK_REFERENCE OUTPUT_JSON` 评分。入口只加载候选权重，基线数值取自工作簿。

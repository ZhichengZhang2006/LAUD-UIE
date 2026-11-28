# LAUD-UIE 1.0 · MambaCue 特征主干融合

1.0 使用给定的 MambaCue 编码器和解码器及其 UIEB EMA 初始化。原始发布见 [`1.0`](https://github.com/ZhichengZhang2006/LAUD-UIE/tree/1.0)。[`release/1.0`](https://github.com/ZhichengZhang2006/LAUD-UIE/tree/release/1.0) 收录同一版本后续的完整训练、恢复工具、评价入口和本说明。

## 候选创新点

1. **整图偏移与空间残差**。场景头估计逐通道的 RGB 偏移，中心化版本将空间修正约束为零均值。Free 是同结构的自由残差对照。
2. **共识亮度顺序约束**。仅在输入和配对参考的亮度顺序一致且对比足够的位置监督输出，覆盖 1、2、4 像素间距。
3. **效用加权的分量蒸馏**。分别测量学生输出向冻结教师靠近时，整图均值和空间残差的参考误差下降量，再由正效用分配蒸馏权重。见 [机制与证据](reports/FUSION_CONTRIBUTION_EVIDENCE.md)。

## 实验结果

**LSUI 开发集。** 下表是相同 512/128 训练/验证划分、seed 42 的五个标准组在第 100 轮的验证结果。后续拟合初始化对照需要独立解读。

| 方法 | 最后一轮验证 PSNR ↑ dB | 验证 SSIM ↑ |
| --- | ---: | ---: |
| Free | 27.0878 | 0.92098 |
| Centered | 26.7356 | 0.91743 |
| Ordinal | 26.7219 | 0.91734 |
| Fixed KD | 26.7891 | 0.91798 |
| Utility KD | 26.7231 | 0.91737 |

Free 在这五组的最终轮验证 PSNR、SSIM 上最高。使用更多 LSUI 训练数据的 Free 对照同样完成 100 轮，在同一验证集达到 29.4549 dB 和 0.93788 SSIM，训练图由 512 增至 2,711，每轮更新次数也随之增加。

**UIEB Free 200 轮。** 固定 UIEB 划分 712/88/89，256×256。选定权重 `90890f7ea1b25cd82a0df411307b66e159fdb20e5898ecd84c9a83b14a1b8bdd` 在 89 对 UIEB 测试图上得到 PSNR **23.4445 dB**、SSIM **0.90958**、LPIPS **0.12226**、ΔE00 **8.5190**、UCIQE **0.39341**。同一权重用于 CH60 和 U45。

| 无参考集 | 图数 | UIQM ↑ | UCIQE ↑ | URanker ↑ | Entropy ↑ | NIQE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CH60 | 60 | 3.54376 | 0.38147 | 1.55151 | 7.30782 | 5.87629 |
| U45 | 45 | 3.48581 | 0.41459 | 1.87880 | 7.59368 | 4.12670 |

以上 UIEB 数据来自 Free 候选，三项新增机制的收益仍待进一步对照。历史完整 MambaCue 的对照值取自 [UIE-data.xlsx 的仓库快照](reports/mambacue_workbook_reference.json)，需核对数据划分和指标几何后才能比较。日志和结果源位于 `school-gpu2:/home/user/luad_fusion_1_0/{formal100,uieb_free_200_scores}`。见 [训练与运行说明](reports/BACKBONE_FUSION_1.0.md) 和 [UIEB 与效率记录](reports/FUSION_UIEB_AND_EFFICIENCY_2026-11-24.md)。

## 复现

HVI 环境需要兼容的 PyTorch、CUDA、`mamba_ssm` 和主干包中的预训练权重。

```bash
python -m luad_uie.scripts.train_backbone_fusion TRAIN_CSV VAL_CSV \
  BACKBONE_WEIGHTS RUN_DIRECTORY --epochs 100
```

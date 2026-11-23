# 有参考数据集验收

训练候选在七组 LSUI 实验各满 100 轮并通过检查点核验后冻结。性能表记录候选的微调和预训练来源、选定轮次、权重 SHA256、样本数及工作簿来源单元格。LSUI 模型在 LSUI、UIEB、UFO-120、EUVP-Scene 和 EUVP-Dark 上使用 256×256 RGB 图评估，报告 SSIM、PSNR、LPIPS-VGG、ΔE00 和 UCIQE。UIEB 微调模型的 UIEB 本域测试另由 `score_fusion_uieb` 记录。

`prepare_fusion_paired` 将 LSUI 的完整开发清单和原 UIEB 预训练的训练及验证清单纳入重叠审计。用文件 SHA256 和原尺寸解码 RGB 像素的 SHA256 检查输入与参考，保守剔除与任一开发清单重合的测试对。剔除后记录原清单和实际清单的指纹。覆盖率不足的数据集只将候选的有效子集成绩与工作簿原记录并列展示。

| 数据集 | 原清单 | 重叠剔除 | 保留 |
| --- | ---: | ---: | ---: |
| LSUI | 428 | 0 | 428 |
| UIEB | 89 | 0 | 89 |
| UFO-120 | 120 | 56 | 64 |
| EUVP-Scene | 219 | 0 | 219 |
| EUVP-Dark | 555 | 0 | 555 |

`score_fusion_paired` 对审计后的测试图逐张验证文件指纹。LPIPS 使用既有 VGG v0.1 离线权重。工作簿缺值的指标保留为空参照。即使覆盖率完整，也应核对工作簿的指标实现和图像协议后再计入胜率。当前评分程序只执行候选模型。

审计目录为 `/home/user/luad_fusion_1_0/paired_audit`。待 LSUI 权重按验证集方案冻结后，在 HVI 环境运行下列命令。LPIPS 依赖和 VGG 权重沿用 UIEB 评分环境。

```sh
python -m luad_uie.scripts.score_fusion_paired formal100 SUITE/ARM \
    paired_audit reports/mambacue_workbook_reference.json candidate_paired
```

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

质量验收沿用固定的 10 项主要指标。LSUI 的五项全参考指标各占一项，CH60 与 U45 各指标按两个数据集等权平均，再占五项。跨数据集测试和 DUO、SIFT、效率分别报告。达到 80% 要求至少 8 项经同口径核验的严格胜出。工作簿缺少 LPIPS 和 ΔE00 的历史值，两项候选实测保留，并计入 10 项分母。

`summarize_fusion_workbook` 读取有参考汇总、UIEB 无参考汇总及同一份工作簿参照。可通过 `--protocol-evidence` 提供以 `LSUI/PSNR` 或 `CH60_U45/NIQE` 等指标键映射到原始协议来源说明的 JSON 文件。只有样本完整且有口径核验记录的数值差才计入正式胜负。若已确认胜出不足 8 项且仍可能达到门槛，整体结论保留为待核验。

Sheet2 的 CH60 数值与历史 `no_reference_results/CH60_from_UIEB.txt` 一致。该记录对应 UIEB 的 NoHead 检查点。U45 数值与 `no_reference_results/CHECKPOINT_SWEEP_RESULTS.md` 的完整 UIEB 检查点记录一致。原无参考入口 `test_nr.py` 通过 `utils/dataset.py` 将短边缩至 256，保留纵横比并裁至 4 的倍数。当前候选使用固定 256×256。这个尺寸差异适用于两个无参考数据集，应将它们与工作簿的差值标为数值对照，暂不计作同口径胜负。

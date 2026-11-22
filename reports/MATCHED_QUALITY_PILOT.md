# LSUI 匹配训练筛选

首轮门控学生相对于 MambaCue 历史权重的质量胜率为 0/10。在验证集上，逐图拟合旧输出头也能达到较高的 PSNR，因此本轮先对齐模型看到的整图内容，再检验直接监督修正场是否能改善训练。

使用已清理的 LSUI 清单，按种子 42 和图像 ID 的 SHA256 固定选择 1,536 张训练图与 128 张验证图。所有训练臂使用相同整图 256×256 双线性缩放、翻转规则、batch size 2、12轮、AdamW 初始学习率 2e-4、余弦降低至 2e-5。根据统一验证集 PSNR 取最佳轮。每组预计进行 9,216 次优化器更新。

| 训练臂 | 结构 | 训练目标 |
|---|---|---|
| 尺度重排 + RGB | 轻量单头学生 | Charbonnier + SSIM |
| 尺度重排 + 频段监督 | 同一个轻量单头学生 | 相同基线损失，加块均值色度误差和去均值的逐像素 RGB 残差误差 |
| 匹配 MambaCue | 服务器 MambaCue，随机初始化 | 与第一组相同的 Charbonnier + SSIM |

两个轻量臂的结构和初始化相同，推理阶段都使用同一个输出头。训练目标将 2×2 块的平滑色度和块内细节分别作为监督。已有 WF-Diff 使用小波、傅里叶特征和频率扩散，已有 WPFNet 融合小波与像素域。频段分解本身已有先例。本轮只检验在一个共享输出头中直接监督有符号、零均值的像素残差是否改善轻量模型。对照结果齐备后，再判定是否保留该方向。[WF-Diff](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Wavelet-based_Fourier_Information_Interaction_with_Frequency_Diffusion_Adjustment_for_Underwater_CVPR_2024_paper.pdf) · [WPFNet](https://www.sciencedirect.com/science/article/pii/S095070512400683X)

该实验使用 HVI，因为第三组必须运行 MambaCue 的 mamba_ssm 算子。三个训练臂在同一进程内依次运行。4 张训练图、4 张验证图的试跑均已通过，三个模型各实际更新 2 步。正式流程已开始，先运行 RGB 学生，再运行频段监督学生，随后训练新 MambaCue。测试图不进入筛选代码。正式验收继续使用 LSUI、U45 与 CH60 的固定十项指标、RUOD 检测和 SUIM 边缘定位，并与原 MambaCue 历史权重比较。短预算同条件结果用于解释结构和训练因素。

服务器目录为 `/home/user/luad_uie_prototype_20261031/stages/packed_20261122`。训练记录位于 `matched_quality.log` 和 `matched_quality/<arm>/epochs.jsonl`，数据 ID、清单哈希和 MambaCue 源码哈希见 `matched_quality/provenance.json`。

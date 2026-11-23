# LUAD-UIE 第二阶段 · 双视图局部结构监督

在配准的训练对中，从原图、参考和输出提取亮度。在水平、垂直方向比较相距 1、2、4 个像素的两个位置。当原图和参考的明暗次序一致、两边的亮度差均达到 0.02，才将这对位置用于辅助监督。输出沿共同次序的亮度差达到两张训练图较小对比度的一半时，结构损失为零。常规重建损失仍覆盖整图。推理图和模型结构沿用第一阶段。

正式比较从第一阶段验证集选定的同一个冻结权重启动，三组使用相同的新优化器状态和数据顺序。三组分别为原损失、加 0.2 多尺度全图梯度差和加 0.2 共识顺序约束。若选定的第一阶段权重带分量监督，三组一并保留。验证记录被保护的位置比例、违反最小对比度的比例、PSNR、SSIM 和两个分量的误差。SIFT 检出数量和几何重复率是检验是否改善可定位结构的外部指标。

入口为 `python -m luad_uie.scripts.train_matched_quality TRAIN_CSV VAL_CSV OUTPUT --width 40 --base-arm SELECTED_SCENE_ARM --initial-checkpoint SELECTED_BEST_PT --arms structure_reconstruction structure_gradient structure_ordinal`。CDF_FFT 中的 13 项 CPU 回归检查通过。在 school-gpu2 的 HVI 环境，使用四张 LSUI 训练图和四张验证图检查三组流程，每组完成两次更新。此记录用于工程核对，正式结构对照在第一阶段选定起点后开展。

验证集 SIFT 用 `python -m luad_uie.scripts.score_validation_structure VAL_CSV OUTPUT_JSON CHECKPOINT_A CHECKPOINT_B CHECKPOINT_C`。复用正式评估的输入尺寸、旋转缩放、共同有效区和匹配阈值。样本为固定的 LSUI 验证集，避免使用 DUO 测试结果选择训练参数。

## 正式对照

以第一阶段零均值残差组第 11 轮为共同起点，复用去重 LSUI 的 1,536 张训练和 128 张验证子集，宽度 40，种子 42，每组 12 轮。历史 MambaCue 在相同验证集上复评。服务器目录为 `/home/user/luad_uie_prototype_20261031/stages/ordinal_stage_20261123`，正式训练记录为 `ordinal_pilot.log`。

开发集评分流程已用三份小样本权重在四张图上检查。每组两种变换共 8 对，全部可计算匹配。正式第二阶段以新权重的验证集 SIFT 和恢复质量为依据。

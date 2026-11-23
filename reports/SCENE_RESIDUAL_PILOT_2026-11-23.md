# LUAD-UIE 第一阶段 · 整图校正与空间残差

此前宽度 40 的整图上下文学生已完成 100 轮。LSUI 的 425 张验证图上，第 100 轮为 27.6970 dB PSNR、0.92336 SSIM。冻结最佳权重的独立验收在十项质量指标中胜出 0 项。在同设备的批量 1、FP32、256×256 GPU 前向测量中，中位值为 0.689 ms，MambaCue 为 58.230 ms。这两项速度在共享 A100 上测得。既有 RUOD 和 SUIM 结果作为下游诊断，DUO 和 SIFT 沿用目标约定单独评估。

## 三组受控实验

全部用宽度 40、整图上下文、一个 15 通道输出投影和相同的解码器。输出每个可见像素的 RGB 残差由整图偏移与空间场相加，再截断到 [0,1]。新增的三个全局通道在空间求均值后构成整图偏移。

| 组别 | 空间场 | 损失 |
| --- | --- | --- |
| scene_unconstrained | 自由空间残差 | Charbonnier + 0.2 SSIM 损失 |
| scene_centered | 逐图逐通道扣除空间均值 | 同上 |
| scene_supervised | 与 scene_centered 相同 | 同上，另对参考减输入的均值和中心化残差各加 0.25 L1 |

空间中心化在裁去补边后、输出截断前执行。模型的参数量与初始权重在三组间一致。每张验证图记录最终输出的整图偏移 MSE、中心化空间 MSE 和饱和像素比例，也报告 PSNR、SSIM 和总 MSE。两项分解 MSE 之和等于最终输出的总 MSE。预期判据是整图偏移误差下降，同时避免把误差转移到空间项。

使用去重的 LSUI 划分，从固定的 train 与 val 清单按 ID 和种子 42 取 1,536 张训练图及 128 张验证图。使用 256×256 整图、batch 2、12 轮、同批次顺序、AdamW 和余弦学习率。历史 MambaCue 权重在相同验证子集重测，保留权重哈希。模型最佳轮次按验证 PSNR 选择。单种子用于筛选，晋级版本和同架构对照再在其他种子复验。

## 复现

入口为 `python -m luad_uie.scripts.train_matched_quality TRAIN_CSV VAL_CSV OUTPUT --epochs 12 --width 40 --arms scene_unconstrained scene_centered scene_supervised --historical-mamba`。新增检查覆盖偶数、奇数及补边输入、零初始化、对两个分量反传、参数匹配和验证误差的分解。CDF_FFT 的 11 项 CPU 检查通过。CDF_FFT 的新进程 CUDA 初始化返回 Error 4，本轮由于包含 MambaCue 历史权重对照，使用可运行 CUDA 的 HVI 执行。

服务器目录为 `/home/user/luad_uie_prototype_20261031/stages/scene_stage_20261123`，训练日志为 `scene_pilot.log`。独立结构监督及分量蒸馏留给后续阶段，依次以晋级结构为共同对照。

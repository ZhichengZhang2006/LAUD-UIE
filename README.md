# LAUD-UIE 2.2

LAUD-UIE 是一个水下图像增强研究项目。2.2 保留既有四级 Mamba 编码器和三级解码器，修改恢复头和训练监督。所有 2.2 训练、验证和测试输入均为 RGB 256×256。

## 版本

| 版本 | 代码与 README | 实验范围 |
| --- | --- | --- |
| 2.2 | 当前分支 · 归档标签 [`2.2`](https://github.com/ZhichengZhang2006/LAUD-UIE/tree/2.2) | LSUI 三模块消融，后续五域长训练与八项测试 |
| 1.0 | [`release/1.0`](https://github.com/ZhichengZhang2006/LAUD-UIE/tree/release/1.0) · [版本说明](README.1.0.md) · [原始标签](https://github.com/ZhichengZhang2006/LAUD-UIE/tree/1.0) | MambaCue 特征主干融合、分量蒸馏与部署 |
| 0.1 | [`0.1`](https://github.com/ZhichengZhang2006/LAUD-UIE/tree/0.1) · [版本说明](README.0.1.md) | E 系列多视图蒸馏原型 |

## 2.2 的三个候选创新点

1. **自适应软光度子空间恢复（ASPR）**。在局部窗口中用输入 RGB 和常量项构造光度子空间，再用可学习门控抑制该子空间可解释的残差。普通仿射头 AFF 用于隔离软投影的作用。
2. **跨水体算子复合一致性（COC）**。对合成通道仿射扰动，约束两视图的颜色矩阵、偏移和残差按右复合关系变化。包括 M0 在内，各组接受相同双视图恢复监督。
3. **光度隔离的对应关系蒸馏（PICD）**。用配对参考图构造匹配分布及正负间隔，对齐两个增强视图的局部对应关系。该损失的直接梯度经残差和门控路径传播，共享主干仍参与更新。

## 实验结果

**2026-11-28 20:36（北京时间）快照。** LSUI 训练/验证包含 2,711/128 对，seed 42，各组从随机初始化训练 300 轮。训练使用 BF16 主干、FP32 恢复头和损失、微批 4、梯度累积 2。每五轮用 FP32 评估 EMA 权重。候选须达到峰值验证 PSNR 减 0.15 dB 的门槛，再按预设组合分数选取一份权重。以下为各完成组该权重在同一验证集的逐图均值。

| 组别 | 完成轮数 | 选定轮次 | PSNR ↑ dB | SSIM ↑ | LPIPS ↓ | ΔE00 ↓ | UCIQE ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M0（Free） | 300 | 300 | 30.0786 | 0.93598 | 0.12379 | 4.7932 | 0.37349 |
| AFF（仿射） | 300 | 295 | 29.9928 | 0.93715 | 0.12179 | 4.6673 | 0.37312 |
| A（ASPR） | 300 | 240 | 30.0237 | 0.93646 | 0.12296 | 4.8016 | 0.37379 |
| AB（ASPR + COC） | 300 | 255 | 29.6900 | 0.93407 | 0.12735 | 4.9933 | 0.37511 |
| AC（ASPR + PICD） | 38 / 300 | — | — | — | — | — | — |
| ABC（三模块） | 排队 | — | — | — | — | — | — |

A 相比 AFF 提高 0.031 dB，但 SSIM、LPIPS 和 ΔE00 较弱。AB 相比 A 的 UCIQE 上升，另四项变差。PICD 的有效性、完整模型的组合效果与跨数据集性能仍待完成实验。五域长训练及八项正式测试尚未开始。以上是单种子 LSUI 验证数据，完整 LSUI 测试保留至正式评估。

结果来源为 `school-gpu2:/home/user/laud_v22_20261125/runs/LSUI_ablation/{组别}_s42/selection.json` 及 `train_history.json`。版本发布采用已部署的训练源文件，现有实验继续记入该目录。

## 代码和复现

| 入口 | 用途 |
| --- | --- |
| [`laud_v22/model_v22.py`](laud_v22/model_v22.py) 与 [`losses_v22.py`](laud_v22/losses_v22.py) | 软投影、恢复头、双视图损失和训练期开关 |
| [`data_v22.py`](laud_v22/data_v22.py) 与 [`metrics_v22.py`](laud_v22/metrics_v22.py) | 固定尺寸清单读取、重复检查与原指标接口 |
| [`run_lsui.sh`](laud_v22/run_lsui.sh) 与 [`train_v22.py`](laud_v22/train_v22.py) | 首轮六组流水线、EMA、选模和断点恢复 |
| [`freeze_formal.py`](laud_v22/freeze_formal.py)、[`make_configs.py`](laud_v22/make_configs.py)、[`evaluate_v22.py`](laud_v22/evaluate_v22.py) | 冻结五域清单、生成配置、按训练域路由八项测试 |

使用 Python 3.10+、可运行 `mamba_ssm` 的 CUDA 环境，以及 LPIPS、Pillow、scikit-image 和原仓库的指标资源。`run_lsui.sh` 使用学校服务器的 HVI 环境和已有数据，可从每组最后提交的 epoch 继续。其他机器可自行生成同格式配置。每个有参考 CSV 的列为 `id,input,target`，无参考 CSV 使用 `id,input`，相对图像路径以 CSV 所在目录为基准。

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python laud_v22/make_configs.py --data-root MANIFESTS --runs-root RUNS \
  --metric-repo MAMBACUE_METRICS_REPO --output configs_v22
python laud_v22/train_v22.py --config configs_v22/LSUI.json \
  --arm M0 --epochs 300 --seed 42 --output runs/LSUI_ablation/M0_s42
```

正式评价使用 `python laud_v22/evaluate_v22.py --suite configs_v22/suite.json --output runs/eval_ABC`。C60 和 U45 使用 UIEB 的同一份权重，EUVP 无参考集使用 EUVP-s 权重。数据、训练权重和日志存放在服务器 `/home/user` 下。

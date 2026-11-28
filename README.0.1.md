# LAUD-UIE 0.1 · 多视图蒸馏原型

0.1 是对最早一套具备完整 E 系列代码及配对 LSUI 结果的提交的归档版本号。采用两级 U-Net，教师宽度 48，学生宽度 24，教师冻结。推理只运行学生。

## 候选创新点

- 将单教师的 RGB 恢复信号拆成色度、低频亮度和一阶细节三种视图，对照单视图和三视图等权蒸馏。
- 以重建头梯度内积、余弦和虚拟 SGD 一步估计监督效用。E4 学习预测每图效用，E5 增加拒绝阈值。后续归档代码也包含逐图梯度路由入口。

## LSUI 实验结果

100 轮，seed 42，428 对历史测试图。教师和学生分别使用验证 PSNR 选择权重。以下是该历史协议的单次测试结果。

| 方法 | 测试 PSNR ↑ dB | 测试 SSIM ↑ |
| --- | ---: | ---: |
| 教师（宽度 48） | 23.5522 | 0.877228 |
| E0 纯恢复学生 | 22.9671 | 0.869489 |
| E1 色度 | 23.1429 | 0.871466 |
| E1 亮度 | 23.0950 | 0.870358 |
| E1 细节 | 23.1661 | 0.870718 |
| E2 三视图等权 | 23.0613 | 0.869698 |

E2 较同宽度 E0 提高 0.0942 dB，但约 45.56% 的测试图下降，且平均 PSNR 低于三种单视图。动态控制的收益仍待验证。旧报告中个别标作 E0 的逐图比较实际使用了宽教师，汇总以修正后的同宽度学生对照为准。见 [第一轮实验报告](reports/LUAD_UIE_experiment_report_2026-11-12.md) 与 [配对 E0 对照](reports/PROGRESSIVE_FRAMEWORK_2026-11-16.md)。

## 复现

使用 CDF_FFT 环境，配置在 [`configs/`](configs/)。配对 RGB 图像使用 `id,input,target` CSV，训练采用 256×256 成对裁剪，验证和测试保持原尺寸。蒸馏组须在配置中提供冻结教师 checkpoint。

```bash
python -m pip install -e '.[test]'
python -m luad_uie.scripts.train --config configs/E0.yaml --output runs/E0_seed42
python -m luad_uie.scripts.train --config configs/E2.yaml --output runs/E2_seed42
python -m luad_uie.scripts.evaluate --checkpoint runs/E2_seed42/best.pt \
  --manifest data/test.csv --output runs/E2_test
```

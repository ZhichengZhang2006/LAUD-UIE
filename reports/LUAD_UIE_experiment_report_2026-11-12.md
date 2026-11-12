# LUAD-UIE 第一轮固定视图实验报告（完成版，2026-11-12）

本报告取代2026-11-11报告中“E2仍训练”的状态；旧版本保留。数据来自服务器实际配置、完整epoch日志、best.pt及428对测试结果。当前已完成100轮教师和三种E1、E2，仍缺同规模学生E0。结论只覆盖单随机种子固定视图，不构成学习效用控制有效性的证据。

## 实验方案

LSUI使用服务器已有train/val/test目录，样本为3423/428/428对。尚未核对作者官方名单，尚无内容近重复/场景分组审计，不能宣称官方划分或场景级无泄漏。图像为RGB [0,1]；训练256x256随机成对裁剪、水平/垂直翻转，小图复制边界；验证与测试原尺寸batch1。val选择best.pt，test不参与优化。

教师为两级U-Net width48，1,059,939参数；学生同结构width24，265,587参数（少74.94%）。卷积块是两层3x3+ReLU，平均池化下采样，双线性上采样+跳接，1x1头与sigmoid输出。教师固定eval和no_grad；推理只运行学生。当前无CLIP、Mamba、多教师或特征KD。

全部运行seed42、100epoch、Adam lr=2e-4、batch4，无学习率调度；每轮856步，总85600步。恢复损失为Charbonnier epsilon=1e-3加0.2*(1-SSIM)。SSIM是RGB Gaussian11x11 sigma1.5 valid，范围1；PSNR/MSE/SSIM逐图计算再平均。

color是D65 Lab a/b除以128后的L1；illumination是加权RGB亮度9x9均值低通后的L1；detail是亮度横纵一阶差分L1平均。E1仅选一个视图，L=Lrec+0.1*Lview；E2各1/3，L=Lrec+0.1*(Lc+Ll+Ld)/3。等权不代表相等梯度贡献，尚无尺度校准或lambda敏感性结论。

所有KD均用同一个已训练100轮教师best.pt，未用oracle。当前效用划分为null，E3/E4/E5尚无真实数据结果。运行在school-gpu2及school-gpu的CDF_FFT环境，PyTorch2.1.0+cu118；服务器文件均在/home/user。detail40轮后中断并恢复模型/优化器/步数，未保存随机数与DataLoader完整状态，因此不等同不中断轨迹。

## 完成结果

验证结果对应按验证PSNR选择的同一个best checkpoint；epoch从1计数。

| 任务 | 完成轮数 | best轮数 | val PSNR | test PSNR | test SSIM | test MSE |
|---|---:|---:|---:|---:|---:|---:|
| LSUI_teacher_seed42 | 100 | 99 | 23.4192 | 23.5522 | 0.877228 | 0.006924 |
| E1_color_LSUI_seed42 | 100 | 100 | 23.0482 | 23.1429 | 0.871466 | 0.007414 |
| E1_illumination_LSUI_seed42 | 100 | 99 | 22.9941 | 23.0950 | 0.870358 | 0.007471 |
| E1_detail_LSUI_seed42 | 100 | 94 | 22.9919 | 23.1661 | 0.870718 | 0.007500 |
| E2_LSUI_seed42 | 100 | 99 | 22.9353 | 23.0613 | 0.869698 | 0.007509 |

E2最终测试PSNR为23.0613dB，相比color低0.0816dB，相比illumination低0.0337dB，相比detail低0.1048dB；SSIM也低于三个E1。本次单种子配置下，三视图简单平均未表现出超过任一单视图的收益。这不能推断其他lambda或随机种子下等权必然失败，也不能证明动态控制器有效。

detail平均PSNR最高，但color的SSIM更高且MSE更低。不同指标排序不同。三种E1均低于较宽教师，不能据此判断KD负迁移。当前没有同规模E0，无法断言蒸馏优于或差于纯恢复学生。

## 逐图对照与错误标注纠正

旧metrics.json的degraded_fraction_vs_E0和delta_psnr_vs_E0实际比较的是width48教师，因为历史流水线将教师metrics传为baseline，教师配置恰好标签E0。该值不代表相对width24学生E0的负迁移率。旧文件保留用于追溯，报告按实际比较对象解释。新阶段评估将检查基线宽度、seed、预算、恢复目标和train/val清单。

| 任务 | 测试PSNR低于教师比例 |
|---|---:|
| E1 color | 75.23% |
| E1 illumination | 79.44% |
| E1 detail | 67.29% |
| E2 | 76.40% |

同一428对test manifest SHA256为9e9450141d28a6a547a45f6e6f8422dc85739ad03b30587cad091e12fd0db673。快照提供配置、完整history、manifest和checkpoint及源码哈希，可追溯采集时文件，不追溯证明全程源码未改。

## 开销、局限与下一步

- LSUI_teacher_seed42：日志训练时间合计20.57小时，峰值allocated显存4.84GiB，记录单图推理10.87ms。
- E1_color_LSUI_seed42：日志训练时间合计32.82小时，峰值allocated显存2.50GiB，记录单图推理12.53ms。
- E1_illumination_LSUI_seed42：日志训练时间合计36.80小时，峰值allocated显存2.50GiB，记录单图推理9.97ms。
- E1_detail_LSUI_seed42：日志训练时间合计37.06小时，峰值allocated显存2.50GiB，记录单图推理12.00ms。
- E2_LSUI_seed42：日志训练时间合计32.74小时，峰值allocated显存2.50GiB，记录单图推理11.26ms。

训练时间不含验证/保存/中断等待，不能当作独占GPU小时；共享服务器、跨服务器、原图尺寸和单次计时均影响延迟。当前耗时仅记录，不用于宣称速度优势。没有多种子、独立外部数据测试、置信区间或定性图像审查结果。

下一步依照NEXT_EXPERIMENT_PLAN_2026-11-12.md：先补齐width24同配置100轮E0并重新计算对照；本轮E0/E1/E2完成后按既定顺序在独立目录将教师从100轮续训到累计250轮（新增150轮），保留旧教师。之后建立R2训练内效用与独立审计划分，检验代理内积/余弦与真实虚拟一步收益的关系；只有有证据时才增加预测控制及梯度尺度校准消融。发生CUDA OOM停止当前项及依赖流水线，不自动降低配置重试。

参考思路来自Ren等ICML2018样本重加权、Chen等ICML2018 GradNorm、Stanton等NeurIPS2021 KD分析；详见预注册方案的原文链接。这些工作提供方法依据，不构成水下增强收益证据。

当前可支持的结论是：固定教师多视图原型可训练，单次配置中等权E2不如三个E1；动态学习效用假设仍待验证，结果不保证成功。

## 文件与复现

服务器根目录/home/user/luad_uie_prototype_20261031/。训练位于runs/{LSUI_teacher_seed42,E1_color_LSUI_seed42,E1_illumination_LSUI_seed42,E1_detail_LSUI_seed42,E2_LSUI_seed42}/；测试位于对应*_test/metrics.json。快照reports/2026-11-12/snapshot.json，本报告同目录。新代码阶段位于stages/20261112_baseline/，GitHub https://github.com/ZhichengZhang2006/LAUD-UIE 。

数据快照UTC：2026-11-12T02:02:10.403357+00:00

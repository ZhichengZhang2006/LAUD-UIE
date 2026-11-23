# 工作簿的应用测试口径

| 工作簿 | 原始存档匹配 | 当前候选如何比较 |
| --- | --- | --- |
| Sheet5 D11:E11，mAP50 0.6205、mAP50:95 0.4196 | 原始 DUO 存档为 0.6205118、0.4196017。使用固定的 320 张训练图与 80 张测试图，每种增强结果各训练一个 YOLOv5s v7.0 检测器，COCO 初始化，300 轮，batch 16，检测输入 640，种子 0，划分种子 42。 | 保留原划分与检测训练配方，候选增强输入使用约定的 256×256，检测器输入尺寸单独记录。较早的 1,111 图固定检测器实验使用不同协议，成绩仅能并列展示。 |
| Sheet6 B11，平均 350.67 个 SIFT 关键点 | CH60 原始存档为每图 350.6667 个。60 张图先用 OpenCV INTER_AREA 缩放到 256×256，再增强、量化为 8 位 RGB，并以 OpenCV 默认 SIFT 在灰度图上计数。历史模型使用 LSUI 权重。 | 使用完成 UIEB 正式训练的候选权重，在相同 60 张图、缩放、量化和计数方式下测量，披露双方训练数据来源。U45 和 DUO 的 SIFT 指标另有各自的样本集合。 |

来源文件为旧项目的 downstream/results/results_bundle.json、downstream/results/ch60_results_bundle.json、downstream/run_ch60.py、downstream/infer.py 与 downstream/features.py。MambaCue 对照数值取自 UIE-data.xlsx。较早项目中的重测结果属于历史记录。

候选的 CH60 SIFT 结果由 `score_fusion_uieb` 写入 `ch60_sift.json`，含图像 ID、逐图关键点数、权重指纹、OpenCV 版本和工作簿单元格。正式评分要求完成至少 100 轮且通过 UIEB 训练及图像清单校验。

## 候选 DUO 流程

`export_fusion_duo` 先核验七组 LSUI 的完整训练记录和所选权重，然后按服务器 `duo_workbook_manifest.json` 导出 400 张 RGB PNG。增强输入为 OpenCV INTER_AREA 直接缩放的 256×256 图。只做缩放时，归一化检测框坐标保持原值。导出清单记录候选权重、每张增强图和每份标签的指纹。

`run_fusion_duo_workbook.sh` 使用历史归档中的 YOLOv5 v7 与 `eval_yolo.py`，后者在服务器保存为 `eval_yolo_workbook.py`。脚本核验初始化权重与评估器的 SHA256，训练 300 轮后才读取候选检测器的测试指标。训练中的验证路径指向训练图像，80 张测试图在最终独立评分时使用。增强器的 256 输入与检测器的 640 输入分别记录。与历史分数并列展示时，需说明历史增强阶段保留图像比例，而新版增强阶段采用 256×256 直接缩放。

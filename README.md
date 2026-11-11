# LUAD-UIE PyTorch 原型

根据本目录的 `LUAD_UIE.md` 实现（用户所述 `LUAD_UIE_初步框架设计.md` 未以该文件名出现）。这是可训练的软件原型，**不是已验证的方法、创新性结论或 SOTA 结果**。

## 已实现

| 模块 | 行为与边界 |
|---|---|
| Student | 轻量两级 U-Net；默认宽度 24，RGB sigmoid 输出；支持奇数图像尺寸；输出 pooled 中间特征 |
| Teacher | 单个冻结、固定 eval、no_grad 的恢复模型；严格加载 checkpoint；内置相同 U-Net 适配器；允许显式 oracle 调试 |
| 三知识视图 | D65 Lab a/b 色度 L1（除以 128）；亮度 9×9 低频 L1；亮度水平/垂直梯度 L1；返回逐样本损失 |
| 恢复目标 | Charbonnier + 0.2×(1−SSIM)；SSIM 为 RGB、Gaussian 11×11、sigma=1.5、valid 窗口、range=1 |
| E0/E1/E2 | 无蒸馏 / 单视图（3 配置）/ 三视图各 1/3；统一学生、增强、优化器及指标 |
| E3 | 固定颜色偏移、亮度偏离、梯度不足启发式；没有 CLIP，也不是物理退化估计 |
| E4 | 训练内效用监督、退化统计+detach 学生状态 MLP、正效用归一化；全部非正可拒绝蒸馏 |
| E5 原型 | E4 加正效用阈值 margin，默认 0.05；**未实现教师可靠性过滤** |
| 效用估计 | 仅重建头参数的内积、余弦、梯度范数；可选 functional_call 虚拟 SGD 一步；不改学生参数、梯度或优化器状态 |
| 评估 | 逐图 PSNR/SSIM/MSE；与同 manifest 的 E0 比较逐图 PSNR 下降比例；学生单独推理耗时；可保存图像 |
| 独立效用审计 | 对额外保留的训练内候选/参考样本记录真实虚拟收益、负收益符号准确率、负样本比例、相关性 |

## 安装和数据

Python ≥3.8，PyTorch ≥2.1；无需 mamba_ssm。服务器使用 **school-gpu2 / CDF_FFT**，已发现解释器 `/home/user/Conda/CDF_FFT/bin/python`（Python 3.8.20）。

```bash
python -m pip install -e '.[test]'
```

每个划分一个 UTF-8 CSV：

```csv
id,input,target
scene001,images/scene001.png,targets/scene001.png
```

图像路径相对 CSV 所在目录；配置内的 CSV 和 checkpoint 路径相对启动工作目录。输入与 GT 必须同尺寸、严格配对，读取为 RGB [0,1]。训练随机裁剪 256×256、成对水平/垂直翻转，小图复制边界补齐；验证/测试保持原尺寸、batch=1。默认数据加载单进程，便于可复现。

请从**原始训练数据池**划出互斥的 `train`、`utility_train`、`utility_eval`、`val`；效用审计另外保留 `audit_candidates` 和 `audit_references`。`test` 是最终测试集。推荐按场景/视频序列分组后划分，以免相邻帧泄漏。所有 E0–E5 使用完全相同的训练划分，即使 E0 不消费效用子集，也不把这部分数据加回 E0 训练。

程序拒绝跨划分重复 ID 或相同解析后文件路径，但**不能识别另存文件、近重复图像或同场景相邻帧**，数据管理者仍须保证语义隔离。必须在配置中登记最终测试 manifest，才能检查其与其他划分的路径/ID 重叠。测试 manifest 不加载进训练 DataLoader，训练只读取其目录元数据用于隔离检查。

没有提供真实数据划分或已训练教师 checkpoint；配置中的路径是需要填写的占位值。配置默认列出全部效用划分以保障公平对照；仅调试 E0–E2 时可将不用的效用路径置为 `null`。

## E0、E1、E2

编辑 `configs/*.yaml` 填入数据路径。E1/E2 需要 `teacher.mode: checkpoint`、`teacher.checkpoint` 和与 checkpoint 匹配的 `teacher.width`。可接受本项目训练保存的 checkpoint（读取 `student` 键），也可直接加载 U-Net state_dict。真实教师必须预训练完成且不使用测试数据；程序只能校验格式，不能判断权重的训练来源。

```bash
python -m luad_uie.scripts.train --config configs/E0.yaml --output runs/E0_seed42
python -m luad_uie.scripts.train --config configs/E1_color.yaml --output runs/E1_color_seed42
python -m luad_uie.scripts.train --config configs/E1_illumination.yaml --output runs/E1_light_seed42
python -m luad_uie.scripts.train --config configs/E1_detail.yaml --output runs/E1_detail_seed42
python -m luad_uie.scripts.train --config configs/E2.yaml --output runs/E2_seed42
```

没有教师时可显式改为 `teacher.mode: oracle` 验证代码链路；此模式直接拿 GT 当教师输出，**不构成合法 KD 实验**，日志与评估会记录该模式。不会自动回退到 oracle 或随机教师。E2 使用平均损失而非三项直接求和，保证总体权重和为 1；不同视图的损失量纲仍不完全一致，需要独立做尺度/λ敏感性分析。

```bash
python -m luad_uie.scripts.evaluate --checkpoint runs/E0_seed42/best.pt --manifest data/test.csv --output runs/E0_test
python -m luad_uie.scripts.evaluate --checkpoint runs/E2_seed42/best.pt --manifest data/test.csv --output runs/E2_test --baseline runs/E0_test/metrics.json --save-images
```

无 GT 推理使用仅有 `id,input` 的 CSV；也可加 `--inference-only` 忽略已有 GT 像素。预测路径只调用学生，不实例化教师或控制器。GT 仅在评估指标计算时使用。图像保存为顺序编号，与 `metrics.json` 的逐图顺序对应。

## 效用估计和动态控制

```bash
python -m luad_uie.scripts.train --config configs/E4.yaml --output runs/E4_seed42
python -m luad_uie.scripts.train --config configs/E5.yaml --output runs/E5_seed42
python -m luad_uie.scripts.analyze_utility --run runs/E4_seed42 --candidates data/audit_candidates.csv --references data/audit_references.csv --output runs/E4_audit
```

每 `utility.interval` 步，从独立训练内候选集取一个样本计算 KD 梯度，从训练内参考集取另一个样本计算恢复梯度。余弦作为有界回归标签（不是直接回归真实收益），内积和范数同时保存。每 `virtual_interval` 步在同一头参数子集上计算真实虚拟 SGD 一步收益；只有与 `interval` 重合的步数执行，建议设为其整数倍。该更新没有恢复梯度、Adam 动量或 weight decay，与实际学生 Adam 联合目标更新不同。

控制器先单独更新，然后学生使用 detach 后的权重更新；前 `warmup_steps` 学生只优化恢复。控制器输入固定 10 维图像统计、学生 pooled 特征和输出统计，均 detach。E4 正余弦预测归一化，E5 将预测减去正 margin 后再截断和归一化；全部无益时权重为零。**E5 只是阈值消融，不是完整的可靠性过滤方法。**

训练期 `utility.jsonl` 的预测记录在当前标签更新之前，但样本会被反复使用，因此这些记录并不等同于独立泛化测试。`analyze_utility` 拒绝候选和参考与训练运行登记的任何划分重叠，专门测量独立样本上的预测能力。仅凭低 MSE 或负样本准确率不能得出成功结论，需要同时查看负样本基率、相关性、真实恢复效果和 E3 对照。

## 输出与替换接口

每次运行独立目录，存在则报错：保存配置、manifest SHA256/解析后划分、教师 SHA256、软件版本、逐步权重和损失、效用诊断、每轮指标/训练时间/GPU 峰值分配显存、`best.pt` 与 `last.pt`。best 仅由验证 PSNR 选出；测试不用于选模型。当前保存优化器状态但**未提供精确断点续训接口**。确定性设置与固定种子已启用；跨 GPU/PyTorch 版本不保证逐位一致。延迟为 batch=1、每图 3 次预热、同步 GPU 后单次计时，仅供初步比较，需要重复测量才能用于论文。

- `Student.forward(x, return_features=False)`；特征字典需含 `pooled`，提供 `feature_dim` 和 `utility_parameters()`（参数名必须匹配 `named_parameters()`）。
- `Teacher.forward(x)` 返回同尺寸 RGB；替换时传入自定义 `nn.Module` 并调整 `build_teacher`，保持冻结约束。
- `KnowledgeView(student, teacher)` 返回 `[B]`；容器 `KnowledgeViews` 返回 `[B,3]`，顺序为 color/illumination/detail。
- `Controller(degradation_state, student_state)` 返回 `[B,3]` 预测；`weights()` 产生非负、和为 1 或全部为 0 的权重。

CLIP 开关明确报未实现；Mamba、多教师、像素路由、二阶元学习、教师可靠性过滤、跨数据集/检测评估均未实现。

## 验证与研究边界

```bash
python -m pytest -q
PYTHONPATH=. python tests/smoke_pipeline.py --output runs/synthetic_smoke --device cuda
```

单测检查形状/梯度、教师冻结、控制器 detach、全零权重与纯恢复梯度一致、虚拟更新与独立副本真实更新一致且不污染状态、划分泄漏、只读验证和基础指标。合成检查覆盖 E0、三种 E1、E2–E5、checkpoint 教师加载、无 GT 推理和独立效用审计。其数据、随机教师和 oracle 分数都不能用于科学论证。

尚待真实数据、多随机种子验证：①不同视图的边际效用是否随退化与阶段稳定变化；②头梯度代理是否与虚拟一步/长期收益相关；③控制器是否优于 E3 统计策略；④是否优于 E0/E2 并降低逐图负迁移；⑤跨域是否可复现。先验证这些条件，再决定是否增加复杂模块。验证执行情况另见 `VALIDATION.md`。

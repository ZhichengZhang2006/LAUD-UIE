# LUAD-UIE：学习效用感知的水下图像增强蒸馏框架（初步设计）

> **状态**：研究构想 / 最小可行框架（MVP），不是已验证的方法或既定创新结论。  
> **用途**：交给 Codex 搭建可扩展的 PyTorch 项目骨架，先验证“不同知识监督的边际学习效用是否随退化状态和学生能力变化”。  
> **暂定名称**：**LUAD-UIE**（Learning-Utility-Aware Distillation for Underwater Image Enhancement），可在实验后修改。

## 1. 核心研究问题

现有动态多教师/退化感知蒸馏已有较多研究，因此**不以“CLIP 识别退化 + 加权多个教师”作为创新点**。

本构想研究：对于颜色偏移、非均匀光照、散射、细节缺失等耦合退化，某类教师知识即使“看起来相关”，也不一定能改善当前学生。能否**基于实际学习效用**预测并选择有效蒸馏监督，减少无效学习和负迁移？

区分三个量：
- **退化需求（Need）**：当前图像需要恢复什么？
- **知识可靠性（Reliability）**：当前教师/知识视图提供的信息是否可信？
- **学习效用（Utility）**：该监督能否真正改善当前学生的恢复目标？

最关键的待检验假设：**高退化需求不等于高蒸馏效用**。

## 2. 初版框架（优先实现）

```text
                        Input underwater image I
                                 |
                  +--------------+---------------+
                  |                              |
          Degradation Encoder               Student UIE Sθ
        (low-level stats first;                |       |
          CLIP optional)                 features Fs  output Is
                  |                              |
                  +----------+-------------------+
                             |
                Utility Predictor / Controller Pφ
                    |         |          |
                  w_color   w_light    w_detail
                    |         |          |
          Color View   Illumination View   Detail View
                    \         |          /
                  Frozen restoration teacher T
                             |
                     Weighted KD losses
                             |
                 L = L_rec + λ · L_KD
```

**MVP 采用“单教师 + 多知识视图”**，减少训练成本并排除教师模型数量这一混杂因素：

1. **Student**：轻量 U-Net（模块接口通用，后续可替换 MambaCue 或其他骨干）。输出增强 RGB 图像。
2. **Frozen Teacher**：一个已训练的 UIE 恢复模型；完全冻结，仅在训练期间提供参考输出/特征。若无可用 checkpoint，先允许以 `teacher_output = target` 的 *oracle 调试模式* 验证软件管线，但**不得将该模式结果当作合法的知识蒸馏实验**。
3. **Knowledge Views**：从教师输出和学生输出建立三种可比较的蒸馏损失：
   - `color`：Lab 色度（a/b）或颜色分布一致性；
   - `illumination`：亮度与低频照明一致性；
   - `detail`：梯度/拉普拉斯/多尺度高频结构一致性。
4. **Degradation Encoder**：第一版使用可微或固定的颜色、亮度、对比度、梯度统计，加小型 CNN/MLP 生成退化表示；**CLIP 作为独立可选分支/对照实验**，不作为 MVP 的强制依赖。
5. **Student-State Encoder**：从学生中间特征的 pooled 表示及输出统计抽取状态；在控制器输入中对学生特征 `detach`，避免因控制器优化无意改变学生主干。
6. **Utility Controller**：输入退化状态、学生状态，输出各视图的效用预测及蒸馏权重。支持 `no_distillation`（全部预测为无益时，仅优化恢复损失）。

**推理阶段只保留 Student**；退化分析器、教师与控制器均可为训练期模块，不增加部署时推理开销。

## 3. 核心训练机制

基础恢复损失可先采用 Charbonnier + SSIM：

\[
L_{rec}=L_{char}(I_s,I_{gt})+\alpha L_{ssim}(I_s,I_{gt}).
\]

每个知识视图对应蒸馏损失 \(L_k\)，动态蒸馏为：

\[
L_{KD}=\sum_{k\in\{c,l,d\}}w_k L_k,\qquad
L=L_{rec}+\lambda L_{KD}.
\]

### 学习效用标签（研究核心）

在**训练集独立划分的效用监督子集/小批次**上，周期性估计某个蒸馏视图对当前学生的一步收益：

\[
U_k=L_{rec}(\theta;B_{eval})-
L_{rec}(\theta-\eta\nabla_\theta L_k(\theta;B_{train});B_{eval}).
\]

正效用表示该候选更新短期有益；负效用表示可能发生负迁移。这个量是**局部代理指标**，不能等同于最终泛化收益。

**MVP 优先实现廉价近似**：

\[
\tilde U_k=\langle \nabla_{\theta_s}L_{rec},\nabla_{\theta_s}L_k\rangle,
\]

其中 \(\theta_s\) 仅选学生最后一层/重建头的参数，减少额外反传成本。先将余弦相似度与梯度范数分别记录，验证它们与真实一步收益的相关性；如不相关，再按固定间隔计算真实虚拟一步更新（使用 `functional_call` 或参数拷贝，不污染真实训练状态）。

使用这些**训练阶段产生的效用标签**监督控制器 \(P_\phi\) 预测 \(\hat U_k\)。先做回归/排序预测，再将正效用转化为稀疏非负权重；允许全零权重，避免 softmax 强制分配教师。控制器与学生的更新需分离，首版可交替训练，并在权重计算路径适当 `detach` 防止目标泄漏或投机优化。

> **关键限制**：最终测试集 GT 严禁用于效用估计、教师选择或权重学习；如使用 GT 进行虚拟一步验证，仅限训练集内部划分。

## 4. 最小实验矩阵

固定**同一学生架构、训练数据、轮数、数据增强和评估代码**，依次对照：

| ID | 训练策略 | 目的 |
|---|---|---|
| E0 | `L_rec`，无蒸馏 | 学生基线 |
| E1 | 单一知识视图蒸馏（各跑一次） | 检验各视图是否有效 |
| E2 | 三视图等权蒸馏 | 检验互补性 |
| E3 | 根据退化统计/CLIP 输出权重 | 原始构想的基线 |
| E4 | 效用预测驱动权重 | 核心研究假设 |
| E5 | E4 + 负效用拒绝/可靠性过滤 | 检验负迁移抑制 |

至少记录：PSNR、SSIM、MSE、各视图权重/效用、负效用预测准确率或排序相关性、**相对 E0 逐图性能下降的比例**、训练开销、学生推理延迟。后续补充跨数据集泛化与目标检测效果。

**必须单独检验两个前置条件**：
1. 不同知识视图在不同退化子集和不同训练阶段的实际效用，是否真的存在可重复差异？
2. 控制器能否在独立样本上预测这种差异，并优于仅根据颜色/亮度/梯度统计的廉价策略？

若两项不能成立，应停止增加复杂模块。

## 5. 请 Codex 先完成的工程范围

仅实现**可训练、可评估、可替换的原型框架**，不要为了论文包装而增加复杂模块。

建议目录：

```text
luad_uie/
├── configs/                 # E0–E5 与数据、教师、学生配置
├── datasets/                # paired RGB / GT；train/val/test 划分
├── models/
│   ├── student.py           # 轻量学生 + features 接口
│   ├── teacher.py           # 冻结教师 checkpoint 适配层
│   ├── knowledge_views.py   # color / illumination / detail KD
│   ├── degradation.py       # 统计特征；可选 CLIP
│   └── utility_controller.py
├── losses/                  # reconstruction、distillation
├── trainers/                # 基线训练、效用估计、交替优化
├── scripts/                 # train / evaluate / analyze_utility
├── tests/                   # shape、梯度、冻结教师、无泄漏单测
└── README.md
```

**实现要求**：
- PyTorch，配置可复现；固定随机种子，自动保存配置、日志与权重。
- 定义统一接口：`Student.forward(x, return_features=False)`、`Teacher.forward(x)`、`KnowledgeView(student, teacher)`、`Controller(degradation_state, student_state)`。
- 先跑通 E0/E1/E2，再实现效用估计和 E4/E5；CLIP 留可选开关。
- 单卡可运行；默认 256×256 RGB；不要将预训练教师或额外数据来源写死。
- 为 `no_distillation`、教师参数不更新、验证集不参与训练、控制器权重范围、测试集无 GT 依赖编写单元测试。
- README 明确区分**已实现组件、占位组件、尚未验证的科学假设**。

## 6. 暂不实现的内容

- 三个大型独立教师、像素级教师路由、在线教师调用、Agent 工作流；
- 推理阶段运行 CLIP/教师；
- 完整梯度冲突投影算法或二阶元学习；
- 在结果证明必要性之前加入 Mamba、扩散模型及额外先验分支。

## 7. 创新性边界与成功标准

目前不能声称动态教师选择、CLIP 感知退化、可靠性加权或学生状态蒸馏本身是新的。应与 **CPLDP、DIE2UIE、DTKD-Net、MTKD-RL、DCKD、LIFT and PLACE** 以及水下频域可靠蒸馏研究进行比较。

**初步成功标准**：
1. 各知识视图的边际效用存在清晰、可复现的样本/退化依赖性；
2. 效用预测优于退化程度直接加权，并能识别负效用监督；
3. 在相同学生推理开销下，优于 E0/E2/E3，且降低负迁移样本比例；
4. 至少在一个独立水下测试集上具有稳定收益。

**优先级**：先验证科学假设，再扩展异构教师、区域级蒸馏和跨域评测。不要在未获得结果前宣称 SOTA 或高创新性。

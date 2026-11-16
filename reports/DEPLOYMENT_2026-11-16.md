# 渐进式蒸馏模块部署记录

- 方案提交：bf29199；实现提交：087e0f8；已推送GitHub main。
- 服务器school-gpu，环境CDF_FFT，阶段目录 `/home/user/luad_uie_prototype_20261031/stages/progressive_087e0f8`。
- 启动前GPU0空闲19367MiB，利用率100%，无本项目其他训练进程；保留其他用户任务，串行运行本项目队列。GPU并非空闲卡，因此不能用此时耗时声称独占GPU速度。
- CPU单元测试10项通过，覆盖逐图解析头梯度与autograd一致性、冲突关闭、预测器梯度隔离和checkpoint状态。
- 完整合成流程通过：E0—E5及RG-full/scale/learned训练与评估，learned续训、无GT推理、教师checkpoint加载。初次`-m tests.smoke_pipeline`因tests包解析失败未运行；改用`PYTHONPATH=. python tests/smoke_pipeline.py`后通过，源实现无须修改。
- 实际队列启动：RG-full100 → RG-scale100 → RG-learned100 → 教师从100续训至累计250。每个任务完成评估后才进入下一项。日志与状态文件在阶段目录runs；自动失败停止，不自动重试。
- 已实现：逐图头梯度路由、EMA尺度校准、稀疏梯度监督的在线MLP路由、训练评估和状态保存、任务队列。
- 待实测：模块超越E1-detail/E2、逐图下降比例降低、稀疏预测器保持效果并降低成本、教师250改善验证性能。尚不支持泛化效用/SOTA结论。

源数据、checkpoint、全部训练日志仅保存在服务器/home/user；仓库保留方案、代码和小型真实结果摘要。实验尚在运行，后续报告必须以实际完成记录为准。

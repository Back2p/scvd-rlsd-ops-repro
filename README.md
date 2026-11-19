# SCVD / RLSD 四数据集训练与评测代码包 v2.1（复查交付版）

以 Qwen3.5-9B 为基模，在 CloudOps、ITBench-AA、OpenRCA Bank 和 CUJBench 上运行 SCVD、RLSD 和同协议基线。评测支持 Qwen3.5-27B 对照。当前仓库采用 v2.1 的训练和评测实现，已通过离线 CPU 与命令入口检查。真实权重、CUDA、多卡训练、采样重载和断点恢复待目标平台验收。

项目方 fork 后，克隆 fork 的 `main` 分支，在仓库根目录执行：

```bash
python verify_package.py
python -m pip install -r requirements-runtime.txt
mkdir -p /platform/config
cp configs/runtime.example.json /platform/config/runtime.json
export CONFIG=/platform/config/runtime.json
# 先按平台先读.md 修改路径、采样服务和教师配置
python -m ops_repro.doctor --config "$CONFIG" --runtime
python -m ops_repro.prepare --config "$CONFIG" --download model cloudops itbench_aa openrca cujbench --plan
```

`/platform` 是示例挂载点，请替换为平台实际路径。上述命令用于配置检查和打印下载计划。后续依次下载或挂载模型与数据、冻结数据、运行 CPU 检查和平台 GPU 验收，再运行基线、SCVD/RLSD、验证集选点和测试集评测。详细配置见 [平台先读](平台先读.md)，完整命令见 [RUNBOOK](RUNBOOK.md)。仓库带案例清单和运行所需的轻量元数据，平台另行准备模型权重与完整观测数据。

本仓库使用 `snapshot-reference-v2` 工具及评分协议。划分与参考 HTML 披露的数量和分布匹配，原始逐题 ID 尚待项目方确认。方法增益应依据同协议重跑的 9B/27B 基线。严格对齐原实验需要原 split ID、工具接口及评分器。参见 [数据划分说明](数据划分说明.md) 和 [实现范围](IMPLEMENTATION.md)。

| 内容 | 入口 |
|---|---|
| 数据下载、内容/标签冻结 | `python -m ops_repro.prepare` |
| 只读 CPU 配置检查 | `python -m ops_repro.doctor` |
| 可选 GPU 零更新验收 | `torchrun ... -m ops_repro.qualify` |
| 原生 HF token 采样服务 | `python -m ops_repro.worker` / `scripts/start_workers.sh` |
| SCVD 候选→教师接续→独立评分 | `python -m ops_repro.collect` |
| 原生模板、教师续写标签 | `python -m ops_repro.encode` |
| SCVD 全文本参数训练 | `torchrun ... -m ops_repro.train --method scvd` |
| RLSD 在线多轮采样与更新 | `torchrun ... -m ops_repro.train --method rlsd` |
| 同实现 GRPO 对照（可选） | `torchrun ... -m ops_repro.train --method grpo` |
| 发布权重/查询 sampler 状态 | `python -m ops_repro.control` |
| 冻结模型、三次重复评测 | `python -m ops_repro.evaluate` |
| 逐数据集 CSV/报告/badcase 初筛 | `python -m ops_repro.report` |

真实清单：CloudOps 604/75/75、OpenRCA Bank 80/13/43、CUJBench 57/14/14、ITBench-AA 19/7/5（train/val/test），共 760 训练、109 验证、137 测试。重建匹配 HTML 数量及已披露边际分布，原始逐 ID 成员尚未确认。ITBench 保留完整 40 题目录与 31 题兼容子集；CUJ 85 题故障子集的两条健康基线仍在原件中。

推荐两组独立 8 卡：一组 FSDP 训练并放置冻结教师，另一组提供 8 个学生采样副本；SCVD 强教师通过显式配置服务调用。两角色共享权重发布目录。参考实现每次 RLSD 更新后发布完整 BF16 权重、等所有 sampler 校验并重载，再开始新采样；磁盘/网络开销显著，没有吞吐优化或完成时长保证。

训练范围是全部 Qwen3.5 **文本骨干与 LM head 参数**，不是 LoRA；视觉塔不载入。SCVD 沿用 3 epoch、LR 1e-5、global 64 等主要参数；采用 FSDP，区别于论文 ZeRO-3 Offload。RLSD 保留环境优势方向、detached token 权重、8 条组内采样、非对称 PPO clip、冻结/周期同步教师等机制。差异必须与结果一起报告。

gold、评分、训练 hint 均在协调端。模型工具仅接受证据 ID；ITBench 采用观测白名单，CUJ 不给截图。SCVD 强教师不读取 gold；RLSD hint 只出现在 train 冻结教师前向中。val/test 不进入采集。评测缺失/基础设施错误不删分母，正式 accuracy 留空并单列覆盖率。

活动入口位于 `ops_repro/`；`src/` 只保留实际使用的数据、编码与审计工具。旧 Protocol 空接口、旧 readiness 草稿和未参与运行的上游框架已从本交付版移除。`reference/` 保留论文配置/来源锁，`tests/fixtures/` 仅保留官方公式回归片段；本包不声称原论文逐数值复现。复查结果见 [reports/代码复查报告.md](reports/代码复查报告.md)。

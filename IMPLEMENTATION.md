# 实现范围与复现差异

本包实现“原始快照→固定清单→多轮工具→SCVD/RLSD 训练→同协议评测”。目标平台 GPU 集成未执行；单元测试、合成结构、零更新脚本和公开 checkpoint 迁移分数均不作训练效果。

| 项目 | 论文/项目参考 | 本包 |
|---|---|---|
| 学生 | 尽量 Qwen3.5-9B | 固定 9B；文本骨干+LM head 全参数；不载入视觉塔 |
| SCVD 候选 | 首个完整候选，同状态教师接续 | propose_diagnosis 边界，深复制会话/历史，同一只读证据版本 |
| SCVD 标签 | 去采集提示，仅教师 continuation | 原生模板；学生/工具 mask，教师工具动作/结束 token 监督；超长隔离 |
| SCVD 配置 | 3 epoch、AdamW 1e-5、global64、cosine/.03 warmup、32K、ZeRO-3 | 同主要数值；FSDP full-shard、FP32 master、BF16；global teacher-token mean；不声称作者 packing/reduction 一致 |
| 采样配置 | SCVD 论文 temp1/top-p.95/top-k20 | 默认统一为 temp1/top-p1/top-k0，满足本包 RLSD 未扭曲策略概率约定；不是 SCVD 原采样数值复现 |
| RLSD 原数据 | MMFineReason、多模态 Qwen3-VL-8B | 四领域快照+Qwen3.5-9B，属于迁移 |
| RLSD rollout | 当前学生 on-policy | HF worker 实际 token/旧 logprob，权重/模板校验；每更新发布/重载 |
| RLSD 教师 | privileged final answer，冻结/同步 | 独立 frozen FSDP；仅 train hint；每 10 优化更新同步；no_grad/detach |
| RLSD 优势 | 环境优势×detached evidence | torch.std correction=1；log-space clip；与固定官方公式数值对照 |
| PPO | .2/.28 clip；repo dual clip=3 | 同非对称/dual clip；无额外 KL/entropy；global generated-token mean |
| 硬件 | SCVD 8 H200；RLSD 32 H200 | 默认 8 训练+8 采样；资源/吞吐未测，不是原硬件数值复现 |
| 数据划分 | HTML 数量/分布，无 ID | 真实 ID 重建；边际相同，成员未经项目方确认 |
| 工具/评分 | HTML 未附实现 | snapshot-reference-v2；需重跑同接口 Base/27B，不能直接比较 HTML 旧分数 |

RLSD：delta=stopgrad(logp_teacher-logp_student)；w=clip(exp(sign(A)*delta),1-eps,1+eps)；A_token=A*((1-lambda)+lambda*w)。A_token 保持环境优势方向。one-based step 下 target lambda=.5、50 步衰减，首更新 .49。工具不在 action 数组；hint 仅改变教师前缀，两侧学生 action token 尾部逐一相同。

旧 logprob 来自采样请求内同一冻结权重的完整词表概率。RLSD 仅允许 temperature=1/top_p=1/top_k=0，避免温度/截断分布与原策略混用；每更新前权重固定，microbatch 不看到半更新权重。重算超过 drift 阈值停止。基础设施失败不伪装负奖励；优势全零不执行纯 weight decay 更新。

CloudOps 暴露 raw_data；CUJ 暴露 tool_cache、不含截图；OpenRCA 暴露 query 日期的 telemetry；ITBench 为 alerts/metrics/四种 raw TSV 白名单。读取分页、搜索为有界字面匹配，不执行生成代码。与原 OpenRCA Python 分析工具相比，这改变能力边界，默认实现不能冒充原 RCA-agent。

评分：CloudOps fault_object+root_cause 去空格/大小写精确匹配；OpenRCA 真实 scoring_points 全对、≤60秒、多根因集合；CUJ 发布 taxonomy 的 component AND type canonical match，layer/evidence 非主指标；ITBench 规范化 legacy/spec，按 root groups/aliases/filters 匹配并拒绝额外实体。仍可能与 HTML 定制口径不同，尤其其 ITBench 明示宽松匹配。

内容/label 指纹留协调端；actor input、权重、版本、协议写入日志，模型工具不能读取 gold/评分器/物理路径。边界检查不证明所有语义侧信道都已消除。每更新全量共享盘发布是 reference 路线，没有吞吐或完成时间测量；平台若改接高吞吐引擎，必须保留实际 action、旧 logprob、revision、教师对齐和更新屏障。

原论文参考：`reference/` 保留配置、固定源码来源锁和上游署名，官方公式片段用于 CPU 回归。本交付版不附未使用的上游训练框架，也不把原论文旧环境混入当前领域实现。SCVD 精确 11,806 任务池、7,419 配对轨迹和作者实现未恢复，因此不宣称四领域迁移等同原论文完整数值复现。

一手来源：[SCVD](https://arxiv.org/html/2609.38812v1)、[RLSD](https://arxiv.org/html/2604.03128v1)、[官方固定源码](https://github.com/iie-ycx/RLSD/tree/aca313b978de4fb3a509ddf1831bf61fe597a2eb)、[Qwen3.5 文本接口](https://huggingface.co/docs/transformers/model_doc/qwen3_5)、[固定 PyTorch 镜像](https://hub.docker.com/layers/pytorch/pytorch/2.8.0-cuda12.8-cudnn9-devel/images/sha256-a7103283ea7113e10ae5d014bd2342acebda0bc53164b2f7b1dd6eb7a766bdb6?tab=layers)。

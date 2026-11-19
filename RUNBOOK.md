# 平台运行说明

以下为平台执行命令。本地准备阶段没有执行训练、真实权重加载、worker 启动、教师调用或 GPU 验收。

## 1. 环境、配置和资源

从包根目录运行。Python 3.11/3.12、Linux、CUDA/NCCL、PyTorch 2.8.0、Transformers 5.2.0，安装 requirements-runtime.txt。Dockerfile 固定官方 PyTorch CUDA 镜像 digest；**本地未拉取/构建这个 8.75 GB 镜像**，驱动及混合注意力内核需平台实测。

首次解压、编辑配置前，可用标准库运行 `python verify_package.py` 核对交付文件 SHA256。CPU 检查脚本为 `bash scripts/check_cpu.sh`；原生模板的 7 项检查需要设置 QWEN_TOKENIZER_DIR，未提供时会明确 skip，而不会伪报通过。

```sh
python -m pip install -r requirements-runtime.txt
mkdir -p /platform/config /platform/output /platform/data /platform/models
cp configs/runtime.example.json /platform/config/runtime.json
export CONFIG=/platform/config/runtime.json
python -m ops_repro.doctor --config "$CONFIG" --runtime
python -m ops_repro.train --config "$CONFIG" --method scvd --output /platform/output/PLAN --plan
python -m ops_repro.train --config "$CONFIG" --method rlsd --output /platform/output/PLAN --plan
```

修改 JSON：四个数据根、模型路径、frozen_data_manifest、worker endpoints、teacher URL/model/revision 与 key 环境变量名。9B 已锁定 c202236235762e1c871ad0ccb60c8ee5ba337b9a；27B 配置锁定 fc05daec18b0a78c049392ed2e771dde82bdf654。更换版本需重新登记配置和基线。teacher 的三个 null 为平台必填项，未配置会阻止实际采集；API key 只通过环境变量/平台 secret 提供。

默认两组 8 卡分工训练/采样，无需两组共建 16-rank FSDP。两节点需共享模型发布目录。全参数 FP32 master/Adam/冻结教师恢复状态、保留 checkpoint、发布副本和数据可能占数百 GB，建议预留约 1 TB 专用共享盘，实际容量以平台测量为准。不能从 CPU 检查承诺 24 GB/80 GB 卡或 32K 一定可跑。

新运行目录内部默认保留最近两份恢复状态和一份 sampler 发布。保留策略仅移除该 run 创建且 owner marker 匹配的旧产物，不清理数据、外部目录或历史实验。输出必须用新目录，恢复也写新目录；需要更多 val 选择点时，正式运行前调大保留数。

## 2. 下载并冻结数据

```sh
python -m ops_repro.prepare --config "$CONFIG" --download model cloudops cujbench openrca itbench_aa --plan
python -m ops_repro.prepare --config "$CONFIG" --download model cloudops cujbench openrca itbench_aa
python -m ops_repro.prepare --config "$CONFIG" --freeze-data
```

目录：CloudOps `<root>/benchmark/<system>/<category>/<id>/raw_data/`；CUJ `<root>/scenarios/<SCEN-ID>/tool_cache/`；OpenRCA `<root>/Bank/query.csv`、`<root>/Bank/telemetry/YYYY_MM_DD/`；ITBench `<root>/sre/Scenario-N/`。OpenRCA 下载官方公开 Drive Bank.zip（约 2.83 GB），解压前要求至少 60 GiB 空间，不使用账号 cookies。Drive 受限时可按数据说明中的官方链接手动下载/挂载。

CloudOps 观测逐文件核对 Git blob，答案 metadata 单独留在协调端。CUJ 核对发布 SHA256。OpenRCA 必须有真实 query.csv/scoring_points，并逐 row_id 核对 instruction，**归档 prediction 不作为标签**。ITBench 使用固定 HF revision。

freeze 对允许观测登记内容 SHA256，对协调端标签单独登记指纹；缺失、查询错位、schema 错误均输出错误清单。complete=false 不能开训。修复后选择新冻结文件路径，不覆盖旧回执。模型工具不能浏览包根、dataset_sources 或 gold 目录。

## 3. 可选平台验收：零更新

```sh
torchrun --standalone --nproc_per_node=8 -m ops_repro.qualify \
  --config "$CONFIG" --output /platform/output/qualify-128 --sequence-length 128
```

只执行合成输入的前向、反向、有限梯度和导出；**无优化器、无 optimizer.step**。可用新目录、sequence-length 32768 检查长序列；仍不含 Adam 状态、真实 rollout 和吞吐，不能当完整训练容量验证。首个真实更新后还需检查优化器状态、保存/重载与实际资源。

另提供 configs/gpu-smoke.example.json：RLSD 只抽 1 个 prompt、2 条 rollout、最多 1 次实际优化更新；SCVD 为 1 epoch/global8。它只供平台以后主动运行最小链路验收，**准备方未执行**，不能当效果实验。按同样路径/服务配置复制修改后运行；冒烟配置与完整配置的 checkpoint 不互相冒充或直接续训，完整实验仍从规定初始模型开始。

## 4. 启动采样服务

在 8 卡采样节点执行，模型与 output 路径要和训练节点一致：

```sh
export MODEL_PATH=/platform/models/Qwen3.5-9B
export MODEL_REVISION=c202236235762e1c871ad0ccb60c8ee5ba337b9a
export RUN_ROOT=/platform/output
bash scripts/start_workers.sh
```

默认仅绑定 127.0.0.1。跨节点设 WORKER_HOST=0.0.0.0，通过 secret 注入 RUNNER_API_KEY，JSON endpoints 改为采样节点地址，并设置 rollout.api_key_env 为 RUNNER_API_KEY。单个 16 卡主机可给 worker 设置 WORKER_DEVICE_OFFSET=8，训练使用前 8 卡。脚本只停止自己创建的 PID，不使用全局 pkill。

```sh
python -m ops_repro.control --config "$CONFIG"
```

每个 worker 显示当前 revision，采样保存真实 prompt/generated IDs、旧 logprob、模板与权重指纹。reload 校验完整文件后再加载模型，短时需要容纳新旧两份 BF16 模型。参考实现为全量共享盘发布，不是已优化的 vLLM 同步系统。

## 5. 同协议基线

```sh
python -m ops_repro.evaluate --config "$CONFIG" --output /platform/output/base9b-val \
  --split val --policy-revision "$MODEL_REVISION" --arm base_9b
```

训练期间只用 val 选择；模型和协议冻结后，按预定方案评 test。若需要 HTML 的 val+test 视图，可最终一次使用 --split eval，246 case×2 模式×3 次=1,476 attempts。不要重跑并挑最好结果，也不要使用原 HTML 基线代替本协议基线。

## 6. SCVD 采集、编码、训练

确保 worker 仍为初始 9B。配置强教师服务、固定版本和 secret 后：

```sh
python -m ops_repro.collect --config "$CONFIG" --output /platform/output/scvd-collection --plan
python -m ops_repro.collect --config "$CONFIG" --output /platform/output/scvd-collection
python -m ops_repro.encode --config "$CONFIG" \
  --records /platform/output/scvd-collection/accepted.jsonl --output /platform/output/scvd-encoded.jsonl
export SCVD_RECORDS=/platform/output/scvd-encoded.jsonl
export OUTPUT=/platform/output/scvd-training
bash scripts/train_scvd.sh
```

默认 760 train case、两模式各一次，预定 1,520 次采集。学生必须先 propose_diagnosis；直接结束、无候选、教师失败或最终评分未通过均保留失败。只有独立评分通过的教师续写进入 accepted；超长/模式不一致进入 held，训练规模以 admitted 和监督 token 为准。

非思考模式默认不监督教师私有 reasoning，仅保留可见动作与回答；思考模式保留提供的 reasoning_content。提供方模式开关可填 teacher.mode_extra_body；API 不支持 seed 时不能声称逐字节重放，冻结 accepted/encoded SHA256 后再训练。

默认把历史 reasoning_content 内联为 assistant 文本发给通用兼容 API；若提供方要求独立字段，设置 teacher.message_reasoning_format="separate_field"。原始学生历史仍按本地原生模板保存；教师响应 ID、响应 model 和请求版本一并登记，提供方若只支持可变别名，应明确报告版本不确定性。

训练为全文本参数 FSDP、FP32 master/Adam、BF16 计算、激活 checkpoint；LM head 只分块投影教师监督位置。global batch 64 在 8-rank 下累积 8 个 microbatch，16-rank 是 4；不足批量用无监督 dummy 补 collective，不重复真实监督样本。

```sh
python -m ops_repro.control --config "$CONFIG" --publication /platform/output/scvd-training/result.json
export SCVD_REVISION="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["final_publication"]["revision"])' /platform/output/scvd-training/result.json)"
python -m ops_repro.evaluate --config "$CONFIG" --output /platform/output/scvd-val \
  --split val --policy-revision "$SCVD_REVISION" --arm scvd
```

## 7. RLSD 在线训练

```sh
export OUTPUT=/platform/output/rlsd-training
bash scripts/train_rlsd.sh
```

从初始 9B 独立开始，先发布 step-0、等待全部 worker 重载。每批抽 256 个 train prompt draw，每个 draw 在一个明确模式下采 8 条轨迹，模式不混入同一 GRPO 组。教师用 train-only hint 在同一学生 action token 上算 logprob；prompt/工具观测不参与策略 loss。

每次优化更新后发布新权重、等所有 worker 确认再采下一批。权重/模板/logprob 不一致则停止。基础设施失败保留回执并中止该批，不作 reward=0；上下文/轮次预算或格式问题属于模型失败。全对/全错使整批优势为 0 时跳过 optimizer.step，避免纯 weight decay 的伪更新。rollout batch 与实际 update 分别计数，最多收集 1,000 批。

目标为 200 实际更新，LR 1e-6、8 条组内采样、PPO clip .2/.28、dual clip 3、evidence clip .2、教师每 10 更新同步。lambda 使用官方 one-based 约定，首更新 .49、step 50 为 0。可选 METHOD=grpo 使用同代码预算但不启用教师权重。

result.json 指向最新权重。val 选择旧 checkpoint 时需要保留相应状态，另开新目录运行相同恢复命令并加 `--export-only`；SCVD 还需同一 `--records`。这会加载/导出模型但不采样、不执行优化更新；随后 control --publication 指向导出的 result.json，再运行 val。不要把不同 revision 混成同一次评测。默认保留数只覆盖最近状态，要比较多个历史点需在正式运行前增加保留/归档。

## 8. 恢复与 27B

恢复只接受该包自行创建的受信任 states/step-N-batch-M/complete.json 所在目录，不从下载数据读取 torch pickle。要求配置、代码、训练输入、数据冻结清单和 DP world size 一致，输出为新目录：

```sh
torchrun --standalone --nproc_per_node=8 -m ops_repro.train --config "$CONFIG" \
  --method rlsd --resume /platform/output/rlsd-training/states/step-000010-batch-000012 \
  --output /platform/output/rlsd-resumed
```

27B 使用独立配置和端口；保持数据、freeze、protocol、模式及预算与 9B 一致。27B 不进入 9B 训练。先复制并修改配置，再执行下载：

```sh
cp configs/eval27b.example.json /platform/config/eval27b.json
export CONFIG27=/platform/config/eval27b.json
python -m ops_repro.prepare --config "$CONFIG27" --download model
```

若复用同一批 GPU，先在 9B worker 启动终端 Ctrl-C，只停止该脚本创建的进程。采样节点另开终端执行以下命令（实际地址/secret 与 CONFIG27 对应）：

```sh
export MODEL_PATH=/platform/models/Qwen3.5-27B
export MODEL_REVISION=fc05daec18b0a78c049392ed2e771dde82bdf654
export RUN_ROOT=/platform/output/27b-services
export BASE_PORT=8200
bash scripts/start_workers.sh
```

然后在评测终端：

```sh
python -m ops_repro.control --config "$CONFIG27"
python -m ops_repro.evaluate --config "$CONFIG27" --output /platform/output/base27b-val \
  --split val --policy-revision fc05daec18b0a78c049392ed2e771dde82bdf654 --arm base_27b
```

`RUN_ROOT` 用单独目录保留 27B 服务日志；27B 只评原模型，无需重载 9B 发布。相同 GPU 之后再评 9B 时，恢复 9B 模型路径/版本、端口 8100 和原共享 RUN_ROOT，重启 9B 服务，按目标检查点 control 重载。

正式 test 在选点方案冻结后逐臂执行。Base9B 应从初始 9B 服务评；SCVD/RLSD 必须先 control 重载各自选定 publication，传该 publication 的 revision；Base27B 使用 27B 服务。每臂新 output，示例：

```sh
python -m ops_repro.evaluate --config "$CONFIG" --output /platform/output/base9b-test \
  --split test --policy-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --arm base_9b
# 仅在已用 control 重载选定 SCVD 权重、并设好对应 SCVD_REVISION 后：
python -m ops_repro.evaluate --config "$CONFIG" --output /platform/output/scvd-test \
  --split test --policy-revision "$SCVD_REVISION" --arm scvd
# 仅在已重载选定 RLSD 权重、并设好对应 RLSD_REVISION 后：
python -m ops_repro.evaluate --config "$CONFIG" --output /platform/output/rlsd-test \
  --split test --policy-revision "$RLSD_REVISION" --arm rlsd
python -m ops_repro.evaluate --config "$CONFIG27" --output /platform/output/base27b-test \
  --split test --policy-revision fc05daec18b0a78c049392ed2e771dde82bdf654 --arm base_27b
```

以上命令需按对应服务状态分开运行，不能在服务仍加载另一臂时直接执行整段。revision 不符会报错，不会自动替换成当前 worker 模型。

## 9. 结果报告

```sh
python -m ops_repro.report --runs /platform/output/base9b-val /platform/output/scvd-val \
  /platform/output/rlsd-val /platform/output/base27b-val --output /platform/output/comparison-val
```

输出 summary.json、summary.csv、报告.md、badcases.jsonl。缺失/基础设施错误仍保留预定分母，accuracy 留空；未提交、超预算、格式错误记录为模型失败。指标为平均单次 strict accuracy，不是 Pass@3。badcase 自动标签仅初筛，根因需回看 traces 的证据、工具返回、预算、提交和 scorer。

回交 run.json、数据冻结与 SHA、采集/更新/评测完整尝试记录、accepted/held 数量、配置/镜像 digest/环境锁、权重与恢复状态、逐数据集/模式/集合统计及典型案例。代码与 CPU 检查不能代替真实运行证据。

## 四数据集覆盖与训练诊断

正式运行使用 runtime 配置中的四个 datasets，保持两种模式和每题三次。给每条正式 `evaluate` 命令加 `--full-suite`。以下是零模型调用的覆盖检查。

```bash
python -m ops_repro.evaluate --config "$CONFIG" --arm base_9b --split eval \
  --policy-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a --output /platform/output/plan-only --full-suite --plan
```

| 数据集 | Train | Val | Test | Val+Test | Val+Test 尝试数 |
|---|---:|---:|---:|---:|---:|
| CloudOps | 604 | 75 | 75 | 150 | 900 |
| OpenRCA Bank | 80 | 13 | 43 | 56 | 336 |
| CUJBench | 57 | 14 | 14 | 28 | 168 |
| ITBench-AA | 19 | 7 | 5 | 12 | 72 |

全量总计 760/109/137，选点后的 Val+Test 每个方法或基线共 1,476 次。最终汇报可用选中的模型各运行一次 `--split eval --full-suite`，出三种口径。也可用同一选定模型的独立 val/test 结果交付。选点过程中评测的其他 checkpoint 不放进正式汇总。对四个正式臂的结果使用 `ops_repro.report --full-suite`，会检查缺失、失败或重复的单元。

训练目录的 `training_coverage.json` 记录四个数据集的训练 case 数和 SCVD 编码记录数。SCVD 每步记录 teacher-token NLL、学习率、梯度范数和每个有监督样本的数据集 NLL。RLSD 每批记录训练奖励、全对/全错组、零优势组、生成和参与更新的 token 数。每步记录 policy loss、梯度范数、采样概率漂移、教师 λ，以及分数据集的实际调权比例和平均权重偏移。

```bash
python -m ops_repro.diagnostics --run /platform/output/scvd-training --output /platform/output/scvd-curves.html
python -m ops_repro.diagnostics --run /platform/output/rlsd-training --output /platform/output/rlsd-curves.html
```

判断收敛时同时看 loss、梯度和同一固定验证集的多检查点曲线。平台可在训练前提高恢复检查点保留数，按预先约定的步数导出并评 val，例如初始、中期、末期。RLSD 各步使用不同在线样本，policy loss 的下降或接近零不足以说明收敛。奖励组内有区分度、并且有参与更新的 token，才说明环境奖励提供了策略梯度信号。调权比例和偏移用于核对教师修正实际进入了多少 token。判断收益需要比较同协议 held-out 结果，RLSD 与同预算 GRPO 的对照用于区分教师调权贡献与环境奖励本身的贡献。

新版实现指纹随代码更新。若此前已生成冻结回执或 SCVD 编码，请用新版新建冻结路径并重新采集、编码。旧版运行保留原版本分析。

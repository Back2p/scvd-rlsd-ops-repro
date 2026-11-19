# 来源与署名

本包 v2.1 的领域适配、采集协调、训练/worker/评测入口为研究用途的参考实现。SCVD、RLSD 的方法思想与论文结果归原作者，本包不声称作者训练实现或已复现论文数字。

RLSD 官方来源为 [iie-ycx/RLSD](https://github.com/iie-ycx/RLSD/tree/aca313b978de4fb3a509ddf1831bf61fe597a2eb)，固定版本 aca313b978de4fb3a509ddf1831bf61fe597a2eb。官方 `_build_stgca_advantages` 公式片段保留在 `tests/fixtures/rlsd_official_formula.py`，仅用于数值回归，缩进改为顶层函数，公式体未改；来源指纹见 `reference/rlsd-formula-source.json`。上游 README 原件保留于 `reference/rlsd-upstream-README.md`，包含引用和许可说明；上游声明遵循 EasyR1 许可，本包不改变第三方授权。公式源文件的 Bytedance copyright 与 Apache-2.0 头已保留，Apache-2.0 全文见 `reference/LICENSE-APACHE-2.0.txt`。历史完整源码锁 `reference/rlsd-source-lock.json` 仅作来源凭证，其中 vendor 路径并非本版交付文件。

数据来自 LLM4Ops/Cloud-OpsBench、microsoft/OpenRCA、haoming29/CUJBench 与 ArtificialAnalysis/ITBench-AA，来源链接、版本和校验依据见数据说明与 dataset_sources。轻量包只携带案例清单、小型公开元数据及参考材料；大数据和 Qwen 权重由平台按各自发布条款下载。原始模型与数据的权利和许可归各发布方。

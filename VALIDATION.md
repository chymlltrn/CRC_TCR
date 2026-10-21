# Validation · 代码包验证

[English](#english) | [简体中文](#简体中文)

## English

The release checks verify code packaging and the stated equivalence boundaries. They do not constitute a new whole-study analysis.

| Module | Checks performed | Limits |
| --- | --- | --- |
| Bulk and figure components | Syntax, restored-literal AST comparisons, external-configuration isolation, and synthetic runtime tests; details in the module validation receipt | Embeddings, large models, full statistical pipelines, and assembled figure pages were not regenerated during packaging |
| Single-cell components | Configuration/dispatcher checks, matching and statistical synthetic tests, cohort/phenotype tests, and a C++ counting fixture | No RNA atlas, full matching scan, or large raw-count pipeline was rerun; some historical intermediate files require regeneration from their documented inputs |
| Revised qPCR | Seven numerical tables match the original exactly; four PDFs match in extracted text and rendered pixels at 144 dpi | PDF file bytes can differ because of generated metadata; real comparison inputs remain private |
| TCGA | Original and adapted scripts produce identical complete statistical tables from the same archived ssGSEA scores; historical main result tables agree within relative tolerance 1e-10 and absolute tolerance 1e-12; CDR conversion and synthetic ssGSEA checks also pass | Raw expression scoring and paired edgeR signature fitting were not repeated; cached-score validation does not verify upstream expression processing |

The current-revision audit records 35 checks. Real-data test inputs, result comparisons and full private receipts remain in the server archive. Only synthetic tests and non-individual validation metadata are included in this repository.

Before publication, the candidate tree is checked for actual sample identifiers and receptor sequences, credentials, binary data, notebook outputs, and paths that write into the source checkout. The final Git index and remote commit are checked separately from the working tree. Source manifests identify the original and adapted script hashes. The version tag identifies the published snapshot; server relocation journals and raw inventory listings are kept outside GitHub.

The bilingual public release also anonymizes residual sample identifiers in two historical scripts. Local-variable renaming and external function-name configuration are checked for equivalence; the release does not rerun the full scientific analysis. See the bulk module's [public-release validation note](pipelines/bulk_and_figures/PUBLIC_RELEASE_VALIDATION.md).

## 简体中文

发布检查验证代码整理及所声明的等价性范围，不代表重新开展整项研究分析。

| 模块 | 已执行检查 | 范围限制 |
| --- | --- | --- |
| Bulk 与图形组件 | 语法检查、恢复外置常量后的 AST 对比、外部配置隔离和合成运行测试；详情见模块验证回执 | 整理期间未重新生成嵌入、大型模型、完整统计流程或最终组装图页 |
| 单细胞组件 | 配置/调度器检查，匹配与统计合成测试，队列/表型测试及 C++ 计数测试样例 | 未重跑 RNA 图谱、全量匹配扫描或大型原始计数流程；部分历史中间文件需按输入说明重新生成 |
| 修订后的 qPCR | 7 张数值表与原结果完全一致；4 个 PDF 的提取文本及 144 dpi 渲染像素一致 | 生成元数据可造成 PDF 文件字节不同；真实对比输入保持私有 |
| TCGA | 原始与适配脚本使用同一归档 ssGSEA 分数时产生完全一致的完整统计表；历史主结果表在相对容差 1e-10、绝对容差 1e-12 内一致；CDR 转换及合成 ssGSEA 测试通过 | 未重复原始表达评分及配对 edgeR 签名拟合；基于缓存分数的验证不能验证上游表达处理 |

当前修订版审计记录了 35 项检查。真实数据测试输入、结果对比和完整私有回执保留在服务器归档中。本仓库仅包含合成测试和不涉及个体的验证元数据。

公开前检查覆盖真实样本标识、受体序列、凭据、二进制数据、notebook 输出，以及写入源代码目录的路径。Git 暂存区和远端提交独立于工作树进行检查。来源清单记录原始与适配脚本的哈希，版本标签标识发布快照；服务器迁移日志及原始数据清单不进入 GitHub。

本次双语公开版更新文档、访问说明及数据忽略规则，并对两份历史脚本的残留样本标识进行匿名化。局部变量重命名和外部函数名配置化另行完成等价性检查；本次未重新完成全量科学分析。详见 [bulk 模块公开版验证说明](pipelines/bulk_and_figures/PUBLIC_RELEASE_VALIDATION.md)。

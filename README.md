# CRC_TCR · 结直肠癌 TCR 分析代码

[English](#english) | [简体中文](#简体中文)

**Public code; Source data will be released separately later. / 代码已公开；Source data 将于后续单独发布。**

## English

This repository contains analysis and visualization code for the CRC TCR manuscript: TCR-BERT and SCEPTR embeddings, bulk repertoire analyses, single-cell links, qPCR, and TCGA follow-up analyses. The scientific snapshot corresponds to v29: five main figures, three supplementary figures, and TCGA results reported in text. The bilingual public release updates documentation and access and removes residual sample identifiers from two historical scripts. The scientific calculations are retained; the anonymization is checked separately from full scientific reruns.

Anyone can browse and download the code without an invitation. The repository contains source code, documentation, configuration templates, and synthetic tests. **Source data have not been released here.** Participant data, real TCR sequences, selected gene-signature results, pretrained weights, figures, actual configurations, and numerical result tables remain outside this repository. See [data availability](DATA_AVAILABILITY.md) for the separate release plan and [access and versions](REVIEW_ACCESS.md) for frozen snapshots.

### Find the analysis

| Manuscript content | Code and instructions |
| --- | --- |
| Bulk preprocessing, TCR-BERT/SCEPTR, atlas clusters, nodal models, MIL, Q1/Q2 candidates, figure assembly | [Bulk and figure pipelines](pipelines/bulk_and_figures/README.md) |
| RNA atlas, bulk-to-single-cell matching, independent/autologous blood detection, programs, local expansion, sharing, c27 selection | [Single-cell pipelines](pipelines/singlecell/README.md) |
| Revised qPCR, donor-paired state signatures, TCGA median-split analyses and statistical follow-up | [Current revision](pipelines/current_revision/README.md) |

Each module includes source manifests or provenance notes, entry points, input requirements, and runtime information. Historical directory names identify calculations that generated current panels; they do not define the final figure order.

### Running the code

1. Read the relevant module README and obtain authorized inputs separately. Keep inputs and output directories outside this checkout.
2. Prepare the documented Python/R environment for each stage. Different stages used different environments; no single environment is claimed to reproduce every stage.
3. Copy configuration examples outside the repository and fill in actual input, output, and software paths. Keep private sample mappings and candidate sequences in these external files.
4. Run stages in dependency order. Some require frozen atlas arrays, pretrained weights, reference databases, or prior intermediate results. These are not downloaded automatically; there is no unattended whole-study runner.

For example, `python pipelines/singlecell/run.py --list` lists single-cell entry points without reading study data. The bulk module dispatches individual source scripts. The current-revision module documents qPCR and signature/TCGA commands separately.

### Interpretation and reproduction boundaries

- Autologous blood detection, independent bulk-blood abundance, and local single-cell clonal expansion are distinct variables. Their definitions and eligible denominators are retained in the single-cell scripts.
- Revised qPCR excludes the specified abnormal participant as a whole and assigns zero to undetected relative quantity; Ct remains a nondetection. Actual eligibility mappings are external inputs.
- TCGA signatures originate from single-cell analysis. Final TCGA postprocessing is required for correction families and diagnostics. Historical TCGA plotting code is retained for audit; TCGA panels are not part of the current manuscript.
- Code produces analytical components and panel inputs. Final figure composition can require separately retained artwork. A complete pixel-identical reconstruction of all final figures from raw data was not rerun during packaging.
- [Validation](VALIDATION.md) distinguishes syntax checks, synthetic tests, and checks against authorized real inputs.

Upstream software and pretrained models retain their respective terms. Public access does not grant a new license to third-party components or study data.

## 简体中文

本项目提供 CRC TCR 文章对应的分析与可视化代码，覆盖 TCR-BERT 和 SCEPTR 嵌入、bulk TCR 分析、单细胞关联、qPCR 以及 TCGA 补充分析。科学内容对应 v29：5 张主图、3 张补图，TCGA 结果仅在正文中表述。本次中英双语公开版更新文档和访问方式，并清除两份历史脚本中的残留样本标识。科学计算保持原逻辑；匿名化等价性检查与全量科学分析重跑分别记录。

任何人均可直接浏览和下载代码，无需邀请。仓库包含源代码、说明文档、配置模板和合成测试。**Source data 尚未在此发布。** 参与者数据、真实 TCR 序列、筛选后的基因签名结果、预训练权重、图片、实际运行配置和数值结果表均保存在仓库之外。后续数据发布安排见[数据可用性说明](DATA_AVAILABILITY.md)，固定版本见[访问与版本说明](REVIEW_ACCESS.md)。

### 查找分析流程

| 文章内容 | 代码和说明 |
| --- | --- |
| bulk 预处理、TCR-BERT/SCEPTR、图谱聚类、淋巴结状态模型、MIL、Q1/Q2 候选及图形组装 | [Bulk 与绘图流程](pipelines/bulk_and_figures/README.md) |
| RNA 图谱、bulk 与单细胞匹配、独立/自体外周血检出、细胞程序、局部扩增、共享及 c27 筛选 | [单细胞流程](pipelines/singlecell/README.md) |
| 修订后的 qPCR、供者配对状态签名、TCGA 中位数分组和后续统计 | [当前修订版流程](pipelines/current_revision/README.md) |

各模块提供来源清单或溯源说明、运行入口、输入要求和环境信息。历史目录名用于定位当前图中结果的计算来源，不代表最终 Figure 顺序。

### 运行方法

1. 阅读对应模块的 README，另行获取有权使用的输入数据。输入和输出目录均应位于本代码目录之外。
2. 按各步骤说明准备 Python/R 环境。不同步骤使用的环境不同，没有一个统一环境被声明可复现全部步骤。
3. 将示例配置复制到仓库外，填写实际输入、输出和软件路径。私有样本映射及候选序列保存在这些外部配置中。
4. 按依赖顺序运行。部分步骤需要冻结的图谱矩阵、预训练权重、参考数据库或已有中间结果；代码不会自动下载这些资源，也没有无人值守的全研究一键运行入口。

例如，`python pipelines/singlecell/run.py --list` 可在不读取研究数据的情况下列出单细胞运行入口。bulk 模块提供逐脚本调度器，当前修订模块分别说明 qPCR 和签名/TCGA 命令。

### 解释与复现边界

- 自体外周血检出、独立 bulk 外周血丰度和局部单细胞克隆扩增是不同变量；对应定义和纳入分母保留在单细胞脚本中。
- 修订后的 qPCR 将指定异常参与者整例排除；相对表达量未检出记为 0，Ct 仍记为未检出。实际纳入映射由外部文件提供。
- TCGA 签名来源于单细胞分析。必须执行最终后处理，以确定多重检验校正范围和诊断结果。历史 TCGA 绘图代码用于核查，当前文章不包含 TCGA 图板。
- 代码生成分析组件和图板输入，最终图形组装可能需要另行保存的原始画稿。代码整理期间没有从原始数据重新生成全部最终图，也不声称所有图可达到逐像素一致。
- [验证说明](VALIDATION.md) 分别列出语法检查、合成测试和使用已授权真实输入的检查范围。

上游软件和预训练模型仍适用各自条款；仓库公开不向第三方组件或研究数据授予新的许可。

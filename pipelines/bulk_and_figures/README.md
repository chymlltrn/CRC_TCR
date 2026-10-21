# Bulk TCR analysis and figure components for manuscript v29

This is the public code release. Source data have not yet been released and will be provided separately at a later date. References below to private archives describe retained provenance and external inputs; those archives are not included in this repository.

本仓库公开发布代码；Source data 尚未发布，将在后续单独提供。下文提及的私有归档用于保留溯源记录和外部输入，不包含在本仓库中。

English documentation is followed by the complete Chinese translation. 英文说明之后提供完整中文说明。

This directory contains analysis code for Main Figures 1–2, Supplementary Figures 1 and 3, supporting Q1/Q2 and antigen computations, and final figure-package assembly. Single-cell computations and the current qPCR/TCGA revisions are owned by the other pipeline directories. The retained supplement mapping is old S1 → current S1, old S3 → current S2, old S5 → current S3.

`source_tree/` preserves the historical project-relative module structure. `source_manifest.json` records source scope, relative original location, original SHA256 and adapted SHA256. `local_only` means the source was recovered from the local results mirror; it was absent from the corresponding server project tree. The private archival package retains the unmodified originals.

## Entry points and dependencies

| Component | Entry relative to `source_tree/` | Required external inputs |
|---|---|---|
| Primary-library/QC ingest | `0929_caseEmbedding/atlas_340/src/01_ingest.py` | Frozen library ledgers, clinical/sample workbooks, MiXCR clone exports, cohort manifests |
| Original sequence banks | `0929_caseEmbedding/src/02_embed.py` | AA index/union, model weights and configuration; produces separate TCR-BERT/SCEPTR banks |
| Atlas bank extension | `0929_caseEmbedding/atlas_340/src/02_extend_embeddings.py` | Original banks, frozen atlas sequence index and new AA set |
| Full atlas clustering | `0929_caseEmbedding/atlas_340/leiden_full_20261001/src/full_leiden_graph.py` and `leiden_full_sceptr_20261002/src/full_leiden_graph.py` | PCA scores, graph-design freeze, matching sequence row index |
| Nodal associations | `0929_caseEmbedding/atlas_340/pooled_N_20261003/src/analyze_pooled.py` | Full patient-cluster abundance tables, known-N covariates and frozen family definition; imports helpers from the adjacent full-Leiden analysis |
| GLIPH2/TCRdist support | `0929_caseEmbedding/{src,atlas_340/src}/04_groups_stats.py` and `04b_tcrdist_projection.py` | Training-only groups/folds, official GLIPH2 reference/binary, clone tables and frozen dictionaries |
| Attention MIL | `0929_caseEmbedding/src/05_mil.py`; atlas `src/08_compare_mil.py` | Frozen fold assignments, patient AA draws, aligned representation banks and run receipts |
| REPO/CLUSTER3 models | `20261009_CMV/pn_ml_models_merged/src/run_ml_battery_merged.py` and cohort battery | Patient design, repertoire feature matrix, frozen feature-set definitions |
| CLUSTERTCR models | `20261010_article_v25/20261009_ML_SHAP/src/build_cluster_tcr_features.py`, then `run_cluster_tcr_battery.py` | Patient AA tables, frozen cluster labels, known-N design and selected feature/fold records |
| Corrected RF SHAP | `20261010_article_v25/revision_v27/ML_SHAP_corrected/scripts/repair_rf_shap.py` | Exact feature matrix/order, recorded fold selection, fixed OOF predictions and original battery source |
| Figure 2 and S1 composition | `20261010_article_v25/revision_v27/scripts/analysis_v26/build_figure2_s1.py` | Frozen original PDF/SVG artwork, all 108 model results, corrected SHAP table |
| Figure 1 and legacy supplement panels | `20261003_paper_v2/revision_v20_narrative_antigen_revision/code/plot_figure1_v19.py`, `plot_figure2_v20.py`, `plot_supplement_v20.py` | Frozen plot-ready source tables; Figure1 additionally requires its frozen baseline SVG |
| Q1/Q2 depth/portability | `20261003_paper_v2/revision_v18_figure_story_revision/reproducibility/historical_analysis/revision_v7/v7_q1_depth_n.py`, `v7_q1_depth_n_models.py` | Frozen candidate patient counts, cohort/features, original ridge fold audit and source `20261003_paper_v1/code/run_ridge.py` |
| Final eight-page assembly | `20261018_article_v27/revision_20261018_v29/figure_package/build_figure_package.py` | User-reordered main PDFs, corrected v28 Figure3/5, and old S1/S3/S5 PDFs |

Older `assemble.py`, `figure_assets.py`, `revision_config.py` and registration helpers remain because figure functions import them. They describe historical package layouts and are not the v29 release entry point. The v29 assembly is the authoritative final selector. Old qPCR rendering, 24-sample verification, and the superseded uncorrected SHAP renderer are excluded from this publication directory; their source originals remain in the private archive. The old whole-atlas `serial_run.py` orchestration is also excluded because it wrote audit/log outputs next to source; the graph, geometry and patient-statistics stages remain separately available.

## External configuration

No source table, matrix, clinical record, observed TCR/epitope/primer sequence, figure, fitted model or model weight is included here. Observed biological literals and individual sample identifiers have been moved to external JSON. Field names that encode individual exclusions are also supplied through external configuration.

1. Copy `config.example.json` **outside** the repository and populate the path fields for an isolated writable analysis workspace. `config_fields.json` lists required fields and use sites; `config.schema.json` describes their structure.
2. Supply the exact original strings in an external `private_literals` mapping, or point `private_literals_file` to an external JSON containing that mapping. The private archive includes the recovered values. Do not infer or replace biological identities from generic examples.
3. Restore the expected data and module layout in that isolated workspace. Several original engines dynamically import sibling files from configured project paths; those paths must contain the corresponding adapted code and frozen inputs. Historical `revision_v7/code` and `revision_v6/code` aliases must point to the recovered historical-analysis modules. A complete scientific rerun is a staged workflow, not an automatic consequence of opening the repository.
4. Invoke one intended component through the launcher, for example:

```sh
python run_module.py --config /absolute/private/config.json \
  0929_caseEmbedding/atlas_340/src/01_validate.py -- [component arguments]
```

The launcher adds the runtime helper to `PYTHONPATH` for child processes. Direct invocation requires the same `PYTHONPATH` and `CRC_TCR_BULK_CONFIG`. The runtime rejects configurations, private-literal files and configured input/output paths inside the code repository. It checks resolved locations against symlinks while preserving the configured spelling of an accepted path. There are no embedded fallback private values.

`20261003_paper_v2/code/q1_consolidate.py` optionally retrieves public GSE120101 GEO metadata from NCBI when the configured local cache is absent; the download is capped at 2 MiB. This is the optional online metadata path. Model loading uses local pretrained assets, and packaging itself does not retrieve data.

The private configuration snapshot preserves original locations for provenance; it is **not** a ready-to-run output plan. Set every output root to the isolated runtime before a rerun. The shell representation-comparison queue separately requires `CRC_TCR_ATLAS_ROOT` and is not started by packaging.

## Scientific and rendering boundaries

- CLUSTERTCR features use natural log encoding `-np.log((r + 1) / (tot + 1))` for present sequences and zero for absence. Earlier README wording describing log2 was incorrect. A larger present value reflects a lower read fraction; this is not a positive abundance score.
- The 339-primary-library analysis, 340 physical libraries, Home126/External163 known-N evaluation and 114-person MIL sensitivity have different denominators. Preserve their frozen identities and folds.
- Figure2f selects the best of twelve algorithms within each family/scenario. Its pointwise OOF bootstrap interval does not repeat model or cluster selection. The three cluster definitions were selected using pooled labels; pooled CV is not independent validation.
- GLIPH2 seed-derived motif/VJ projection is a derived classifier, not official full-repertoire GLIPH2 membership. Preserve negative and no-candidate outcomes. Sequence-reference/antigen predictions are not proof of antigen specificity.
- Use the corrected held-out RF SHAP scripts for Figure2g. The old SHAP path is retained only in private provenance.
- Current figures combine computed panels with author-selected/manual layouts and surgical PDF replacements. Frozen graphical inputs are necessary to reproduce those layouts. This package does not claim a single command can regenerate all eight final pages from raw measurements.
- The original Figure1 baseline SVG must be retained at `revision_v20_narrative_antigen_revision/reproducibility/frozen_v18_baseline/figures/main/Figure1.svg`. Figure2/S1 composition requires `revision_v27/SourceData/ml_figures/frozen/{Figure2_original,FigureS1_original}.{pdf,svg}`. These assets belong in the private data archive, not GitHub.

## Validation performed

`validation_report.json` records syntax checks and normalized AST comparison for every adapted Python source. The released source has been modified to externalize private values and paths, use generic local variable names, and resolve a private external-module attribute through configured `getattr`. After restoring external literals and that attribute, applying the documented local-variable alpha normalization and package-root repair, and excluding documentation redaction, the normalized ASTs match the archived originals. The path repair relocates the `build_figure2_s1.py` package root. These checks establish equivalence within the documented normalization rules; they do not claim byte-identical source. The configuration runtime has six synthetic tests covering separate private files, missing fields/config, relative paths and checkout exclusion. The additional identifier review and synthetic call tests are described in [PUBLIC_RELEASE_VALIDATION.md](PUBLIC_RELEASE_VALIDATION.md).

No large scientific analysis, embedding, model fitting, qPCR recalculation or figure regeneration was launched during code packaging. The configuration boundary tests establish packaging behavior; they do not establish fresh scientific results.


---

# 中文说明：v29 稿件的 bulk TCR 分析及图件组件

本目录包含主图 1–2、补充图 1 和 3 的分析代码、Q1/Q2 及抗原相关的支持分析，以及最终图件包的组装代码。单细胞计算和当前 qPCR/TCGA 修订由其他流程目录提供。保留的补充图编号映射为：旧 S1 → 当前 S1，旧 S3 → 当前 S2，旧 S5 → 当前 S3。

`source_tree/` 保留历史项目的相对模块结构。`source_manifest.json` 记录源代码范围、原始相对位置、原始 SHA256 和适配后 SHA256。`local_only` 表示代码从本地结果镜像恢复，在相应的服务器项目目录中不存在。未修改的原件保留在私有归档包中。

## 入口与依赖

下表中的入口路径均相对于 `source_tree/`；缩写路径沿用英文表中的模块上下文。

| 组件 | 相对于 `source_tree/` 的入口 | 所需外部输入 |
|---|---|---|
| 主分析文库及 QC 数据导入 | `0929_caseEmbedding/atlas_340/src/01_ingest.py` | 冻结的文库台账、临床/样本工作簿、MiXCR 克隆导出文件及队列清单 |
| 原始序列嵌入库 | `0929_caseEmbedding/src/02_embed.py` | AA 索引/并集、模型权重及配置；分别生成 TCR-BERT 和 SCEPTR 嵌入库 |
| 图谱嵌入库扩展 | `0929_caseEmbedding/atlas_340/src/02_extend_embeddings.py` | 原始嵌入库、冻结的图谱序列索引及新增 AA 集合 |
| 全图谱聚类 | `0929_caseEmbedding/atlas_340/leiden_full_20261001/src/full_leiden_graph.py` 和 `leiden_full_sceptr_20261002/src/full_leiden_graph.py` | PCA 得分、冻结的图构建方案及匹配的序列行索引 |
| 淋巴结状态关联 | `0929_caseEmbedding/atlas_340/pooled_N_20261003/src/analyze_pooled.py` | 完整的患者–聚类丰度表、已知 N 状态的协变量及冻结的检验家族定义；从相邻的全量 Leiden 分析导入辅助函数 |
| GLIPH2/TCRdist 支持分析 | `0929_caseEmbedding/{src,atlas_340/src}/04_groups_stats.py` 和 `04b_tcrdist_projection.py` | 仅由训练集确定的分组/折划分、官方 GLIPH2 参考文件/二进制程序、克隆表及冻结字典 |
| 注意力多实例学习（MIL） | `0929_caseEmbedding/src/05_mil.py`；图谱目录下的 `src/08_compare_mil.py` | 冻结的折划分、患者 AA 抽样结果、行对齐的表征嵌入库及运行回执 |
| REPO/CLUSTER3 模型 | `20261009_CMV/pn_ml_models_merged/src/run_ml_battery_merged.py` 及队列级模型批量评估流程 | 患者设计表、免疫组库特征矩阵及冻结的特征集定义 |
| CLUSTERTCR 模型 | `20261010_article_v25/20261009_ML_SHAP/src/build_cluster_tcr_features.py`，随后运行 `run_cluster_tcr_battery.py` | 患者 AA 表、冻结的聚类标签、已知 N 状态的分析设计及所选特征/折的记录 |
| 修正后的随机森林（RF）SHAP | `20261010_article_v25/revision_v27/ML_SHAP_corrected/scripts/repair_rf_shap.py` | 精确的特征矩阵及其顺序、已记录的折选择、固定的折外（OOF）预测和原始模型批量评估代码 |
| 图 2 与 S1 组装 | `20261010_article_v25/revision_v27/scripts/analysis_v26/build_figure2_s1.py` | 冻结的原始 PDF/SVG 图稿、全部 108 项模型结果及修正后的 SHAP 表 |
| 图 1 及历史补充图面板 | `20261003_paper_v2/revision_v20_narrative_antigen_revision/code/plot_figure1_v19.py`、`plot_figure2_v20.py`、`plot_supplement_v20.py` | 冻结的绘图源表；图 1 还需其冻结的基线 SVG |
| Q1/Q2 测序深度与可迁移性 | `20261003_paper_v2/revision_v18_figure_story_revision/reproducibility/historical_analysis/revision_v7/v7_q1_depth_n.py`、`v7_q1_depth_n_models.py` | 冻结的候选序列患者计数、队列/特征、原始 ridge 模型折划分审计，以及源码 `20261003_paper_v1/code/run_ridge.py` |
| 最终八页图件组装 | `20261018_article_v27/revision_20261018_v29/figure_package/build_figure_package.py` | 用户重新排序后的主图 PDF、修正后的 v28 图 3/5，以及旧 S1/S3/S5 PDF |

保留旧的 `assemble.py`、`figure_assets.py`、`revision_config.py` 和注册辅助程序，是因为绘图函数会导入它们。这些程序描述历史图件包的布局，不是 v29 的发布入口；最终采用哪些图件，以 v29 组装程序为准。旧 qPCR 绘图、24 样本核验及已被替代的未修正 SHAP 绘图程序不包含在本发布目录中，其源码原件保留在私有归档内。旧的全图谱编排程序 `serial_run.py` 也已排除，因为它会在源码旁写入审计/日志输出；图构建、几何诊断和患者统计三个阶段仍可分别使用。

## 外部配置

本目录不包含源数据表、矩阵、临床记录、实际观测到的 TCR/表位/引物序列、图件、拟合模型或模型权重。实际生物学字符串和个体样本标识符已移至外部 JSON；包含个体排除信息的字段名也通过外部配置提供。

1. 将 `config.example.json` 复制到仓库**之外**，并为独立、可写的分析工作区填写路径字段。`config_fields.json` 列出必需字段及使用位置；`config.schema.json` 描述其结构。
2. 在外部 `private_literals` 映射中提供准确的原始字符串，或将 `private_literals_file` 指向包含该映射的外部 JSON。私有归档保留了恢复的原始值。不得根据通用示例推断或替换生物学身份。
3. 在该独立工作区恢复预期的数据和模块布局。部分原始分析程序会从配置的项目路径动态导入相邻文件，因此这些路径必须包含对应的适配代码和冻结输入。历史别名 `revision_v7/code` 和 `revision_v6/code` 必须指向恢复的历史分析模块。完整的科学分析复跑需要分阶段执行，打开仓库不会自动完成复跑。
4. 通过启动器调用指定组件，例如：

```sh
python run_module.py --config /absolute/private/config.json \
  0929_caseEmbedding/atlas_340/src/01_validate.py -- [component arguments]
```

启动器会将运行时辅助模块加入子进程的 `PYTHONPATH`。直接调用也需要相同的 `PYTHONPATH` 和 `CRC_TCR_BULK_CONFIG`。运行时拒绝位于代码仓库内部的配置文件、私有字符串文件，以及配置的输入/输出路径；它会解析符号链接后检查实际位置，同时保留已接受路径的配置写法。代码中没有内嵌的私有值回退设置。

当配置的本地缓存不存在时，`20261003_paper_v2/code/q1_consolidate.py` 可选择从 NCBI 获取公开的 GSE120101 GEO 元数据，下载上限为 2 MiB。这是可选的在线元数据获取路径。模型加载使用本地预训练资源；代码打包本身不会获取数据。

私有配置快照保留原始位置以便溯源；它**不是**可直接运行的输出方案。复跑前须将每个输出根目录设置为独立运行工作区。用于表征比较的 shell 队列还要求单独设置 `CRC_TCR_ATLAS_ROOT`，打包过程不会启动该队列。

## 科学解释与图件重建边界

- CLUSTERTCR 特征对存在的序列采用自然对数编码 `-np.log((r + 1) / (tot + 1))`，缺失时为零。早期 README 中关于 log2 的表述有误。对于存在的序列，编码值越大表示读段比例越低，因此它不是随丰度升高而增大的分值。
- 339 个主分析文库、340 个物理文库、Home126/External163 的已知 N 状态评估，以及 114 人的 MIL 敏感性分析使用不同分母。须保留各自冻结的身份映射和折划分。
- 图 2f 在每个模型家族/场景内，从十二种算法中选择表现最佳者。其逐点 OOF bootstrap 区间未重复模型选择或聚类选择过程。三个聚类定义使用合并数据的标签选出，因此合并数据交叉验证不是独立验证。
- 基于 GLIPH2 种子分组得到的 motif/VJ 投影是派生分类器，不等同于官方 GLIPH2 对完整免疫组库给出的分组归属。须保留阴性及无候选结果。序列参考匹配或抗原预测不能证明抗原特异性。
- 图 2g 应使用修正后的留出集 RF SHAP 脚本。旧 SHAP 路径仅保留在私有溯源记录中。
- 当前图件结合了计算面板、作者选择/手工排版，以及局部 PDF 替换。重现这些布局需要冻结的图形输入。本代码包不声称能够用单条命令从原始测量重建全部八页最终图件。
- 图 1 的原始基线 SVG 必须保留在 `revision_v20_narrative_antigen_revision/reproducibility/frozen_v18_baseline/figures/main/Figure1.svg`。图 2/S1 的组装需要 `revision_v27/SourceData/ml_figures/frozen/{Figure2_original,FigureS1_original}.{pdf,svg}`。这些资源保存在私有数据归档中，不纳入 GitHub 代码发布。

## 已完成的验证

`validation_report.json` 记录每个适配后 Python 源文件的语法检查及规范化抽象语法树（AST）比较。发布源码已作修改：将私有值和路径外置、将局部变量改为通用名称，并通过配置驱动的 `getattr` 获取外部模块的私有属性。恢复外部字符串和该属性、应用已记录的局部变量 alpha 规范化及包根目录修复，并排除文档脱敏后，规范化 AST 与归档原件一致。路径修复针对 `build_figure2_s1.py` 的包根目录。这些检查确认的是已记录规范化规则内的等价性，不表示源码逐字节未变。配置运行时包含六项合成测试，覆盖独立私有文件、缺失字段/配置、相对路径及仓库内部路径拒绝等情况。追加的标识符审查与合成调用测试见 [PUBLIC_RELEASE_VALIDATION.md](PUBLIC_RELEASE_VALIDATION.md)。

代码打包期间未启动大型科学分析、嵌入计算、模型拟合、qPCR 重算或图件重绘。配置边界测试验证的是打包行为，不代表获得了新的科学结果。

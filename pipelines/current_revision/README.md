# Current revision analysis code

[English](#english) | [中文](#中文)

## English

**Publication status:** Code is public. Source data are planned for a separate later release; no release date has been specified. Real study inputs and private configurations remain outside this repository, with access conditions managed separately.

Run all commands below from `pipelines/current_revision/`.

Runnable source for the revised Q1 qPCR cohort, four frozen subtype signatures, and TCGA ssGSEA/stage/survival analyses. The final v29 manuscript uses **TCGA text only**; TCGA plots here reproduce the earlier exploratory review outputs and are optional.

This directory contains code, method documentation and synthetic/header-only interface examples. It contains no patient data, clinical tables, real sample lists, actual TCR sequences, expression matrices, frozen gene lists or original artwork. All inputs, output folders and real configuration files must remain outside the repository. The historical `reference_redraw_figure5.py` is intentionally excluded: the current qPCR entry reads a reference PDF, not that obsolete script.

### Runtime

The original server TCGA run recorded R 4.5.1 on Ubuntu 22.04.5, GSVA 2.2.1, BiocParallel 1.44.0, survival 3.8-6, data.table 1.18.4, ggplot2 4.0.3 and matrixStats 1.5.0; marker fitting recorded edgeR 4.6.2. Required executables: `Rscript`, `sha256sum`, `gzip`, and C++17 compiler with zlib for the raw-count streamer. Cairo support is needed for R PNG/SVG export. No dependencies are installed by these scripts.

qPCR depends on Python, NumPy, pandas, SciPy, Matplotlib and PyMuPDF (`fitz`); signature scripts use NumPy/pandas/SciPy and Matplotlib. The CDR converter additionally uses openpyxl. No original qPCR lock file was captured. The packaging verification environment was Python 3.9.6, NumPy 2.0.2, pandas 2.3.3, SciPy 1.13.1, Matplotlib 3.9.4, PyMuPDF 1.26.5. Local R validation used R 4.6.0 / GSVA 2.6.0 and the original cached scores; it does not replace the recorded server environment for full scoring. See `ENVIRONMENT.md`.

Copy `examples/environment.example.sh` and JSON examples to a location outside this repository, replace the synthetic placeholders with authorized inputs, and export their values. Example JSON is an interface illustration, not a reproduction dataset.

### qPCR

External `CRC_QPCR_ROOT/inputs/` requires:

- `qpcr_samples_original.tsv`: at least `sample_id`, `group`, `reported_copies_per_ng_rna`, `measurement_class`; source columns are retained in outputs. `group == Healthy` is Healthy; the other recorded groups are CRC.
- `qpcr_well_ct_original.tsv`: at least `sample_id`, `ct`, `entry_type`; numeric wells use `numeric_ct`, and other entries are nondetections. Configured legacy columns are removed.
- `qpcr_exploratory_values_original.tsv`: `sample_id`, `analysis_value`, for the frozen retained cohort; exact comparison prevents accidental value changes.
- `Figure5_v27_reference.pdf`: frozen one-page reference artwork; original geometry is required because the replacement rectangle is fixed. This external artwork may itself contain real sequence labels.

`CRC_QPCR_CONFIG` is JSON with ordered `cohorts.CRC` / `cohorts.Healthy` lists, `whole_sample_exclusions`, `expected_nd_sample_ids`, `drop_well_columns`; optional `expected_statistics` locks known aggregate outputs. The actual study exclusion and nondetection lists are supplied privately. Whole-sample exclusion applies to samples and wells. `not_detected` abundance is zero; no Ct value is imputed. All other source estimates remain unchanged. The conditional rank-sum test enumerates label allocations using tied midranks and twice the smaller inclusive tail. Detection Fisher counts are derived from retained rows.

```sh
python3 qpcr/build_qpcr_revision.py
```

Creates external `SourceData/`, `Figures/`, `QA/`, verifies unchanged artwork outside the c/d replacement rectangle, and writes data/hash receipts. `verify_preservation.py` is also called automatically in a fresh process. Real identifiers exist only in external inputs/configuration and generated private outputs.

### TCGA

`CRC_TCGA_DB_ROOT` must contain the original layout:

```text
tumor update 2026/<project>/output_mRNA_lncRNA_expr/<TCGA-project>_mrna_expr_tpm.csv
tumor update 2026/<project>/output_mRNA_lncRNA_expr/<TCGA-project>_clinical.csv
output-clinical/<project>/*_clinical_indexed.csv
output-clinical/<project>/*_clinicalXML_patient.csv
output-clinical/<project>/*_clinicalXML_followUp.csv
```

The first expression column is gene symbol; other columns are complete sample barcodes. RNA metadata contain `barcode`, `preservation_method`, `is_ffpe`. Clinical files use the GDC/indexed/XML field names read explicitly by `prepare_tcga.py`; no column inference or grade-to-stage substitution occurs. Use a fresh external output root. Copy `tcga/ANALYSIS_PLAN.md` there before preparation.

The official CDR workbook comes from the [GDC TCGA-CDR publication page](https://gdc.cancer.gov/about-data/publications/PanCan-Clinical-2018), download UUID `1b5f413e-a8d1-4d10-92eb-7c4ae739ed81`. Place it outside the repository. The converter verifies the frozen workbook SHA256 and preserves all rows/columns, including the redaction field; analysis excludes redacted CDR endpoints.

```sh
mkdir -p "$CRC_TCGA_ROOT"
cp tcga/ANALYSIS_PLAN.md "$CRC_TCGA_ROOT/ANALYSIS_PLAN.md"
python3 tcga/extract_cdr.py /external/TCGA-CDR-SupplementalTableS1.xlsx "$CRC_TCGA_ROOT/TCGA-CDR.tsv"
python3 tcga/prepare_tcga.py
Rscript tcga/run_tcga.R
Rscript tcga/postprocess_tcga.R
# Optional earlier-review figures, omitted from v29 manuscript:
Rscript tcga/plot_tcga.R
# On a runtime without Cairo, explicit vector-PDF-only export:
CRC_TCGA_PLOT_FORMATS=pdf Rscript tcga/plot_tcga.R
```

Set `CRC_SIGNATURE_GMT` to the independent frozen four-set GMT. Its required SHA256 is `32992a8abdefd4a990e1d3d470e2d18e2707bb8d9d34e0456da97c6622097465`; do not refit/reselect using TCGA outcomes. The analysis uses sample type 01, one primary sample per patient, non-FFPE/unknown/FFPE then lexicographic barcode selection, linear-scale duplicate-gene averaging, log2(TPM+1), and ssGSEA alpha=.25/normalize=TRUE. CRC combines COAD+READ; other projects are separate. Each cohort's median is fixed before endpoint missingness exclusion; High > median, Low <= median.

The code retains the locked eligibility gates, stage Fisher test, GDC/XML OS, separate CDR OS sensitivity and PFI, log-rank/Cox/PH outputs, and all unavailable/null/reverse results. **Always run postprocess:** it adds full-scan q-values, adjusted-model convergence flags and auxiliary stage-rank/CRC-project-stratified analyses. `figure_candidates.tsv` is a nominal adverse-direction presentation filter; it does not redefine the complete inference family. No independent prognostic or cell-abundance claim follows from these expression scores.

The original code reuses `scores/<cohort>_ssgsea_scores.tsv` if present. A normal full reproduction must start with no cache. Reusing a validated cache is suitable only for an explicitly documented statistics rerun with the same input GMT, expression cohort, score values and median groups. Cached-score use does not revalidate upstream expression or gene coverage.

### Signatures

The already frozen GMT is the input to manuscript TCGA results. Reconstructing signatures is a separate provenance workflow; use a **fresh** external `CRC_SIGNATURE_ROOT`, never the original frozen directory. `CRC_SIGNATURE_SOURCE` points to upstream baseline all-RNA pseudobulk files produced by the single-cell pipeline (`../singlecell`): `pseudobulk_metadata.tsv`, `gene_map.tsv`, `pseudobulk_counts.npz`, `raw_counts.bin`, `cell_group.bin`, `gene_map.bin`. Set `CRC_SIGNATURE_CONFIG` to the private donor-exclusion JSON. Source metadata must already have the documented baseline CRC tumor scope.

```sh
python3 signatures/prepare.py
mkdir -p /external/build
g++ -O3 -std=c++17 signatures/stream_detection.cpp -lz -o /external/build/stream_detection
# 36027 genes and 95 donor/state groups in this frozen study.
# The compile/invocation recipe is reconstructed from the source interface.
/external/build/stream_detection /external/GSE236581_counts.mtx.gz "$CRC_SIGNATURE_ROOT/inputs/gene_map.bin" "$CRC_SIGNATURE_ROOT/inputs/cell_group.bin" 36027 95 "$CRC_SIGNATURE_ROOT/inputs/raw"
Rscript signatures/fit_markers.R "$CRC_SIGNATURE_ROOT"
python3 signatures/release.py
python3 signatures/verify_delivery.py
# Optional earlier-review signature heatmap:
python3 signatures/plot_signatures.py
```

The stream keeps true detected-cell counts separately from summed UMI. Marker criteria remain paired edgeR QL, log2FC >.5, BH q <.05, donor support >=.7, target detection >=.1 and detection gap >=.05, excluded receptor/mitochondrial/ribosomal families, fixed specificity ranking and top30. The source scripts retain study-specific aggregate assertions; these are not a general arbitrary-dataset pipeline. Reconstructed freeze receipts are labelled as reconstructions, and never certify historical pre-outcome timing. The verifier checks hashes and selection invariants without claiming a fresh independent or visual review.

### Validation

```sh
python3 tests/test_core.py
Rscript tests/smoke_ssgsea.R
```

Private regression validation additionally compares the original and released qPCR seven source tables and four PDF renders, and original/released TCGA complete statistics, patient score manifests, warnings and auxiliary outputs using archived cached scores. No private fixtures are committed. Original script hashes and the full equivalence receipt remain in the separately archived `audit/` and `private_inputs/` trees. Full 1.31-billion-triplet streaming, paired edgeR refits and TCGA raw-TPM scoring were not repeated during packaging.

---

## 中文

### 当前修订版分析代码

本目录提供修订后 Q1 qPCR 队列、四套已冻结亚群表达签名，以及 TCGA ssGSEA、分期和生存分析的可运行源代码。最终 v29 稿件中的 **TCGA 内容仅以文字呈现**；此处的 TCGA 绘图代码用于复现此前探索性审阅图，属于可选步骤。

**发布安排：代码已公开；Source data（源数据）计划后续单独发布，日期待定。真实研究输入和私有配置保存在仓库外，其访问条件另行管理。** 仓库中的合成示例或仅含表头的示例只说明数据接口，不能替代真实数据完成研究结果复现。

本目录包含代码、方法说明和合成或仅含表头的接口示例，不包含患者数据、临床表、真实样本名单、真实 TCR 序列、表达矩阵、冻结基因列表或原始图稿。所有输入、输出目录和真实配置文件均应位于仓库之外。历史脚本 `reference_redraw_figure5.py` 未纳入当前入口；当前 qPCR 脚本读取参考 PDF。下列命令均在 `pipelines/current_revision/` 目录执行。

### 运行环境

原始服务器 TCGA 运行记录为 Ubuntu 22.04.5、R 4.5.1、GSVA 2.2.1、BiocParallel 1.44.0、survival 3.8-6、data.table 1.18.4、ggplot2 4.0.3 和 matrixStats 1.5.0；marker 拟合记录的 edgeR 版本为 4.6.2。所需可执行程序包括 `Rscript`、`sha256sum`、`gzip`，以及用于原始计数流式处理的 C++17 编译器和 zlib。R 导出 PNG/SVG 需要 Cairo 支持。脚本不会安装依赖。

qPCR 依赖 Python、NumPy、pandas、SciPy、Matplotlib 和 PyMuPDF（`fitz`）；签名脚本使用 NumPy、pandas、SciPy 和 Matplotlib；CDR 转换器另需 openpyxl。原始 qPCR 运行未保存环境锁定文件。代码打包核验使用 Python 3.9.6、NumPy 2.0.2、pandas 2.3.3、SciPy 1.13.1、Matplotlib 3.9.4 和 PyMuPDF 1.26.5。本地 R 核验使用 R 4.6.0 / GSVA 2.6.0，并复用原始缓存分数；该环境不能替代完整评分所记录的服务器环境。详见 [运行环境说明](ENVIRONMENT.md)。

将 `examples/environment.example.sh` 和 JSON 示例复制到仓库之外，用获准使用的输入替换合成占位值，再导出环境变量。示例 JSON 仅用于说明接口，不是复现数据集。

### qPCR

外部目录 `CRC_QPCR_ROOT/inputs/` 需要以下文件：

- `qpcr_samples_original.tsv`：至少包含 `sample_id`、`group`、`reported_copies_per_ng_rna` 和 `measurement_class`；源表列会保留在输出中。`group == Healthy` 归为 Healthy，其他已记录组别归为 CRC。
- `qpcr_well_ct_original.tsv`：至少包含 `sample_id`、`ct` 和 `entry_type`。数值 Ct 孔使用 `numeric_ct`，其他条目视为未检出；配置指定的历史列会被移除。
- `qpcr_exploratory_values_original.tsv`：包含冻结保留队列的 `sample_id` 和 `analysis_value`；通过精确比较防止意外改变数值。
- `Figure5_v27_reference.pdf`：冻结的单页参考图稿。替换矩形位置固定，因此必须保留原始页面几何尺寸；该外部图稿本身可能包含真实序列标签。

`CRC_QPCR_CONFIG` 指向 JSON 配置，其中包含有序的 `cohorts.CRC` / `cohorts.Healthy` 列表，以及 `whole_sample_exclusions`、`expected_nd_sample_ids` 和 `drop_well_columns`。可选字段 `expected_statistics` 用于锁定已知汇总输出。真实研究的排除和未检出名单由私有输入提供。整例排除同时作用于样本表和孔表；`not_detected` 的丰度按 0 计，不填补 Ct。其他来源估计值保持不变。条件秩和检验使用并列值的平均秩，枚举标签分配，并取较小的包含观测值的尾概率的两倍作为双侧 P 值。检出率 Fisher 检验的计数从保留记录计算。

```sh
python3 qpcr/build_qpcr_revision.py
```

运行后在外部目录生成 `SourceData/`、`Figures/` 和 `QA/`，核验 c/d 替换矩形之外的图稿未改变，并输出数据和哈希回执。程序还会在新进程中自动调用 `verify_preservation.py`。真实标识符只存在于外部输入、配置及生成的私有输出中。

### TCGA

`CRC_TCGA_DB_ROOT` 必须保留原有目录结构：

```text
tumor update 2026/<project>/output_mRNA_lncRNA_expr/<TCGA-project>_mrna_expr_tpm.csv
tumor update 2026/<project>/output_mRNA_lncRNA_expr/<TCGA-project>_clinical.csv
output-clinical/<project>/*_clinical_indexed.csv
output-clinical/<project>/*_clinicalXML_patient.csv
output-clinical/<project>/*_clinicalXML_followUp.csv
```

表达表第一列为基因符号，其余列为完整样本条形码。RNA 元数据包含 `barcode`、`preservation_method` 和 `is_ffpe`。临床文件使用 `prepare_tcga.py` 显式读取的 GDC/indexed/XML 字段名，不推断列名，也不以组织学分级替代分期。使用全新的外部输出目录，并在准备数据之前将 `tcga/ANALYSIS_PLAN.md` 复制到该目录。

官方 CDR 工作簿来自 [GDC TCGA-CDR 论文资源页](https://gdc.cancer.gov/about-data/publications/PanCan-Clinical-2018)，下载 UUID 为 `1b5f413e-a8d1-4d10-92eb-7c4ae739ed81`。文件须存放在仓库之外。转换器核验冻结工作簿的 SHA256，保留所有行列，包括撤回标记（redaction）字段；分析时排除被标记撤回的 CDR 终点记录。

```sh
mkdir -p "$CRC_TCGA_ROOT"
cp tcga/ANALYSIS_PLAN.md "$CRC_TCGA_ROOT/ANALYSIS_PLAN.md"
python3 tcga/extract_cdr.py /external/TCGA-CDR-SupplementalTableS1.xlsx "$CRC_TCGA_ROOT/TCGA-CDR.tsv"
python3 tcga/prepare_tcga.py
Rscript tcga/run_tcga.R
Rscript tcga/postprocess_tcga.R
# 可选：此前审阅图；不纳入 v29 稿件。
Rscript tcga/plot_tcga.R
# 运行环境缺少 Cairo 时，显式选择仅导出矢量 PDF。
CRC_TCGA_PLOT_FORMATS=pdf Rscript tcga/plot_tcga.R
```

将 `CRC_SIGNATURE_GMT` 指向独立冻结的四套签名 GMT，要求 SHA256 为 `32992a8abdefd4a990e1d3d470e2d18e2707bb8d9d34e0456da97c6622097465`，不得根据 TCGA 结局重新拟合或筛选签名。分析仅纳入样本类型 01，每位患者保留一个原发肿瘤样本；按非 FFPE、状态未知、FFPE 的优先级，再按条形码字典序选择样本。重复基因在线性 TPM 尺度取均值，随后进行 log2(TPM+1) 转换，并使用 ssGSEA alpha=.25、normalize=TRUE。CRC 合并 COAD+READ，其他项目分别分析。各队列中位数在剔除终点缺失记录前固定：High > median，Low <= median。

代码保留预先锁定的纳入门槛、分期 Fisher 检验、GDC/XML OS、独立的 CDR OS 敏感性分析及 PFI、log-rank/Cox/PH 输出，以及所有不可分析、无显著差异和反向结果。**必须运行 postprocess**：该步骤补充全扫描 q 值、调整模型的收敛标记，以及有序分期秩检验和 CRC 项目分层辅助分析。`figure_candidates.tsv` 仅按名义显著且不良方向筛选展示候选，不重新定义完整检验家族。这些表达分数不能直接支持独立预后或细胞丰度结论。

原始代码在存在 `scores/<cohort>_ssgsea_scores.tsv` 时复用缓存。常规完整复现必须从无缓存状态开始。只有在输入 GMT、表达队列、分数值和中位数分组一致，且明确记录为统计重跑时，才适合复用经过核验的缓存。使用缓存分数不会重新核验上游表达数据或基因覆盖率。

### 表达签名

稿件 TCGA 结果使用已经冻结的 GMT。重新构建签名属于独立的来源追溯流程，必须使用**全新的**外部 `CRC_SIGNATURE_ROOT`，不得写入原始冻结目录。`CRC_SIGNATURE_SOURCE` 指向单细胞流程（`../singlecell`）产生的基线 all-RNA pseudobulk 文件：`pseudobulk_metadata.tsv`、`gene_map.tsv`、`pseudobulk_counts.npz`、`raw_counts.bin`、`cell_group.bin` 和 `gene_map.bin`。`CRC_SIGNATURE_CONFIG` 指向私有供者排除 JSON。源元数据必须已经满足记录的基线 CRC 肿瘤范围。

```sh
python3 signatures/prepare.py
mkdir -p /external/build
g++ -O3 -std=c++17 signatures/stream_detection.cpp -lz -o /external/build/stream_detection
# 本次冻结研究包含 36027 个基因和 95 个供者/状态分组。
# 此编译和调用方式根据源码接口重建。
/external/build/stream_detection /external/GSE236581_counts.mtx.gz "$CRC_SIGNATURE_ROOT/inputs/gene_map.bin" "$CRC_SIGNATURE_ROOT/inputs/cell_group.bin" 36027 95 "$CRC_SIGNATURE_ROOT/inputs/raw"
Rscript signatures/fit_markers.R "$CRC_SIGNATURE_ROOT"
python3 signatures/release.py
python3 signatures/verify_delivery.py
# 可选：此前审阅用的签名热图。
python3 signatures/plot_signatures.py
```

流式处理将真实检出细胞数与 UMI 总计数分开保存。marker 规则保持为配对 edgeR QL、log2FC >.5、BH q <.05、供者同向支持比例 >=.7、目标检出率 >=.1、检出率差 >=.05；排除受体、线粒体和核糖体基因家族，按固定特异性规则排序并取 top30。源码保留本研究特有的汇总数量断言，因此不是面向任意数据集的通用流程。重建后的冻结回执明确标记为重建，不证明历史上在查看结局前完成冻结。核验器检查哈希和筛选不变量，不声称进行了新的独立审查或视觉审查。

### 验证

```sh
python3 tests/test_core.py
Rscript tests/smoke_ssgsea.R
```

私有回归验证还比较原版与发布版 qPCR 的七张源表和四个 PDF 渲染，以及利用历史缓存分数得到的原版/发布版 TCGA 完整统计、患者分数清单、警告和辅助输出。私有测试输入不提交到仓库。原始脚本哈希和完整等价性回执保存在另行归档的 `audit/` 与 `private_inputs/` 目录。代码打包期间未重跑完整的 13.1 亿条三元组流式处理、配对 edgeR 拟合或 TCGA 原始 TPM 评分。

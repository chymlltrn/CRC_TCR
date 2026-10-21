# Single-cell and peripheral-blood analysis code / 单细胞与外周血分析代码

[English](#english) | [中文](#中文)

**Availability: public code; Source data not yet released.** Historical versions and verification records are retained for provenance. The data release will be documented separately.

**发布状态：代码公开，Source data 尚未发布。** 历史版本和验证记录作为溯源依据保留；数据发布将另行说明。

## English

This package contains the source calculations and plotting components used for v29 main Figures 3–4 and Supplementary Figure 2, plus the exploratory c27 localization and the raw-count pseudobulk preparation upstream of the four frozen signatures. The four current signature-selection scripts are maintained separately in [current_revision](../current_revision/README.md).

The package is **code only**. No receptor sequences, donor/sample records, cell matrices, clinical tables, patient contrasts or figure source tables are included. The source files were inspected and copied; no full scientific analysis was rerun during this release. The finalized multi-panel artwork contains later PDF/Illustrator assembly. These scripts reproduce calculations and components from their documented inputs; this is not a claim of a one-command reconstruction of the final page layout.

### Configuration and execution

Use Python 3.10 or newer for the historical analysis scripts. The RNA embedding used a separate Python 3.11 environment. Copy `config.example.json` **outside the repository**, and fill paths and private constants using the locally retained source configuration and `config.schema.json`. The example intentionally has empty values. Set `CRC_TCR_CONFIG` to that file. The private source-default file is a provenance record, not a ready-to-run configuration: its output locations point at historical sources. Select fresh execution directories and stage the documented inputs under the expected layout before running.

All paths in the configuration must be absolute and outside this repository. Some historical scripts resolve sibling paths from a configured paper root, so retain the `single_cell/`, `literature/`, `revision_v5/`, `revision_v6/`, and other stated subdirectories. Preserve protocol documents, receipts and exact input hashes; do not weaken their assertions to make a new dataset pass. Code hashes change with release adaptation: original self-hash receipts must be regenerated for an authorized rerun, while the scientific input/protocol hashes remain enforced.

```bash
python run.py --list
python run.py --check-config
python run.py bulk_sc_match_ingest -- --root /absolute/new-paper-root/revision_v5 --stage all
python run.py bulk_sc_analysis -- --root /absolute/new-paper-root/revision_v5 --run
python run.py v6_cohort_sharing -- cohort
python run.py v6_cohort_sharing -- sharing
python run.py v6_cohort_sharing -- longitudinal
python run.py v6_phenotypes_bridge -- --root /absolute/new-paper-root/revision_v6 --run --cohort-receipt /absolute/new-paper-root/revision_v6/cohort/cohort_receipt.json
python run.py bulk_abundance_states -- preflight
python run.py bulk_abundance_states -- analysis
```

Run these commands from `pipelines/singlecell/`; `/absolute/...` values are placeholders for configured locations outside the repository. These are entry interfaces, not an automatically executable workflow. The cohort producer requires pre-existing protocol and verification inputs. `single_cell_analysis`, `single_cell_rna`, `single_cell_statistics`, and `prepare_tcr_umap_inputs` have top-level execution; **do not import them merely to inspect dependencies or request help**. `prepare_tcr_umap_inputs` additionally checks its project working directory. `run.py --list` and the interface tests are data-free.

### Dependency order and scientific scope

1. `legacy/code/single_cell_analysis.py` reads the GSE236581 metadata/VDJ files, official clinical workbook and frozen historical candidate dictionary. Its sibling `single_cell_common.py` defines productive chain acceptance, allele-stripped V/J identity and the six fixed RNA programs. It writes RNA/TCR inventories, the matched-patient registry and state fractions. `single_cell_rna.py` reads the complete raw MatrixMarket counts to derive the fixed log1p(CP10K) marker-program scores. `single_cell_statistics.py` produces patient-level tissue contrasts. The original RNA reader loads the large sparse matrix before extracting genes; it is memory-intensive.
2. `legacy/revision_v5/code/bulk_sc_match_ingest.py` reads the 339-library frozen roster, clone-detail Parquets, sample context and single-cell inventories. It scans complete files, records QC denominators, and produces AA/V/J, NT/V/J and AA-only matching support. `bulk_sc_analysis.py` produces patient-level matched–unmatched state/program contrasts, including the equal-clonotype program sensitivity used in S2b. Both need existing protocol/receipt inputs. The adjacent tests and verifier are retained.
3. `legacy/revision_v6/code/v6_cohort_sharing.py` defines the 19-patient baseline triplet cohort, same-donor blood detection, depth-matched self–other sharing and longitudinal tracking. Its selected on-treatment visit mapping is external private configuration. `v6_phenotypes_bridge.py` produces local expansion, Auto/Inde fractions, own-blood and composition-standardized contrasts. Auto/Inde uses exact AA/V/J on both sources; Auto is observed in the same patient's single-cell blood, while independent bulk detection requires at least two qualifying reads in a library from the independent 339-person reference. Own-blood primary analyses elsewhere use NT/V/J and at least one blood cell. These are different analysis definitions.
4. `legacy/revision_v16_core_narrative/code/bulk_abundance_states.py` is byte-identical at source to its v14 predecessor. Positive identity-level mean CPM values are split into tertiles, with zero/unmatched kept separate. Combined abundance includes zero contributions across all 339 bulk donors. The primary c03/c20 selection is the complete 32-state × two-tissue high-minus-low BH64 family, not cell-level correlation or gene-expression high/low. Conventional-CD8 programs exclude c26/c27 MAIT states.
5. `legacy/revision_v7/code/v7_antigen_bridge.py` supplies the exact-AA/V/J viral-reference context used in Figure 3. It needs externally supplied normalized antigen references and the frozen bridge membership. Its historical full `main()` also emits candidate and GLIPH inputs; those additional inputs are required if using that entry. Antigen-reference matches are annotation context, not verified antigen binding. The normalization of the antigen-reference database is a separate upstream bulk/antigen lane.
6. `umap/prepare_tcr_umap_inputs.py` provides accepted-chain and library-resolved matching evidence. `umap/recompute_embedding.py` recomputes RNA geometry for the fixed 369,063 scTCR-covered RNA T cells; `render_umaps.py` joins the existing subtype/CPM CSVs to these coordinates. The existing display CSVs are required inputs; their historical kernel export is not recreated by these three files. The 369,063-cell display scope includes all retained tissues/time points and is distinct from the baseline 19-patient inference scope.
7. `c27/build_cluster_mapping.py` maps the three predefined sequence-cluster labels using explicit sequence indices, and `analyze_subtype_preference.py` summarizes subtype localization with donor bootstrap support. It needs the atlas AA union, BERT/SCEPTR label arrays, accepted beta-chain keys, RNA/CPM plot tables and cluster statistics/context tables. These are exploratory c27 localization, not a direct continuation of the conventional-CD8 effects in Figures 3–4.
8. `signature_upstream/signature_prepare.py` and `signature_stream.cpp` build raw all-RNA donor/state pseudobulks from GSE236581. This upstream stage originally spans 20 CRC donors. The current signature lane separately restricts the frozen 19-patient baseline tumor scope before marker fitting. Never substitute the older combined c26+c27 MAIT identity signature for the current c27 signature.

### Required inputs and excluded outputs

Provide GSE236581's `GSE236581_CRC-ICB_metadata.txt.gz`, `GSE236581_VDJ_merge.txt.gz`, counts, barcodes and features; official sample/patient metadata; the 339-library cohort registry and clone details; protocol documents and original producer/verification receipts; reference annotation tables; frozen figure source tables; and the separately prepared UMAP CSVs. Exact absolute provenance and per-script input references are in the external audit map and `source_manifest.json`. No restricted dataset is downloaded by the adapter.

All output inventories, clonotype keys, per-library supports, pseudobulks, cell/donor metadata, patient contrast tables and figures must remain outside the repository. The `.gitignore` is a secondary guard; it does not replace an explicit code allowlist. Private configuration and original code snapshots are stored outside the repository. This public code repository does not distribute raw participant data. Source data have not yet been released and will be addressed in a later release.

### Plotting and signature interface

Historical Figure 4 became current Figure 3 after reordering. `plot_figure34_v18.py` and `plot_figure4_v20.py` consume frozen tables; their historical figure labels are preserved for traceability. `plot_supplement_v20.py` now runs only its historical S3 routine, which is current v29 S2; unrelated S1/S2/S4/S5/S6 functions were removed from this scoped copy. It expects the existing v19 source exports, including the standardized autologous and independent-bulk equal-clonotype contrasts, plus depth-500 longitudinal trajectories.

`current_figure3/build_figure3_revision.py` checks full BH64 state families and redraws only panel 3f on the finalized baseline PDF. It uses the adjacent frozen renderer and verifies that pixels outside the replacement rectangle are unchanged. This entry reproduces the v28 selection/plot artifact. Running it now does not establish a new pre-TCGA freeze; preserve the original dated lock and SHA records.

The upstream raw-count executable is built by `signature_prepare.py`:

```bash
c++ -O3 -std=c++17 signature_upstream/signature_stream.cpp -lz -o /absolute/build/signature_stream
/absolute/build/signature_stream raw_counts.mtx.gz gene_map.bin cell_group.bin NUMBER_OF_GENES NUMBER_OF_GROUPS output_prefix
```

The current four-signature GMT has SHA256 `32992a8abdefd4a990e1d3d470e2d18e2707bb8d9d34e0456da97c6622097465`. Its c03/c20 sets are primary bulk-associated states, c08 is a prespecified secondary extension and c27 has a separate exploratory rationale. The marker-selection scripts and edgeR model live in [current_revision](../current_revision/README.md). These scores do not measure cell abundance or establish independent prognostic value.

### Environment and verification limits

Core dependencies: NumPy, pandas, SciPy, PyArrow, Biopython, openpyxl and threadpoolctl; statsmodels for abundance/verification. Plotting additionally needs Matplotlib, PyMuPDF (`fitz`), lxml and Pillow. The RNA embedding additionally needs anndata, scanpy, harmonypy, umap-learn, pynndescent, igraph, leidenalg, scikit-learn and scikit-misc. Raw-count streaming needs a C++17 compiler and zlib. No dependencies were installed by this release task.

The historical v5 analysis receipt records Python 3.10.12. The UMAP parameter receipt records Python 3.11.17 with scanpy 1.11.5, anndata 0.12.19, NumPy 2.4.6, SciPy 1.17.1, harmonypy 2.0.2, umap-learn 0.5.12, pynndescent 0.6.0, igraph 1.0.0, leidenalg 0.12.0, scikit-learn 1.9.1 and scikit-misc 0.5.2. These are observed provenance, not a newly tested universal environment lock. Current signature fitting records R 4.5.1, edgeR 4.6.2, limma 3.66.0 and data.table 1.18.4 in its separate lane.

Run data-free adapter checks with `python -m unittest discover -s tests`. The release audit records Python compilation, configuration isolation, static dependency resolution and C++ syntax checks. Full data access, scientific reruns and exact final-artwork reproduction were not performed. The old remote `20261003_paper_v2` directory is unavailable at audit time; selected unmodified local frozen scripts were preserved separately. Recover the required data/protocol artifacts before attempting historical recomputation.

Release validation completed in existing remote Python 3.10.12 (NumPy 1.26.4, pandas 2.1.4, SciPy 1.13.0, PyArrow 23.0.1, Biopython 1.79, statsmodels 0.14.3): eight adapter/interface checks, four matching tests, ten patient-statistics checks, thirteen cohort/sharing checks, thirteen phenotype checks, C++ compilation and three stream-count fixture checks passed. The statistical synthetic harness uses an artificial protocol and matching in-memory hash; production protocol checks remain unchanged. See `verification.json`.

## 中文

本包包含 v29 主图 3–4、补充图 2 所用的计算与绘图组件，以及探索性 c27 亚群定位和四套冻结表达签名上游的原始计数 pseudobulk（伪总体）准备代码。当前四套签名的筛选脚本单独维护在 [current_revision](../current_revision/README.md)。

本包**仅发布代码**，不包含受体序列、供者/样本记录、细胞矩阵、临床表、患者层级对比结果或作图源表。本次代码发布检查并复制了源码，没有重新运行完整科学分析。最终多面板图还经过后续 PDF/Illustrator 组装。这些脚本使用所列输入重现计算与图形组件，不代表能够用一条命令重建最终整页排版。

### 配置与执行

历史分析脚本使用 Python 3.10 或更新版本；RNA 嵌入计算使用独立的 Python 3.11 环境。将 `config.example.json` 复制到**仓库外部**，依据本地保留的原始配置和 `config.schema.json` 填写路径及私有常量。示例配置有意保留为空。将 `CRC_TCR_CONFIG` 指向该文件。私有的原始默认配置用于溯源，不能直接作为运行配置：其中的输出位置指向历史来源目录。运行前应选择新的执行目录，并按预期目录结构准备所需输入。

配置中的所有路径必须是仓库外部的绝对路径。部分历史脚本根据配置的论文根目录解析同级路径，因此需保留 `single_cell/`、`literature/`、`revision_v5/`、`revision_v6/` 等指定子目录。应保留协议文件、执行/验证记录以及输入文件的精确哈希，不能削弱断言来使新数据集通过。代码经过发布适配后，代码哈希会改变：经授权重新运行时，需重新生成记录脚本自身哈希的验证文件，同时继续强制检查科学输入和协议的哈希。

```bash
python run.py --list
python run.py --check-config
python run.py bulk_sc_match_ingest -- --root /absolute/new-paper-root/revision_v5 --stage all
python run.py bulk_sc_analysis -- --root /absolute/new-paper-root/revision_v5 --run
python run.py v6_cohort_sharing -- cohort
python run.py v6_cohort_sharing -- sharing
python run.py v6_cohort_sharing -- longitudinal
python run.py v6_phenotypes_bridge -- --root /absolute/new-paper-root/revision_v6 --run --cohort-receipt /absolute/new-paper-root/revision_v6/cohort/cohort_receipt.json
python run.py bulk_abundance_states -- preflight
python run.py bulk_abundance_states -- analysis
```

以上命令应在 `pipelines/singlecell/` 目录下运行；`/absolute/...` 是仓库外配置路径的占位符。这些命令展示入口接口，尚不构成可直接自动执行的完整工作流。队列生成程序要求预先提供协议和验证输入。`single_cell_analysis`、`single_cell_rna`、`single_cell_statistics` 和 `prepare_tcr_umap_inputs` 会在顶层执行；**不要仅为检查依赖或查看帮助而导入这些模块**。`prepare_tcr_umap_inputs` 还会检查项目工作目录。`run.py --list` 和接口测试不需要科学数据。

### 依赖顺序与科学分析范围

1. `legacy/code/single_cell_analysis.py` 读取 GSE236581 元数据/VDJ 文件、官方临床工作簿以及冻结的历史候选字典。同目录的 `single_cell_common.py` 定义 productive 链的接受规则、去除等位基因后缀后的 V/J 身份和六套固定 RNA 程序。该步骤输出 RNA/TCR 清单、配对患者登记表及细胞状态比例。`single_cell_rna.py` 读取完整的原始 MatrixMarket 计数，计算固定的 log1p(CP10K) 标志基因程序评分。`single_cell_statistics.py` 生成患者层级的组织对比结果。原始 RNA 读取程序先加载大型稀疏矩阵，再提取基因，内存需求较高。
2. `legacy/revision_v5/code/bulk_sc_match_ingest.py` 读取冻结的 339 个文库名单、克隆明细 Parquet、样本背景信息和单细胞清单。它扫描完整文件，记录 QC 分母，并生成 AA/V/J、NT/V/J 和仅 AA 的匹配支持结果。`bulk_sc_analysis.py` 生成患者层级的匹配与未匹配组之间的状态/程序对比，包括补充图 S2b 使用的克隆型等权程序敏感性分析。两者均需要已有协议及执行/验证记录。同目录的测试和验证程序也予以保留。
3. `legacy/revision_v6/code/v6_cohort_sharing.py` 定义 19 名患者的基线三联样本队列、同一供者外周血检出、深度匹配后的自身与其他供者共享情况，以及纵向追踪。选定的治疗中访次映射通过外部私有配置提供。`v6_phenotypes_bridge.py` 生成局部扩增、Auto/Inde 比例、自身外周血相关对比及细胞组成标准化后的对比。Auto/Inde 的两种来源均采用精确 AA/V/J 匹配：Auto 表示在同一患者的单细胞外周血中检出；独立 bulk 检出则要求在独立的 339 人参考队列中，至少一个文库内有两条合格 reads。其他自身外周血主分析采用 NT/V/J 匹配，并要求至少一个血液细胞检出。上述分析的定义不同。
4. `legacy/revision_v16_core_narrative/code/bulk_abundance_states.py` 的原始源码与 v14 前序版本逐字节一致。身份层级的正值平均 CPM 按三分位数分箱，零值/未匹配组单独保留。合并丰度计算包含全部 339 名 bulk 供者的零贡献。c03/c20 的主分析筛选来自完整的 32 个状态 × 两种组织的 high-minus-low BH64 检验家族，并非细胞层级相关或基因表达高低分组。常规 CD8 程序排除 c26/c27 MAIT 状态。
5. `legacy/revision_v7/code/v7_antigen_bridge.py` 提供图 3 使用的精确 AA/V/J 病毒参考注释。它需要外部提供的标准化抗原参考表和冻结的 bridge 成员表。历史完整 `main()` 还会输出候选及 GLIPH 输入；运行该入口时也必须提供相应的额外输入。抗原参考匹配提供注释背景，不能视为已经验证的抗原结合。抗原参考数据库的标准化属于另一个上游 bulk/抗原处理分支。
6. `umap/prepare_tcr_umap_inputs.py` 提供已接受链和按文库区分的匹配证据。`umap/recompute_embedding.py` 为固定的 369,063 个具有 scTCR 覆盖的 RNA T 细胞重新计算 RNA 嵌入坐标；`render_umaps.py` 将已有亚群/CPM CSV 与这些坐标连接。已有展示 CSV 是必需输入；这三个文件不重建其历史共享内核导出过程。369,063 个细胞的展示范围包含保留的全部组织/时间点，与基线 19 名患者的统计推断范围不同。
7. `c27/build_cluster_mapping.py` 使用显式序列索引映射三个预定义序列簇标签，`analyze_subtype_preference.py` 则通过供者层级 bootstrap 汇总亚群定位。它需要 atlas AA 并集、BERT/SCEPTR 标签数组、已接受 β 链键值、RNA/CPM 作图表，以及簇统计/背景信息表。这些分析用于探索性 c27 定位，并非图 3–4 中常规 CD8 效应的直接延续。
8. `signature_upstream/signature_prepare.py` 和 `signature_stream.cpp` 根据 GSE236581 的全部 RNA T 细胞生成供者/状态层级的原始计数 pseudobulk。该上游步骤最初涵盖 20 名 CRC 供者。当前签名分析分支在拟合标志基因前，另行限制到冻结的 19 名患者基线肿瘤范围。不能用旧版合并 c26+c27 的 MAIT 身份签名替代当前 c27 签名。

### 所需输入与不纳入仓库的输出

需准备 GSE236581 的 `GSE236581_CRC-ICB_metadata.txt.gz`、`GSE236581_VDJ_merge.txt.gz`、计数、条形码和基因特征文件；官方样本/患者元数据；339 文库队列登记表及克隆明细；协议文件以及原始生成和验证记录；参考注释表；冻结的作图源表；以及单独准备的 UMAP CSV。精确绝对来源和逐脚本输入引用记录于仓库外审计映射及 `source_manifest.json`。适配器不会下载受限数据集。

所有输出清单、克隆型键值、逐文库支持结果、pseudobulk、细胞/供者元数据、患者对比表和图形都必须存放在仓库外部。`.gitignore` 仅提供辅助保护，不能替代明确的代码文件允许清单。私有配置和原始代码快照保存在仓库外部。本公开代码仓库不分发原始参与者数据。Source data 尚未发布，后续发布时将另行说明。

### 绘图与签名接口

调整顺序后，历史图 4 成为当前图 3。`plot_figure34_v18.py` 和 `plot_figure4_v20.py` 使用冻结表格；为便于追溯，保留了历史图号。`plot_supplement_v20.py` 现在仅执行历史 S3 例程，对应当前 v29 的 S2；与本次范围无关的 S1/S2/S4/S5/S6 函数已从该发布副本移除。该程序需要已有 v19 源数据导出，包括标准化后的自体外周血对比、独立 bulk 的克隆型等权对比，以及深度 500 的纵向轨迹。

`current_figure3/build_figure3_revision.py` 检查完整的 BH64 状态检验家族，并仅在最终基准 PDF 上重绘面板 3f。它使用相邻的冻结绘图程序，并验证替换矩形之外的像素保持不变。该入口重现 v28 的筛选/绘图产物。现在重新运行不能建立新的 TCGA 分析前冻结记录；应保留原始带日期的锁定文件及 SHA 记录。

上游原始计数可执行程序由 `signature_prepare.py` 编译生成：

```bash
c++ -O3 -std=c++17 signature_upstream/signature_stream.cpp -lz -o /absolute/build/signature_stream
/absolute/build/signature_stream raw_counts.mtx.gz gene_map.bin cell_group.bin NUMBER_OF_GENES NUMBER_OF_GROUPS output_prefix
```

当前四套签名 GMT 的 SHA256 为 `32992a8abdefd4a990e1d3d470e2d18e2707bb8d9d34e0456da97c6622097465`。其中 c03/c20 集合来自与 bulk 丰度关联的主分析状态；c08 是预设的次要扩展，c27 则具有独立的探索性筛选依据。标志基因筛选脚本和 edgeR 模型位于 [current_revision](../current_revision/README.md)。这些评分不能用于测量细胞丰度，也不能确立独立预后价值。

### 环境与验证边界

核心依赖包括 NumPy、pandas、SciPy、PyArrow、Biopython、openpyxl 和 threadpoolctl；丰度分析/验证还使用 statsmodels。绘图另需 Matplotlib、PyMuPDF（`fitz`）、lxml 和 Pillow。RNA 嵌入另需 anndata、scanpy、harmonypy、umap-learn、pynndescent、igraph、leidenalg、scikit-learn 和 scikit-misc。原始计数流式处理需要支持 C++17 的编译器和 zlib。本次代码发布没有安装新依赖。

历史 v5 分析记录的 Python 版本为 3.10.12。UMAP 参数记录为 Python 3.11.17，配合 scanpy 1.11.5、anndata 0.12.19、NumPy 2.4.6、SciPy 1.17.1、harmonypy 2.0.2、umap-learn 0.5.12、pynndescent 0.6.0、igraph 1.0.0、leidenalg 0.12.0、scikit-learn 1.9.1 和 scikit-misc 0.5.2。这些版本是已有执行来源记录，不能视为本次新验证的通用环境锁定文件。当前签名拟合分支另行记录 R 4.5.1、edgeR 4.6.2、limma 3.66.0 和 data.table 1.18.4。

使用 `python -m unittest discover -s tests` 可运行不需要科学数据的适配器检查。代码发布审计记录了 Python 编译、配置隔离、静态依赖解析和 C++ 语法检查。没有执行完整数据访问、科学分析重跑或最终图稿的精确重建。历史代码发布审计时，旧远端 `20261003_paper_v2` 目录不可用；选定的未修改本地冻结脚本已单独保留。尝试历史重算前，应先恢复所需数据和协议文件。

发布验证在已有远端 Python 3.10.12 环境中完成，依赖版本为 NumPy 1.26.4、pandas 2.1.4、SciPy 1.13.0、PyArrow 23.0.1、Biopython 1.79 和 statsmodels 0.14.3。八项适配器/接口检查、四项匹配测试、十项患者统计检查、十三项队列/共享检查、十三项表型检查、C++ 编译和三项流式计数测试均通过。统计合成测试使用人工协议及内存中与之匹配的哈希；生产代码的协议检查保持不变。详见 `verification.json`。

# Frozen TCGA analysis plan — 2026-10-18

[English](#english) | [中文](#中文)

## English

**Publication status:** Code is public. Source data are planned for a separate later release; no release date has been specified. Real study inputs and private configurations remain outside this repository, with access conditions managed separately.

This is a bilingual documentation edition of the archived plan and its pre-test amendments, not a newly frozen plan. The initial availability statements and their later resolution are retained in sequence. The prescribed audit outputs are retention requirements and do not imply that patient-level data have been published.

Frozen before expression scores or endpoint associations are computed. Source inputs stay read-only. All analysis outcomes, including null, reverse-direction and unavailable endpoints, remain in the audit tables. Figure eligibility is a presentation filter, not an analysis filter.

### Scope and signatures
- Primary cohort: CRC, combining TCGA-COAD and TCGA-READ at the patient level before scoring and median splitting.
- Exploratory cohorts: every other TCGA project with a local TPM matrix. No pooled pan-cancer analysis. No separate COAD/READ hypothesis tests.
- Frozen signatures from an independent source-marker extraction: c27_CD8_MAIT_SLC4A10, c03_CD4_Tn_NR4A2, c20_CD8_Tem_GZMK (primary states), c08_CD4_Tfh_CXCL13_IL6ST (secondary state). No gene/cutpoint changes after observing endpoints.
- Marker extraction is independent of TCGA outcome data; input GMT and SHA256 will be recorded before scoring.

### Expression and sample selection
- Existing local protein-coding mRNA TPM CSVs, gene symbols. log2(TPM+1), GSVA 2.2.1 ssGSEA, alpha=0.25, normalize=TRUE, minSize=5. No alternative enrichment method selected by smaller P.
- Primary tumor sample type 01 only. LAML blood sample type 03 is ineligible under this primary-solid-tumor scope. Normal, recurrent and metastatic samples excluded.
- One RNA aliquot per patient: non-FFPE if explicitly documented preferred over unknown over FFPE; then complete barcode lexicographic order. No signature or outcome used for duplicate resolution. Available metadata lack a uniform sequencing-quality field, so no inferred quality ranking.
- Duplicate gene symbols, if any, averaged on the linear TPM scale. Missing/nonfinite/negative expression is not silently imputed. All-zero/constant genes removed before scoring. Gene coverage per signature/cohort reported; require >=5 genes and >=50% frozen signature coverage.
- Median computed once within each cohort using all expression-eligible patients. High > median; Low <= median. Endpoint-specific missingness does not change the cutpoint. Ties and group sizes recorded.

### Clinical fields and endpoints
- Clinical source: ${CRC_TCGA_DB_ROOT}/output-clinical, indexed patient records plus XML patient and longitudinal followup tables. RNA-table paper_* historical labels are not used as primary clinical evidence.
- Stage: indexed ajcc_pathologic_stage if a valid I–IV value; XML stage_event_pathologic_stage only if indexed is unavailable. Substages collapsed to I/II/III/IV. Stage I/II vs III/IV, two-sided Fisher exact, OR for High vs Low. No substitution of grade, T category, clinical stage, FIGO or Ann Arbor stage for missing pathologic stage.
- OS: death event from explicit Dead/Deceased vital status with positive days_to_death. Prefer indexed death date; otherwise a unique valid XML patient/followup death date. Conflicting nonindexed death dates without indexed resolution are excluded and logged. An XML death record can update an old indexed Alive record, with provenance flag. For living patients, censor at maximum valid days_to_last_follow_up / days_to_last_followup / days_to_last_known_alive across indexed and XML records. Missing/zero/negative time excluded. Unknown status is not coded alive.
- Age: indexed age_at_diagnosis / 365.25, then XML age_at_initial_pathologic_diagnosis. Sex: indexed gender then XML gender. All source fields, fallback choices, conflicts and exclusions saved per patient.
- No local TCGA-CDR endpoint dataset found. PFI is unavailable and will not be reconstructed from OS or recurrence fragments.

### Tests and fixed eligibility gates
- Stage: n >=40, High and Low each >=15, both early and late stages present. Fisher two-sided P and 95% CI.
- OS: n >=40, High/Low each >=15, deaths >=10 total and >=3 per group. Log-rank primary P; univariable Cox HR and 95% CI. Cox likelihood/Wald P also recorded, but no method switching for significance.
- Sensitivity: age + sex + pathologic stage (categorical) adjusted Cox using complete cases, n>=40, groups>=15, at least 10 events per estimated regression coefficient. If a covariate has no variation or stage is unavailable, the full prespecified model is marked unavailable; a reduced model is not silently substituted. Schoenfeld PH test stored for univariable and adjusted models.
- CRC is its own multiplicity family across all four signatures for each endpoint. Non-CRC cohorts form exploratory families across all four signatures x every eligible cohort, separately for stage Fisher and OS log-rank. BH q-values calculated over all valid tests, not only candidate positives. Adjusted-Cox q-values are a separate sensitivity family.
- Candidate figures require nominal two-sided P<0.05 with stage OR>1 or OS HR>1. Both P and q are shown. Primary c27/CRC relevance takes priority; no selection by the smallest P across unrelated cancers. Non-CRC findings remain explicitly exploratory and cannot be described as CRC clinical validation.

### Runtime and output
- Server snapshot: 96 logical CPUs, ~537 GB available RAM, ~5 TB free disk. 33 TPM files total 1.54 GB. Sequential cohorts; capped internal parallelism (2 workers, BLAS/OMP 1). No installations.
- Deliver all candidate and noncandidate results; per-patient source/sample manifests; score/median/groups; coverage; source hashes; endpoint eligibility; model diagnostics; analysis script/session info; candidate PNG/PDF/SVG plots.
- Official references checked: GSVA manual https://bioconductor.org/packages/release/bioc/manuals/GSVA/man/GSVA.pdf (installed API verified directly); GDC TCGA-CDR endpoint resource https://gdc.cancer.gov/about-data/publications/PanCan-Clinical-2018 .

### Pre-test availability amendment
Before any scores or endpoint tests, an independent reviewer is attempting an official GDC-linked TCGA-CDR download. If the published CDR table is obtained and its schema/provenance validated before testing, add its PFI as a secondary endpoint with the same fixed event/group thresholds, log-rank/Cox tests and separate CRC/exploratory BH families. Its OS is a separately labelled sensitivity endpoint; GDC/XML OS remains primary, irrespective of P values. If download/validation fails, PFI remains unavailable. No endpoint choice is based on observed association results.

### Availability branch resolved before scoring/testing
Official GDC-linked TCGA-CDR SupplementalTableS1 obtained and validated (11,160 unique patients; 57 redacted records excluded from CDR endpoints). Workbook SHA256 ea594c0fbb6731477c7ac511fab449ca9c38b0d42d269591ed9f5c4090e75a5a. GDC/XML OS stays primary; CDR OS is separately labelled sensitivity; CDR PFI is secondary. Published CDR event/time pairs remain unmodified. Each of these endpoints has its own CRC/exploratory BH family. CDR OS sensitivity alone does not nominate a main figure.

Frozen signature GMT SHA256: 32992a8abdefd4a990e1d3d470e2d18e2707bb8d9d34e0456da97c6622097465. Freeze occurred independently at 2026-10-18T08:01:24.488299+00:00. Exact sets: CRC19_c03_CD4_Tn_NR4A2_TOP30; CRC19_c08_CD4_Tfh_CXCL13_IL6ST_TOP30 (secondary); CRC19_c20_CD8_Tem_GZMK_TOP30; CRC19_c27_CD8_MAIT_SLC4A10_TOP30. All 30 genes per set are retained without endpoint-informed editing.

---

## 中文

### 冻结的 TCGA 分析计划——2026-10-18

本节是历史归档计划及其检验前补充条款的中文译文，不代表新制定或重新冻结了分析计划。文中的初始数据可用性状态和随后解决该状态的条款按原顺序保留。

**发布安排：代码已公开；Source data（源数据）计划后续单独发布，日期待定。真实研究输入和私有配置保存在仓库外，其访问条件另行管理。** 本文规定的完整结果与逐患者审计输出属于分析留存要求，不表示这些数据已经公开。

计划在计算表达分数或检验终点关联之前冻结。来源输入保持只读。所有分析结果，包括无显著差异、反向关联和不可分析的终点，均保留在审计表中。图形候选资格只控制展示，不控制哪些结果进入分析。

### 范围与签名

- 主要队列：CRC。在评分和中位数分组之前，按患者层面合并 TCGA-COAD 与 TCGA-READ。
- 探索队列：具有本地 TPM 矩阵的其他所有 TCGA 项目。不合并为泛癌总体分析，也不另做 COAD/READ 单独假设检验。
- 冻结签名来自独立的来源 marker 提取：c27_CD8_MAIT_SLC4A10、c03_CD4_Tn_NR4A2、c20_CD8_Tem_GZMK 为主要状态，c08_CD4_Tfh_CXCL13_IL6ST 为次要状态。查看终点后不改变基因或分组切点。
- marker 提取独立于 TCGA 结局数据；在评分前记录输入 GMT 及其 SHA256。

### 表达数据与样本选择

- 使用现有本地蛋白编码 mRNA TPM CSV，基因标识为基因符号。进行 log2(TPM+1) 转换，使用 GSVA 2.2.1 的 ssGSEA，参数为 alpha=0.25、normalize=TRUE、minSize=5。不根据更小的 P 值选择其他富集方法。
- 仅纳入原发肿瘤样本类型 01。在本次原发实体瘤范围内，LAML 血液样本类型 03 不符合条件。排除正常、复发和转移样本。
- 每位患者保留一份 RNA 分装样本：有明确记录的非 FFPE 优先于状态未知，状态未知优先于 FFPE，然后按完整条形码字典序选择。去重不使用签名或结局。现有元数据没有统一的测序质量字段，因此不推断质量排序。
- 若存在重复基因符号，在线性 TPM 尺度取均值。不静默填补缺失、非有限或负表达值。评分前移除全零或恒定基因。报告每个签名/队列的基因覆盖情况，要求至少 5 个基因且覆盖冻结签名的至少 50%。
- 使用各队列全部表达合格患者计算一次中位数：High > median，Low <= median。终点缺失不会改变切点；记录与中位数相等的情况及组别人数。

### 临床字段与终点

- 临床来源：`${CRC_TCGA_DB_ROOT}/output-clinical` 中的 indexed 患者记录、XML 患者表和纵向随访表。RNA 表中的历史 `paper_*` 标签不作为主要临床证据。
- 分期：优先使用 indexed `ajcc_pathologic_stage` 中有效的 I–IV 值；仅在其不可用时使用 XML `stage_event_pathologic_stage`。各亚分期归并为 I/II/III/IV。比较 I/II 与 III/IV，进行双侧 Fisher 精确检验，OR 表示 High 相对于 Low 的晚期分期优势比。不以组织学分级、T 分类、临床分期、FIGO 或 Ann Arbor 分期代替缺失的病理分期。
- OS：死亡事件必须具有明确的 Dead/Deceased 生存状态及正值 `days_to_death`。优先使用 indexed 死亡时间；否则使用 XML 患者/随访记录中唯一有效的死亡时间。若非 indexed 死亡时间相互冲突且无法由 indexed 记录解决，则排除并记录。XML 死亡记录可更新较旧的 indexed Alive 记录，同时标记来源。存活患者的删失时间取 indexed 和 XML 记录中 `days_to_last_follow_up`、`days_to_last_followup`、`days_to_last_known_alive` 的最大有效值。排除缺失、零或负时间；未知状态不编码为存活。
- 年龄：优先使用 indexed `age_at_diagnosis / 365.25`，其次使用 XML `age_at_initial_pathologic_diagnosis`。性别：优先 indexed `gender`，其次 XML `gender`。逐患者保存全部来源字段、备用字段选择、冲突和排除记录。
- 初始计划记录：当时未找到本地 TCGA-CDR 终点数据集，因此 PFI 不可用，也不会由 OS 或零散复发字段重建。随后在检验前解决的数据可用性分支见下文。

### 检验与固定纳入门槛

- 分期：n >=40，High 和 Low 各 >=15，且早期、晚期均有患者。使用 Fisher 双侧 P 值及 95% CI。
- OS：n >=40，High/Low 各 >=15，总死亡事件 >=10，且每组 >=3。主要 P 值来自 log-rank 检验；报告单变量 Cox HR 和 95% CI。同时记录 Cox 似然/Wald P 值，但不为获得显著性切换检验方法。
- 敏感性分析：对完整病例使用年龄 + 性别 + 病理分期（分类变量）调整的 Cox 模型，要求 n >=40、每组 >=15，且每个待估计回归系数至少有 10 个事件。若协变量无变异或病理分期不可用，则将完整预设模型标记为不可用，不静默替换为简化模型。保存单变量及调整模型的 Schoenfeld 比例风险（PH）检验。
- 每个终点中，CRC 的四套签名构成独立的多重检验家族。非 CRC 队列中，四套签名 × 全部合格队列构成探索性家族，分期 Fisher 和 OS log-rank 分别校正。BH q 值基于全部有效检验计算，不只包含阳性候选。调整 Cox 的 q 值属于单独的敏感性分析家族。
- 候选图要求名义双侧 P<0.05，且分期 OR>1 或 OS HR>1，同时展示 P 和 q。优先考虑主要 c27/CRC 问题的相关性，不按不同癌种中最小的 P 值选择结果。非 CRC 结果明确属于探索性结果，不能描述为 CRC 临床验证。

### 运行环境与输出

- 服务器记录快照：96 个逻辑 CPU、约 537 GB 可用内存、约 5 TB 可用磁盘空间；33 个 TPM 文件共 1.54 GB。队列顺序处理，内部并行受限为 2 个 worker、BLAS/OMP 线程数为 1。不安装软件。
- 留存全部候选和非候选结果、逐患者来源/样本清单、分数/中位数/分组、基因覆盖情况、来源哈希、终点纳入资格、模型诊断、分析脚本/会话信息，以及候选 PNG/PDF/SVG 图。这些输出的留存与公开发布是不同事项。
- 已核查官方参考资料：[GSVA 手册](https://bioconductor.org/packages/release/bioc/manuals/GSVA/man/GSVA.pdf)（直接验证已安装版本 API）及 [GDC TCGA-CDR 终点资源](https://gdc.cancer.gov/about-data/publications/PanCan-Clinical-2018)。

### 检验前的数据可用性补充条款

在任何评分或终点检验之前，由独立审阅者尝试下载 GDC 官方链接提供的 TCGA-CDR。若在检验前获得已发表 CDR 表，且核验其字段结构和来源，则将 PFI 加为次要终点，使用相同的固定事件/分组门槛、log-rank/Cox 检验，以及独立的 CRC/探索性 BH 家族。其 OS 单独标记为敏感性终点；无论 P 值如何，GDC/XML OS 始终为主要终点。若下载或核验失败，PFI 继续标记为不可用。终点选择不依据观察到的关联结果。

### 在评分和检验前解决的数据可用性分支

已取得并核验 GDC 官方链接的 TCGA-CDR SupplementalTableS1，包含 11,160 位唯一患者；57 条撤回记录从 CDR 终点分析中排除。工作簿 SHA256 为 `ea594c0fbb6731477c7ac511fab449ca9c38b0d42d269591ed9f5c4090e75a5a`。GDC/XML OS 仍为主要终点，CDR OS 单独标记为敏感性终点，CDR PFI 为次要终点。保留已发表的事件/时间配对，不作修改。每个终点均有各自的 CRC/探索性 BH 家族。仅 CDR OS 敏感性分析阳性不足以提名主图。

冻结签名 GMT 的 SHA256 为 `32992a8abdefd4a990e1d3d470e2d18e2707bb8d9d34e0456da97c6622097465`。独立冻结时间为 `2026-10-18T08:01:24.488299+00:00`，集合名称精确为：`CRC19_c03_CD4_Tn_NR4A2_TOP30`、`CRC19_c08_CD4_Tfh_CXCL13_IL6ST_TOP30`（次要）、`CRC19_c20_CD8_Tem_GZMK_TOP30` 和 `CRC19_c27_CD8_MAIT_SLC4A10_TOP30`。每套 30 个基因全部保留，不依据终点结果编辑。

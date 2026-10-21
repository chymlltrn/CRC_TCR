# Public-release identifier review / 公开发布前标识符审查

This review covers the bulk module's identifier adaptations for public code publication. Source data are not yet released. Repository visibility and final repository-wide checksums are handled separately; this document does not assert that publication has already occurred.

## Changes and configuration

- In [v6_candidate_audit.py](source_tree/20261003_paper_v2/revision_v18_figure_story_revision/reproducibility/historical_analysis/revision_v6/v6_candidate_audit.py), two subject-derived local variable names were replaced with `selected_subject` and `selected_subject_key` at seven AST name occurrences. Selection criteria, the selected row/key, exclusion scopes and downstream calculations remain the same under this renaming.
- In [nodal_metrics_v11.py](source_tree/20261003_paper_v2/revision_v18_figure_story_revision/reproducibility/historical_analysis/revision_v11/nodal_metrics_v11.py), the subject-derived external-module attribute is now accessed as `getattr(M, _private_literal('private_5b340947280f76b5'))(f)`. The exact attribute name is supplied by external private configuration. No substitute function or new exclusion rule was introduced.
- [config_fields.json](config_fields.json) records the new field and use site. [config.example.json](config.example.json) continues to use an external `private_literals_file`; it does not embed the real value or an inline override. The private-literal field count is now 165; the external-path field count remains 47.
- [source_manifest.json](source_manifest.json) contains the updated hashes for the two adapted scripts. Archived originals remain unchanged. The public source is adapted code, not a byte-for-byte copy of those originals.

## Verification and limits

The 2026-10-21 checks recorded in [validation_report.json](validation_report.json) passed:

1. Both changed scripts compile. Restoring the renamed identifiers and configured attribute reconstructs the ASTs of their pre-fix release snapshots.
2. All 118 released Python source scripts pass normalized AST comparison against archived originals, with external-literal restoration, the documented root repair, local-variable alpha normalization and documentation exclusions. The retained shell script passes `bash -n`.
3. Three synthetic dynamic-call cases—empty, all-false and mixed masks—give identical return values, argument identity and call counts for direct and configured attribute access. Missing attributes raise `AttributeError` through both paths. The configured name is loaded through the actual external-configuration runtime.
4. All six configuration-boundary tests pass. A case-insensitive scan of the bulk publication directory finds no remaining occurrence of the reviewed subject identifier.

The private review directory retains the exact restoration rules, original identifier values, pre-fix source snapshots and detailed test receipts; they are not included in this public code module. These checks cover the stated adaptations. They do not constitute a universal de-identification guarantee, a fresh scientific rerun or validation of every external environment. No scientific pipelines were imported and no source data were used in the focused synthetic tests. Final package checksums are refreshed by the release coordinator after all module changes.

---

# 中文说明：公开发布前标识符审查

本次审查针对 bulk 模块公开代码前的标识符适配。Source data 尚未发布。仓库可见性及最终仓库级校验和另行处理；本文不表示代码已经完成公开。

## 改动与配置

- 在 [v6_candidate_audit.py](source_tree/20261003_paper_v2/revision_v18_figure_story_revision/reproducibility/historical_analysis/revision_v6/v6_candidate_audit.py) 中，两个包含个体标识的局部变量已改为 `selected_subject` 和 `selected_subject_key`，共修改七处 AST 名称节点。该重命名保留原有选择条件、选中行/键、排除范围及后续计算。
- 在 [nodal_metrics_v11.py](source_tree/20261003_paper_v2/revision_v18_figure_story_revision/reproducibility/historical_analysis/revision_v11/nodal_metrics_v11.py) 中，包含个体标识的外部模块属性改为通过 `getattr(M, _private_literal('private_5b340947280f76b5'))(f)` 访问。准确属性名由外部私有配置提供，没有替换原函数或增加新的排除规则。
- [config_fields.json](config_fields.json) 记录新增字段及使用位置。[config.example.json](config.example.json) 继续使用外部 `private_literals_file`，不内嵌真实值或会覆盖外部值的内联配置。私有字符串字段现为 165 个；外部路径字段仍为 47 个。
- [source_manifest.json](source_manifest.json) 已更新两份适配脚本的哈希。归档原件未改动；公开源码经过适配，并非原件的逐字节副本。

## 验证与边界

[validation_report.json](validation_report.json) 记录的 2026-10-21 检查均通过：

1. 两份修改脚本均可通过语法编译。恢复重命名的标识符及配置属性后，其 AST 与修复前发布快照一致。
2. 全部 118 份发布 Python 源脚本在恢复外部字符串、应用已记录的根目录修复与局部变量 alpha 规范化、排除文档改动后，规范化 AST 与归档原件一致。保留的 shell 脚本通过 `bash -n`。
3. 三组合成动态调用用例（空掩码、全假掩码及混合掩码）中，直接属性调用与配置驱动调用的返回值、参数对象身份和调用次数一致。属性不存在时，两种路径均抛出 `AttributeError`。测试通过实际的外部配置运行时加载属性名。
4. 六项配置边界测试全部通过。对 bulk 发布目录进行不区分大小写的扫描后，未发现本次审查的个体标识残留。

准确还原规则、原始标识符值、修复前源码快照及详细测试回执保留在私有审查目录中，不包含在公开代码模块内。这些检查仅覆盖上述适配，不构成普遍脱敏保证，不代表重新运行了科学分析，也未验证所有外部环境。专项合成测试未导入科学分析流程或使用源数据。全部模块改动完成后，由发布协调者刷新最终包校验和。

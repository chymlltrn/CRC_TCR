# Dependencies by stage

This document accompanies the public code release. Source data have not yet been released; external inputs, model assets and private provenance archives are not bundled with the code.

本文随公开代码一同发布。Source data 尚未发布；外部输入、模型资源和私有溯源归档不随代码提供。

English documentation is followed by the complete Chinese translation. 英文说明之后提供完整中文说明。

Use the original stage-specific environments/receipts. These modules were not validated against one newly combined environment, and installing every dependency into one environment is not a supported reproduction claim.

- General tables/statistics: numpy, pandas, scipy, pyarrow, statsmodels, scikit-learn, joblib, threadpoolctl, openpyxl and psutil.
- Bulk preprocessing: Java and MiXCR 4.7.0; preserve the original human RNA-seq YAMLs and clone-export semantics.
- TCR-BERT: PyTorch, transformers, local tokenizer/model config and weights. The frozen bank manifest records torch 2.5.1+cu121 / transformers 4.46.3 and model revision. Mean pooling excludes special tokens; respect the encoded bank's exact configuration.
- SCEPTR: SCEPTR 1.2.0, libtcrlm 1.1.3, torch 2.5.1+cu121, official CDR3-only model files. The two embedding environments recorded different numpy versions; consult the bank manifest.
- Geometry: faiss, igraph (recorded 0.10.8), umap-learn, pynndescent, annoy and numba where used.
- GLIPH/TCRdist: original GLIPH2 `irtools` binary and reference files, tcrdist3, pwseqdist and OLGA; model/reference artifacts remain external.
- MIL/representation comparison: PyTorch, scikit-learn, optional TAPE/tcr2vec for the recorded comparison arms, and corresponding checkpoints.
- Prediction/SHAP: scikit-learn, xgboost, joblib and SHAP; preserve stored fold selection and feature order.
- Antigen/sequence: Biopython, tcrdist3, torch and the recorded ERGO environment/checkpoints. Predictions are exploratory reference evidence.
- Graphics/assembly: matplotlib, Pillow, PyMuPDF (`fitz`), lxml, pypdf and fonts used by the original renderer. Some optional historical plotting paths use omicverse.
- Release tests: Python standard library only; source syntax and release-boundary tests do not import scientific pipelines.

The active atlas's `embeddings/bank_manifest.json`, projection receipts and model manifests are the authoritative version/provenance records. Third-party code, whole environments, trained weights and input data are not vendored here.


---

# 中文说明：各阶段依赖

请使用原始的分阶段环境及运行回执。这些模块尚未在一个新合并的环境中统一验证；将所有依赖安装到同一环境中，不代表本项目支持以该环境重现全部分析。

- 通用表格与统计：numpy、pandas、scipy、pyarrow、statsmodels、scikit-learn、joblib、threadpoolctl、openpyxl 和 psutil。
- Bulk 预处理：Java 和 MiXCR 4.7.0；须保留原始 human RNA-seq YAML 配置和克隆导出的字段含义及处理规则。
- TCR-BERT：PyTorch、transformers、本地 tokenizer/模型配置和权重。冻结的嵌入库清单记录了 torch 2.5.1+cu121 / transformers 4.46.3 及模型修订版本。均值池化排除特殊 token；须遵循生成该嵌入库时的精确配置。
- SCEPTR：SCEPTR 1.2.0、libtcrlm 1.1.3、torch 2.5.1+cu121，以及官方 CDR3-only 模型文件。两个嵌入环境记录的 numpy 版本不同，具体以嵌入库清单为准。
- 几何分析：faiss、igraph（记录版本为 0.10.8）、umap-learn、pynndescent、annoy，以及相应模块使用的 numba。
- GLIPH/TCRdist：原始 GLIPH2 `irtools` 二进制程序与参考文件、tcrdist3、pwseqdist 和 OLGA；模型/参考资源仍作为外部输入提供。
- MIL/表征比较：PyTorch、scikit-learn、已记录比较分支可选使用的 TAPE/tcr2vec，以及对应检查点。
- 预测/SHAP：scikit-learn、xgboost、joblib 和 SHAP；须保留已存储的折选择与特征顺序。
- 抗原/序列：Biopython、tcrdist3、torch，以及已记录的 ERGO 环境/检查点。预测仅作为探索性的参考证据。
- 绘图/组装：matplotlib、Pillow、PyMuPDF（`fitz`）、lxml、pypdf，以及原始绘图程序使用的字体。部分可选历史绘图流程使用 omicverse。
- 发布验证测试：仅需 Python 标准库；源码语法检查和发布边界测试不会导入科学分析流程。

当前图谱的 `embeddings/bank_manifest.json`、投影回执及模型清单是版本和溯源信息的权威记录。本仓库未内置第三方代码、完整环境、训练后权重或输入数据。

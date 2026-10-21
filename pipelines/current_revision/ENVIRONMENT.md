# Runtime provenance

[English](#english) | [中文](#中文)

## English

**Publication status:** Code is public. Source data are planned for a separate later release; no release date has been specified. Real study inputs and private configurations remain outside this repository, with access conditions managed separately.

The environment availability statements below describe the recorded packaging verification snapshot, rather than a new inspection of the reader's current runtime.

The original analysis and the packaging verification use different recorded environments. These are evidence snapshots, not an automatically installed lock.

| Component | Original analysis evidence | Packaging check |
|---|---|---|
| TCGA R | 4.5.1, Ubuntu 22.04.5 | 4.6.0, macOS |
| GSVA | 2.2.1 | 2.6.0 synthetic API check; historical scores reused for statistics |
| BiocParallel | 1.44.0 | 1.46.0 |
| survival | 3.8-6 | 3.8-6 |
| data.table | 1.18.4 | 1.18.2.1 |
| ggplot2 | 4.0.3 | 4.0.3 |
| matrixStats | 1.5.0 | 1.5.0 |
| edgeR | 4.6.2 in original marker fit | 4.10.0 available, full fit not rerun |
| qPCR Python/packages | No historical version lock recorded | Python 3.9.6; NumPy 2.0.2; pandas 2.3.3; SciPy 1.13.1; Matplotlib 3.9.4; PyMuPDF 1.26.5 |
| TCGA preparation | Python standard library | Python 3.9.6 |
| CDR extraction | Official workbook hash frozen | openpyxl reads original workbook |
| C++ stream | C++17 plus zlib source | Compiler + exact synthetic count/detection fixture |

Server Python currently lacks PyMuPDF, so its default environment cannot reproduce qPCR figures without an appropriate existing runtime. The local R installation lacks working X11/Cairo libraries; vector PDF export was tested there, and PNG/SVG requires the original server Cairo-capable runtime. Code rejects requested PNG/SVG when Cairo is unavailable rather than silently omitting files. No package installation is performed.

Original reproducibility inputs and private configurations belong outside GitHub. Use the same original R environment for strict full-score reconstruction; cached-score statistical equivalence is reported separately.

---

## 中文

### 运行环境来源

原始分析与代码打包核验使用不同的已记录环境。这些记录是环境证据快照，不是自动安装的环境锁文件。以下“当前”运行环境描述沿用打包核验时的记录，并非对阅读本文时机器状态的重新检测。

**发布安排：代码已公开；Source data（源数据）计划后续单独发布，日期待定。真实研究输入和私有配置保存在仓库外，其访问条件另行管理。**

| 组件 | 原始分析证据 | 代码打包核验 |
|---|---|---|
| TCGA R | 4.5.1，Ubuntu 22.04.5 | 4.6.0，macOS |
| GSVA | 2.2.1 | 2.6.0：检查合成数据 API；统计核验复用历史分数 |
| BiocParallel | 1.44.0 | 1.46.0 |
| survival | 3.8-6 | 3.8-6 |
| data.table | 1.18.4 | 1.18.2.1 |
| ggplot2 | 4.0.3 | 4.0.3 |
| matrixStats | 1.5.0 | 1.5.0 |
| edgeR | 原始 marker 拟合为 4.6.2 | 可用版本为 4.10.0；未重跑完整拟合 |
| qPCR Python/软件包 | 未记录历史版本锁 | Python 3.9.6；NumPy 2.0.2；pandas 2.3.3；SciPy 1.13.1；Matplotlib 3.9.4；PyMuPDF 1.26.5 |
| TCGA 数据准备 | Python 标准库 | Python 3.9.6 |
| CDR 提取 | 已冻结官方工作簿哈希 | 使用 openpyxl 读取原始工作簿 |
| C++ 流式处理 | C++17 与 zlib 源代码 | 编译器及精确的合成计数/检出测试输入 |

打包核验时，服务器默认 Python 缺少 PyMuPDF，因此需要适当的现有运行环境才能复现 qPCR 图。本地 R 缺少可正常加载的 X11/Cairo 库；已在该环境测试矢量 PDF 导出，PNG/SVG 则需要原服务器支持 Cairo 的环境。请求 PNG/SVG 而 Cairo 不可用时，代码会明确报错，不会静默遗漏文件。脚本不会安装软件包。

原始复现输入和私有配置应保留在 GitHub 仓库之外。严格重建全部分数应使用原始 R 环境；基于缓存分数的统计等价性核验单独报告。

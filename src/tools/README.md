# Tools 模块

本目录为 LangG 系统提供 LangChain 标准工具集，封装为 `BaseTool` 格式，可无缝集成到 LangGraph 工作流中供 LLM 调用。按功能分为九大类。

## 文件结构

```
src/tools/
├── __init__.py               # 工具注册表（list_tools / list_tools_with_web_search / list_all_tools）
├── basic_tools.py            # 基础工具（echo、word_count）
├── clinical_tools.py         # 临床工具（病史解析/病理/CT/MRI/分子标记）
├── rag_tools.py              # RAG 检索工具（7种：通用/治疗/分期/药物/来源/混合/TOC/章节读取）
├── web_search_tools.py       # 网络搜索工具（通用/临床证据/药物/指南更新/最新研究）
├── database_tools.py         # 数据库工具（病例查询/影像/统计/搜索/随机病例）
├── tumor_screening_tools.py  # YOLOv8 肿瘤检测工具
├── tumor_localization_tools.py # U-Net 肿瘤分割工具
├── radiomics_tools.py        # 影像组学工具（U-Net + PyRadiomics + LASSO）
├── pathology_clam_tools.py   # CLAM 病理切片分类工具
├── card_formatter.py         # 卡片格式化工具
└── tool/                     # 第三方 AI 模型文件
    ├── Tumor_Detection/      # YOLOv8 模型
    ├── Tumor_Localization/   # U-Net 模型
    └── Pathological_Slide_Classification/CLAM_Tool/  # CLAM 病理分类（20+ 文件）
```

## 工具分类

### 1. 基础工具（basic_tools.py）

- `echo(text)` — 原样返回输入，测试工具链路
- `word_count(text)` — 统计单词数

### 2. 临床工具（clinical_tools.py）

混合解析策略（规则 + LLM 增强），从非结构化医疗文本中提取结构化信息。

| 工具 | 功能 |
|------|------|
| `PatientHistoryParserTool` | 解析患者病史（主诉/肿瘤位置/症状/家族史/风险因素），文本<100字符规则解析，复杂文本LLM增强 |
| `PathologyParserTool` | 解析病理报告（组织学/分化程度/分子标记MSI/MMR/病理分期pT/pN），中英文混合识别 |
| `VolumeCTSegmentorTool` | CT 报告 M 分期评估（肝/肺/腹膜转移灶检测 + 淋巴结评估） |
| `RectalMRStagerTool` | 直肠 MRI 局部分期（T/N/MRF/EMVI/CRM + 新辅助治疗推荐） |
| `MolecularGuidelineTool` | 分子标记物→治疗建议（KRAS/NRAS/BRAF） |
| `PolypDetectionTool` | 文本息肉检测（mock） |

### 3. RAG 检索工具（rag_tools.py）

继承 `BaseTool`，使用 Pydantic 输入模式，双通道输出（文本引用锚点 `[[Source:File|Page:N]]` + `<retrieved_evidence>` JSON）。

| 工具 | 用途 |
|------|------|
| `ClinicalGuidelineSearchTool` | 通用指南搜索（混合检索 + 重排序） |
| `TreatmentSearchTool` | 治疗方案专项搜索（化疗/靶向/免疫/新辅助/辅助） |
| `StagingSearchTool` | TNM 分期标准搜索 |
| `DrugInfoSearchTool` | 药物信息搜索（用法/禁忌/不良反应/相互作用） |
| `GuidelineSourceSearchTool` | 按来源过滤（NCCN/CSCO/ESMO） |
| `HybridSearchTool` | 高级混合检索（多维度元数据过滤） |
| `GuidelineStructureTool` | 获取指南目录结构 |
| `GuidelineReaderTool` | 读取指南完整章节 |

工厂函数：`get_guideline_tool()`、`get_all_rag_tools()`、`get_enhanced_rag_tools()`

### 4. 网络搜索工具（web_search_tools.py）

基于 `WebSearchService`，实时获取最新医学资料，带结果验证。

| 工具 | 用途 |
|------|------|
| `WebSearchTool` | 通用网络搜索 |
| `ClinicalEvidenceSearchTool` | 临床证据搜索（优先 NCCN/ESMO/CSCO/PubMed/RCT/Meta分析） |
| `DrugInfoSearchTool` | 处方药信息搜索（说明书/FDA/NMPA），支持 dosage/interaction/adverse/indication |
| `GuidelineUpdateSearchTool` | 指南更新追踪 |
| `LatestResearchSearchTool` | 最新研究文献（优先 PubMed/Cochrane/高影响因子期刊） |

工厂函数：`get_web_search_tool()`、`get_all_web_search_tools()`、`get_clinical_web_search_tools()`

### 5. 数据库工具（database_tools.py）

封装 `VirtualCaseDatabase`，LangChain `@tool` 格式。

| 工具 | 功能 |
|------|------|
| `get_patient_case_info` | 按患者ID查询完整病历 |
| `get_patient_imaging` | 获取患者影像资料路径 |
| `get_patient_pathology_slides` | 获取患者病理切片缩略图 |
| `get_database_statistics` | 数据库整体统计 |
| `search_cases` | 多条件组合搜索（部位/分期/组织学/MMR/年龄/CEA） |
| `list_imaging_folders` | 列出所有影像文件夹 |
| `get_random_case` | 随机获取病例（支持筛选条件） |
| `summarize_patient_existing_info` | 患者已有数据摘要 |
| `upsert_patient_info` | 写入/更新病例记录 |

### 6. 肿瘤检测工具（tumor_screening_tools.py）

基于 YOLOv8，延迟导入（torch/ultralytics 首次调用时加载），模型单例懒加载。

| 工具 | 功能 |
|------|------|
| `tumor_screening_tool` | 批量肿瘤筛选（YOLO 检测 → 阳性图像复制到输出目录） |
| `quick_tumor_check` | 单张图像快速检测（has_tumor + confidence + bounding_boxes） |
| `get_tumor_screening_status` | 工具状态查询（模型/依赖/GPU） |
| `perform_comprehensive_tumor_check` | 按患者ID自动查找影像文件夹并执行完整检测（StructuredTool） |

### 7. 肿瘤定位工具（tumor_localization_tools.py）

基于 U-Net，像素级肿瘤分割。

| 工具 | 功能 |
|------|------|
| `tumor_localization_tool` | 单张 CT 分割（mask + 可视化 + 面积/边界框） |
| `batch_tumor_localization` | 目录批量分割（保持文件夹结构 + 汇总报告） |
| `get_localization_status` | 工具状态查询 |

### 8. 影像组学工具（radiomics_tools.py）

完整影像分析工具链，模型单例懒加载。

| 工具 | 功能 |
|------|------|
| `unet_segmentation_tool` | U-Net 分割（mask + 肿瘤面积/占比/边界框） |
| `radiomics_feature_extraction_tool` | PyRadiomics 特征提取（~1500 维：shape/firstorder/glcm/glrlm/glszm/gldm/ngtdm） |
| `lasso_feature_selection_tool` | LASSO 特征筛选（Top-K 重要特征，单个样本基于方差筛选） |
| `comprehensive_radiomics_analysis` | 一键完整分析：YOLO → U-Net → PyRadiomics → LASSO |

### 9. 病理 CLAM 工具（pathology_clam_tools.py）

基于 CLAM 深度学习模型，全切片图像（WSI）分析，支持 .svs/.tif/.ndpi 格式。

| 工具 | 功能 |
|------|------|
| `pathology_slide_classify` | 完整分析：组织分割 → 切片提取 → 特征提取 → CLAM 推理 → 热力图 + Top-K 切片 |
| `quick_pathology_check` | 快速分类（仅 TUMOR/NORMAL，无热力图） |
| `get_pathology_clam_status` | 状态查询（模型/依赖/GPU/支持格式） |
| `perform_comprehensive_pathology_analysis` | 按患者ID自动查找所有切片并批量分析（StructuredTool） |

子进程执行 CLAM 管线：`create_patches_fp.py`（切片）→ `extract_features_fp.py`（特征提取）→ `create_heatmaps.py`（推理+可视化），动态生成 YAML 配置。

### 10. 卡片格式化工具（card_formatter.py）

纯展示逻辑，将原始数据转换为前端可渲染的卡片格式。

- `CardFormatter`（模块单例 `formatter`）
  - `format_patient_card()` — 患者信息卡
  - `format_imaging_card()` — 影像卡（含 base64 预览图嵌入）
  - `format_pathology_slide_card()` — 病理切片卡
  - `format_tumor_screening_result()` — 肿瘤检测结果卡
  - `format_comprehensive_tumor_detection()` — 综合检测卡
  - `format_radiomics_report_card()` — 影像组学报告卡
- `_embed_preview_images()` — 读取并 base64 编码图片（限制8张、2MB/张）

## 工具注册表

```python
list_tools()                   # 临床 + RAG + 肿瘤检测 + 定位 + 影像组学 + 病理CLAM
list_tools_with_web_search()   # 上述 + 网络搜索
list_all_tools()               # 上述 + 数据库 + 卡片格式化 + 全部变体
get_all_rag_tools()            # 全部 RAG 工具
get_enhanced_rag_tools()       # 推荐 RAG 子集
get_clinical_web_search_tools() # 临床网络搜索子集
get_database_tools()           # 全部数据库工具
```

## 依赖

| 包名 | 用途 | 触发条件 |
|------|------|----------|
| langchain / langchain-core | 工具框架基础 | 核心 |
| pydantic | 输入模式定义 | 核心 |
| torch | 深度学习框架 | 肿瘤定位/影像组学/病理CLAM |
| ultralytics | YOLO 目标检测 | 肿瘤筛选 |
| pyradiomics / SimpleITK | 影像组学特征提取 | 影像组学 |
| scikit-learn | LASSO 特征筛选 | 影像组学 |
| opencv-python | 图像处理 | 肿瘤筛选/定位/影像组学 |
| openslide-python | WSI 读取 | 病理CLAM |
| h5py / timm / pyyaml | CLAM 依赖 | 病理CLAM |

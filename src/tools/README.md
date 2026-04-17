# Tools 模块技术文档

## 概述

Tools 模块是 LangG 项目的核心工具层封装，为临床决策支持系统提供丰富的功能工具集。该模块按照功能职责分为九大类：基础工具、临床工具、RAG 检索工具、联网搜索工具、数据库查询工具、肿瘤筛选工具（YOLO）、肿瘤定位工具（U-Net）、影像组学工具和病理 CLAM 工具。每类工具都封装为 LangChain 标准的 BaseTool 格式，可以无缝集成到 LangGraph 工作流中供 LLM 调用。

工具模块的设计遵循专业化分工原则：临床工具专注于医学文档解析和分期评估，能够从患者病史、病理报告、影像学报告等非结构化文本中提取结构化临床信息；RAG 工具封装了混合检索能力，支持按内容类型、疾病类型、指南来源等多维度检索临床指南；联网搜索工具提供了实时获取最新医学资料的能力；数据库工具提供虚拟病例数据的查询和检索能力；肿瘤筛选工具集成 YOLOv8 模型，提供 CT 图像的肿瘤检测功能；肿瘤定位工具集成 U-Net 模型，提供像素级肿瘤分割和定位功能；影像组学工具提供完整的影像分析工具链（U-Net 分割 + PyRadiomics 特征提取 + LASSO 特征筛选）；病理 CLAM 工具基于 CLAM 深度学习模型，提供病理全切片图像的自动分类、注意力热力图生成和高关注区域提取功能。整体架构支持工具的灵活组合和扩展，便于根据业务需求新增特定领域的专业工具。

---

## 目录结构

```
src/tools/
├── __init__.py              # 模块入口，导出公共接口
├── basic_tools.py           # 基础工具（echo、word_count）
├── clinical_tools.py        # 临床工具集
│   ├── PatientHistoryParserTool    # 患者病史解析
│   ├── PolypDetectionTool          # 息肉检测
│   ├── PathologyParserTool         # 病理报告解析
│   ├── VolumeCTSegmentorTool       # CT分期评估
│   ├── RectalMRStagerTool          # 直肠MRI分期
│   └── MolecularGuidelineTool      # 分子标志物建议
├── rag_tools.py             # RAG检索工具集
│   ├── ClinicalGuidelineSearchTool     # 基础临床指南搜索
│   ├── TreatmentSearchTool              # 治疗方案搜索
│   ├── StagingSearchTool                # 分期标准搜索
│   ├── DrugInfoSearchTool               # 药物信息搜索
│   ├── GuidelineSourceSearchTool        # 指南来源搜索
│   ├── HybridSearchTool                 # 混合检索
│   ├── ListGuidelineTOCTool             # 列出指南目录
│   └── ReadGuidelineChapterTool         # 读取指南章节
├── web_search_tools.py      # 联网搜索工具集
│   ├── WebSearchTool              # 通用联网搜索
│   ├── ClinicalEvidenceSearchTool # 临床证据搜索
│   ├── DrugInfoSearchTool         # 药物信息搜索
│   ├── GuidelineUpdateSearchTool  # 指南更新搜索
│   └── LatestResearchSearchTool   # 最新研究搜索
├── database_tools.py        # 数据库查询工具集
│   ├── get_patient_case_info         # 查询患者病历
│   ├── get_patient_imaging           # 获取患者影像
│   ├── get_database_statistics      # 数据库统计
│   ├── search_cases                  # 搜索病例
│   ├── list_imaging_folders          # 列出影像文件夹
│   ├── get_random_case               # 随机病例
│   └── perform_comprehensive_tumor_check # 完整肿瘤检测
├── tumor_screening_tools.py  # 肿瘤筛选工具集（基于YOLO）
│   ├── tumor_screening_tool          # 批量肿瘤筛选
│   ├── quick_tumor_check             # 快速肿瘤检测
│   ├── get_tumor_screening_status    # 工具状态查询
│   └── perform_comprehensive_tumor_check # 综合肿瘤检测
├── tumor_localization_tools.py # 肿瘤定位工具集（基于U-Net）
│   ├── tumor_localization_tool      # 肿瘤定位
│   ├── batch_tumor_localization      # 批量肿瘤定位
│   └── get_localization_status       # 工具状态查询
├── radiomics_tools.py        # 影像组学工具集
│   ├── unet_segmentation_tool               # U-Net肿瘤分割
│   ├── radiomics_feature_extraction_tool    # PyRadiomics特征提取
│   ├── lasso_feature_selection_tool        # LASSO特征筛选
│   └── comprehensive_radiomics_analysis     # 完整影像组学分析
├── pathology_clam_tools.py   # 病理 CLAM 工具集（全切片分析）
│   ├── pathology_slide_classify             # 病理切片分类
│   ├── quick_pathology_check                # 快速病理检查
│   ├── get_pathology_clam_status           # 工具状态查询
│   └── perform_comprehensive_pathology_analysis # 综合病理分析
├── card_formatter.py         # 卡片格式化工具
│   ├── CardFormatter                  # 卡片格式化类
│   ├── format_patient_card           # 患者卡片格式化
│   ├── format_imaging_card            # 影像卡片格式化
│   ├── format_tumor_screening_result # 肿瘤筛选结果格式化
│   └── format_comprehensive_tumor_detection # 完整肿瘤检测格式化
└── tool/                     # 模型文件目录
    ├── Tumor_Detection/              # YOLO模型
    │   └── best.pt
    ├── Tumor_Localization/           # U-Net模型
    │   └── checkpoint_epoch_last.pth
    └── Pathological_Slide_Classification/ # 病理切片分类
        └── CLAM_Tool/                # CLAM 全切片分析工具
            ├── s_4_checkpoint.pt     # CLAM 模型权重
            ├── models/               # 模型定义
            ├── wsi_core/             # WSI 处理核心
            └── ...
```

---

## 核心组件

### 1. 基础工具（basic_tools.py）

基础工具模块提供了两个最简单的工具实现，作为工具系统的示范和测试用途。这两个工具不涉及医学领域知识，主要用于验证工具调用链路的正确性。

#### Echo 工具

```python
@tool
def echo(text: str) -> str:
    """Echo the provided text."""
    return text
```

Echo 工具接收任意文本输入并原样返回，是最简单的工具原型。该工具可用于测试 LLM 是否能够正确调用工具并接收返回值，也可用于简单的文本传递场景。

#### Word Count 工具

```python
@tool
def word_count(text: str) -> int:
    """Count words in the provided text."""
    return len(text.split())
```

Word Count 工具计算输入文本中的单词数量，以空格分隔统计。该工具展示了工具如何接收输入、执行计算并返回结果的过程，可作为复杂工具开发的参考模板。

---

### 2. 临床工具（clinical_tools.py）

临床工具模块是系统的核心功能组件，专门用于解析各类医学文档并提取结构化临床信息。这些工具直接面向医学诊断场景，能够将非结构化的医疗文本转换为标准化的数据格式，为后续的分期评估和治疗决策提供数据支撑。

#### 患者病史解析工具（PatientHistoryParserTool）

患者病史解析工具将自由文本形式的患者病史转换为结构化数据模型，是患者信息采集的第一个环节。该工具采用混合解析策略：对于简单明确的病史使用基于规则的快速解析，对于复杂模糊的病史调用 LLM 进行语义理解增强解析。这种设计在保证处理速度的同时确保了复杂病例的解析准确性。

```python
@tool
def PatientHistoryParserTool(history_text: str) -> Dict:
    """
    基于大模型增强的患者病史解析工具（V2版本）
    
    输出字段：
    - chief_complaint: 患者主诉
    - tumor_location: 肿瘤位置 (colon/rectum/multiple/unknown)
    - tumor_location_confidence: 位置判断可信度 (0-1)
    - symptoms: 症状列表
    - symptom_duration: 症状持续时间
    - family_history: 是否有家族史
    - biopsy_confirmed: 是否已活检确诊
    - risk_factors: 风险因素列表
    """
```

工具内部实现了智能路由函数 `should_use_llm()`，根据文本长度、是否存在模糊表述等因素决定解析策略。文本长度小于 100 字符时直接使用规则解析；超过 1000 字符的长文本调用 LLM 增强解析；中等长度文本检查是否存在"可能"、"大概"、"建议进一步检查"等模糊表述，有则使用 LLM。

解析结果包含 `parsing_method` 字段标记使用的解析策略（rule_based 或 llm_enhanced），以及 `confidence_score` 字段反映整体解析的可信度。对于解析失败的场景，工具会自动回退到规则方案并添加说明注释。

#### 病理报告解析工具（PathologyParserTool）

病理报告解析工具从病理报告中提取金标准诊断信息，包括病理类型、分化程度、特殊病理特征和分子标志物状态。病理报告是癌症诊断的最终依据，该工具能够准确识别各种病理类型表述。

```python
@tool
def PathologyParserTool(pathology_text: str) -> Dict:
    """
    Parse pathology report to extract gold-standard diagnostic information.
    
    提取内容：
    - pathology_confirmed: 是否为恶性
    - histology_type: 病理类型 (adenocarcinoma/squamous_cell_carcinoma等)
    - differentiation_grade: 分化程度 (well/moderately/poorly differentiated)
    - special_features: 特殊病理特征
    - molecular_markers: 分子标志物状态 (MSI/MMR)
    - pathological_staging: 病理分期 (pT/pN)
    """
```

工具支持中英文混合识别，能够处理多种表述形式：例如"中分化腺癌"和"moderately differentiated adenocarcinoma"会被统一解析为相同的结构化数据。分化程度映射规则包括：G1/高分化/ well differentiated 对应 well_differentiated，G2/中分化/ moderately differentiated 对应 moderately_differentiated，G3/低分化/ poorly differentiated 对应 poorly_differentiated。

特殊病理特征检测包括：粘液成分（mucinous_component）、印戒细胞（signet_ring_cells）、神经侵犯（perineural_invasion）、脉管侵犯（lymphovascular_invasion）。分子标志物状态检测包括 MSI-H/MSS 和 dMMR/pMMR，这些信息对于免疫治疗决策至关重要。

#### CT 分期评估工具（VolumeCTSegmentorTool）

CT 分期评估工具分析 CT 报告用于肿瘤分期，重点关注远处转移灶的检测。该工具专注于 M 分期评估，能够识别肝脏、肺脏、腹膜等常见转移部位的影像学表现。

```python
@tool
def VolumeCTSegmentorTool(ct_report_text: str) -> Dict:
    """
    Text-mode CT report analyzer for colon cancer staging and distant metastasis detection.
    
    分析内容：
    - M 分期评估 (M0/M1a/M1b/M1c)
    - 转移部位检测 (liver/lung/peritoneum)
    - 淋巴结评估 (nodes_suspected)
    - 远处转移摘要
    """
```

M 分期判定规则：M1a 表示单一器官转移（仅肝脏或仅肺部），M1b 表示多器官转移（至少两个部位），M1c 表示腹膜转移。工具会检测以下关键词模式来判断转移灶：肝脏转移检测"低密度"、"占位"、"转移"、"结节"、"mass"；肺部转移检测"结节"、"转移"、"占位"；腹膜转移检测"腹膜"、"腹水"、"种植"。

#### 直肠 MRI 分期工具（RectalMRStagerTool）

直肠 MRI 分期工具专门用于直肠癌的局部分期评估，是直肠癌新辅助治疗决策的关键依据。该工具检测多个局部复发风险因素，包括 MRF 状态、EMVI 状态、CRM 状态以及 T、N 分期。

```python
@tool
def RectalMRStagerTool(text_context: str) -> Dict:
    """
    Text-mode rectal MRI report analyzer for local staging (T, N, MRF, EMVI).
    
    分析内容：
    - MRF (Mesorectal Fascia) 状态
    - EMVI (Extramural Vascular Invasion) 状态
    - CRM (Circumferential Resection Margin) 状态
    - T 分期 (T1/T2/T3/T4)
    - N 分期 (N0/N1/N2)
    - 新辅助治疗建议
    """
```

新辅助治疗推荐基于以下高危因素：MRF 阳性、EMVI 阳性、T3/T4 期或 N1/N2 期。这些因素符合 NCCN 指南中关于新辅助放化疗的适应症标准，工具输出的 neoadjuvant_recommended 字段可直接用于治疗决策参考。

#### 分子标志物工具（MolecularGuidelineTool）

分子标志物工具检测可靶向治疗的基因突变状态，输出治疗建议的参考信息。该工具主要检测 KRAS、NRAS、BRAF 三个靶点，这些基因状态决定了是否能使用 EGFR 抗体类靶向药物。

---

### 3. RAG 检索工具（rag_tools.py）

RAG 检索工具模块封装了混合检索能力，为 LLM 提供基于临床指南的知识查询功能。该模块与 src/rag 模块深度集成，支持多种检索模式和元数据过滤，能够从已索引的临床指南中精准检索相关信息。

#### 基础临床指南搜索工具（ClinicalGuidelineSearchTool）

基础临床指南搜索工具是默认的知识检索入口，适用于查询标准治疗原则、指南建议等通用医学知识。该工具使用混合检索策略，结合向量相似度和 BM25 关键词匹配，并通过重排序优化结果质量。

```python
class ClinicalGuidelineSearchTool(BaseTool):
    name: str = "search_clinical_guidelines"
    
    def _run(self, query: str, top_k: int = 6) -> str:
        docs = hybrid_search(query=q, k=top_k, use_rerank=True)
        return _format_docs(docs or [])
```

工具返回格式化为特定结构的文本，包含引用锚点格式 `[[Source:Filename|Page:N]]`，便于 LLM 生成带引用的回答。同时在结果末尾附加 `<retrieved_metadata>` JSON 块，包含每个检索片段的详细信息（来源、页码、预览、评分等），供下游节点解析使用。

#### 治疗方案搜索工具（TreatmentSearchTool）

治疗方案搜索工具专门用于检索化疗方案、靶向治疗方案、免疫治疗方案等治疗相关内容。该工具支持按疾病类型过滤，适用于查询特定分期、特定分子分型患者的治疗建议。

```python
class TreatmentSearchTool(BaseTool):
    name: str = "search_treatment_recommendations"
    
    适用场景：
    1. 查询特定化疗方案 (FOLFOX vs XELOX)
    2. 查询线数特异性治疗 (一线/二线/三线)
    3. 查询分期特异性治疗建议
    4. 新辅助/辅助治疗策略
```

工具通过元数据过滤机制优先检索内容类型为"治疗方案"的片段，确保返回结果与治疗决策直接相关。

#### 分期标准搜索工具（StagingSearchTool）

分期标准搜索工具专门用于查询 TNM 分期的定义和标准。当分期评估节点需要验证分期逻辑或查询分期依据时，可调用此工具获取官方分期标准。

```python
class StagingSearchTool(BaseTool):
    name: str = "search_staging_criteria"
    
    示例查询：
    - "T3 分期标准是什么？"
    - "N2a 淋巴结如何定义？"
    - "TNM 分期分类体系"
```

#### 药物信息搜索工具（DrugInfoSearchTool）

药物信息搜索工具用于检索药物的说明书信息，包括用法用量、禁忌症、不良反应、药物相互作用等。该工具与治疗方案搜索工具的区别在于：药物信息工具查询单一药物的详细信息，治疗方案工具查询多个药物组合的治疗方案。

```python
class DrugInfoSearchTool(BaseTool):
    name: str = "search_drug_information"
    
    适用场景：
    - 药物用法用量查询
    - 禁忌症和不良反应
    - 靶向治疗标记物要求 (如 RAS 状态)
    - 药物相互作用信息
```

#### 指南来源搜索工具（GuidelineSourceSearchTool）

指南来源搜索工具允许指定检索特定来源的指南内容，支持 NCCN（美国）、CSCO（中国）、ESMO（欧洲）三大权威机构。当用户明确要求参考特定地区指南时，使用此工具可确保结果符合对应地区的临床实践标准。

#### 列出指南目录工具（ListGuidelineTOCTool）

列出指南目录工具用于获取所有已索引的临床指南的目录结构，包括章节标题和页码范围。该工具可以帮助 LLM 了解指南库的整体结构，便于更有针对性地检索特定章节。

#### 读取指南章节工具（ReadGuidelineChapterTool）

读取指南章节工具用于精准读取指南的特定章节内容。当 LLM 需要获取某个指南的完整章节信息时，可调用此工具获取该章节的所有文本内容，确保信息的完整性和准确性。

#### 混合检索工具（HybridSearchTool）

混合检索工具提供高级检索能力，支持同时应用多个元数据过滤条件。当标准检索返回过多不相关结果时，可通过此工具精确限定内容类型、疾病类型、指南来源等约束条件。

#### 检索结果格式化

所有 RAG 工具共享统一的格式化函数 `_format_docs()`，该函数生成"双通道"输出格式：

```python
# 文本通道（供 LLM 阅读）
[REF_1] [[Source:CSCO_2024.pdf|Page:45]]
Content: ...

# 数据通道（供代码解析）
<retrieved_metadata>[{"ref_id": "REF_1", "source": "CSCO_2024.pdf", "page": 45, ...}]</retrieved_metadata>
```

文本通道使用 `[[Source:Filename|Page:N]]` 格式的引用锚点，训练 LLM 学会在回答中引用这些来源。数据通道使用 XML 标签包裹 JSON，方便正则提取，使 State 节点能够获得精确的页码和文件路径信息用于后续处理。

---

### 4. 联网搜索工具（web_search_tools.py）

联网搜索工具模块封装了实时网络搜索能力，用于获取最新的临床研究、指南更新、药物信息等动态资料。与静态的 RAG 知识库相比，联网搜索能够突破知识库的时效性限制，获取发布时尚未被收录的最新医学证据。

#### 通用联网搜索工具（WebSearchTool）

通用联网搜索工具是基础的网络搜索入口，可执行任意医学主题的实时查询。该工具使用 gpt-4o-search-preview 模型进行搜索，返回结果会明确标注来源出处。

```python
class WebSearchTool(BaseTool):
    name: str = "web_search"
    
    def _run(self, query: str, context: Optional[str] = None) -> str:
        """
        参数：
        - query: 搜索查询内容（必填）
        - context: 上下文信息（可选）
        
        返回：
        - 搜索结果文本，如无相关资料则明确返回"没有找到相关资料"
        """
```

工具内置严格的结果验证机制：如果响应内容包含"没有找到"、"未找到"等模式，或内容长度小于 100 字符，则标记为无结果状态。这一设计确保工具不会为不存在的信息编造答案。

#### 临床证据搜索工具（ClinicalEvidenceSearchTool）

临床证据搜索工具专门搜索高质量的临床证据，优先检索 NCCN/ESMO/CSCO 指南、RCT 研究、Meta 分析、PubMed 文献等权威来源。该工具适用于需要最新临床证据支持治疗决策的场景。

```python
class ClinicalEvidenceSearchTool(BaseTool):
    name: str = "search_clinical_evidence"
    
    def _run(self, topic: str, disease: Optional[str] = None, treatment: Optional[str] = None):
        """
        参数：
        - topic: 临床主题
        - disease: 疾病名称（可选）
        - treatment: 治疗方案（可选）
        """
```

#### 药物信息联网搜索工具（DrugInfoSearchTool）

药物信息联网搜索工具搜索处方药物的详细信息，优先检索官方药品说明书、FDA/NMPA 批准信息、权威药物数据库。该工具专门针对处方药物（化疗药物、靶向药物、免疫药物），不适用于营养补充剂或非处方药。

```python
class DrugInfoSearchTool(BaseTool):
    name: str = "search_drug_online"
    
    info_type 参数：
    - all: 完整信息
    - dosage: 用法用量
    - interaction: 药物相互作用
    - adverse: 不良反应
    - indication: 适应症
```

#### 指南更新搜索工具（GuidelineUpdateSearchTool）

指南更新搜索工具用于追踪临床指南的最新变化和版本更新。当需要了解最新指南与旧版本的差异，或确认某项推荐是否仍为当前标准时，可调用此工具获取最新指南信息。

#### 最新研究搜索工具（LatestResearchSearchTool）

最新研究搜索工具搜索最新的临床研究文献，优先检索 PubMed、Cochrane 库和高影响因子期刊。该工具适用于了解前沿研究进展、获取最新临床证据。

---

### 5. 数据库查询工具（database_tools.py）

数据库查询工具模块封装了虚拟病例数据库的查询能力，为 LLM 提供结构化的临床病例数据访问接口。该模块与 src.services.virtual_database_service 深度集成，支持患者信息查询、病例搜索、统计分析等功能。

#### 查询患者病历工具（get_patient_case_info）

根据患者 ID 查询详细病历信息，包括基本人口学信息、肿瘤位置、病理类型、分期、分子标志物等关键临床数据。这是最常用的原子化工具，其他复合工具都基于此构建。

```python
@tool
def get_patient_case_info(patient_id: int) -> Dict[str, Any]:
    """
    根据患者ID查询详细病历信息。
    
    参数：
        patient_id: 患者编号 (数字，例如 93)
    
    返回：
        包含患者完整病历信息的字典
    """
```

#### 获取患者影像工具（get_patient_imaging）

获取患者的影像资料路径信息，支持按患者 ID 或文件夹名称查询。该工具返回影像文件的详细路径列表，可用于后续的影像分析处理。

```python
@tool
def get_patient_imaging(patient_id: str) -> Dict[str, Any]:
    """
    获取患者的影像资料/图片/CT/MRI。
    
    参数：
        patient_id: 患者ID (例如 "093" 或 "93")
    
    返回：
        包含影像信息的字典
    """
```

#### 数据库统计工具（get_database_statistics）

查询虚拟病例数据库的整体统计信息，包括病例总数、性别分布、年龄统计、肿瘤部位分布、分期分布等。该工具适用于了解数据集的整体概况和患者群体特征。

```python
@tool
def get_database_statistics() -> Dict[str, Any]:
    """
    查询数据库整体统计信息。
    
    返回：
        包含统计信息的字典
    """
```

#### 病例搜索工具（search_cases）

支持多条件组合搜索，可按肿瘤部位、分期、病理类型、分子标志物、年龄范围、CEA 水平等条件筛选病例。该工具适用于查找特定类型的病例进行对比分析或参考。

```python
@tool
def search_cases(
    tumor_location: Optional[str] = None,
    ct_stage: Optional[str] = None,
    cn_stage: Optional[str] = None,
    histology_type: Optional[str] = None,
    mmr_status: Optional[int] = None,
    age_min: Optional[int] = None,
    age_max: Optional[int] = None,
    cea_max: Optional[float] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    搜索虚拟病例数据库，查找符合条件的临床病例数据。
    
    支持的搜索条件：
    - tumor_location: 肿瘤部位
    - ct_stage: cT分期
    - cn_stage: cN分期
    - histology_type: 组织类型
    - mmr_status: MMR状态
    - age_min, age_max: 年龄范围
    - cea_max: 最大CEA水平
    - limit: 返回结果数量 (默认10)
    """
```

#### 列出影像文件夹工具（list_imaging_folders）

列出所有可用的影像文件夹名称，返回格式化的患者 ID 列表。该工具适用于快速了解数据库中有哪些患者的影像资料。

```python
@tool
def list_imaging_folders() -> List[str]:
    """
    列出所有可用的影像文件夹名称。
    
    返回：
        所有患者影像文件夹名称列表
    """
```

#### 随机病例工具（get_random_case）

随机获取一个病例，可指定筛选条件。该工具适用于演示、测试或随机抽取参考案例的场景。

```python
@tool
def get_random_case(
    tumor_location: Optional[str] = None,
    ct_stage: Optional[str] = None,
    mmr_status: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    随机获取一个病例（可指定部位）。
    
    可选筛选条件：
    - tumor_location: 限定肿瘤部位
    - ct_stage: 限定cT分期
    - mmr_status: 限定MMR状态
    """
```

#### 完整肿瘤检测工具（perform_comprehensive_tumor_check）

高级聚合工具，输入患者 ID，自动执行完整的肿瘤检测流程。该工具集成了 YOLO 模型检测，遍历患者所有影像进行肿瘤筛查，返回详细的检测报告。

```python
@tool
def perform_comprehensive_tumor_check(patient_id: str) -> Dict:
    """
    高级聚合工具：输入患者ID，自动执行完整的肿瘤检测流程。
    
    功能说明：
    - 输入患者ID（如 "93" 或 "093"）
    - 自动查找对应的影像文件夹
    - 遍历所有CT切片进行肿瘤检测
    - 返回汇总的检测报告
    
    触发条件：
    - 用户想对患者影像进行肿瘤检测、癌症筛查、病灶识别
    - 触发关键词：肿瘤检测、癌症筛查、影像诊断、病灶分析、CT检测
    """
```

---

### 6. 肿瘤筛选工具（tumor_screening_tools.py）

肿瘤筛选工具模块基于 YOLOv8 深度学习模型，提供 CT 图像的肿瘤检测和筛选功能。该模块采用延迟导入策略，避免因依赖缺失导致系统启动失败。

#### 批量肿瘤筛选工具（tumor_screening_tool）

批量肿瘤筛选工具遍历指定目录下的所有 CT 图像，使用训练好的 YOLO 模型进行肿瘤检测，将检测到肿瘤的图像复制到输出目录。该工具支持保持原始文件夹结构，并返回详细的筛选报告。

```python
@tool
def tumor_screening_tool(
    input_dir: str,
    output_dir: str = None,
    model_path: str = None,
    confidence_threshold: float = 0.5,
    image_extensions: List[str] = None
) -> Dict:
    """
    肿瘤图像筛选工具 - 使用YOLO模型在CT图像目录中筛选包含肿瘤的图像
    
    功能说明：
    - 遍历指定目录下的所有图像
    - 使用训练好的YOLO模型进行肿瘤检测
    - 将检测到肿瘤的图像复制到输出目录
    - 支持保持原始文件夹结构
    - 返回详细的筛选报告
    
    输入参数：
    - input_dir: 输入图像目录路径
    - output_dir: 输出目录路径（可选）
    - model_path: YOLO模型文件路径（可选，使用默认模型）
    - confidence_threshold: 检测置信度阈值，0-1之间（默认0.5）
    - image_extensions: 图像文件扩展名列表（可选）
    
    注意事项：
    - 模型文件(best.pt)需要预先训练并放置在正确位置
    - 输入目录不能包含中文字符（可能导致路径读取问题）
    - 建议置信度阈值设置在0.5-0.7之间，平衡精确率和召回率
    """
```

#### 快速肿瘤检测工具（quick_tumor_check）

快速肿瘤检测工具对单张 CT 图像进行肿瘤检测，返回检测结果和置信度。该工具适用于随机抽查或验证场景。

```python
@tool
def quick_tumor_check(image_path: str, model_path: str = None) -> Dict:
    """
    快速肿瘤检测工具 - 检测单张图像是否包含肿瘤
    
    功能说明：
    - 对单张CT图像进行快速肿瘤检测
    - 返回检测结果和置信度
    - 适用于随机抽查或验证场景
    
    输出说明：
    - has_tumor: 是否检测到肿瘤 (true/false)
    - confidence: 最高检测置信度
    - bounding_boxes: 检测框坐标列表
    - processing_time: 处理时间信息
    """
```

#### 肿瘤筛选状态查询工具（get_tumor_screening_status）

查询肿瘤筛选工具的状态信息，包括模型加载状态、工具版本、支持的图像格式等。该工具用于诊断工具使用问题。

```python
@tool
def get_tumor_screening_status() -> Dict:
    """
    获取肿瘤筛选工具状态信息
    
    功能说明：
    - 检查模型是否已加载
    - 返回工具版本和配置信息
    - 诊断工具使用问题
    """
```

---

### 7. 肿瘤定位工具（tumor_localization_tools.py）

肿瘤定位工具模块基于 U-Net 深度学习模型，提供 CT 图像的肿瘤分割和定位功能。该模块生成像素级的分割掩膜，提取肿瘤的位置、大小、边界框等精确信息。

#### 肿瘤定位工具（tumor_localization_tool）

肿瘤定位工具对单张 CT 图像进行精确的肿瘤分割，生成肿瘤区域的分割掩膜，并提取肿瘤位置、大小、边界框等信息。该工具支持生成可视化结果图像。

```python
@tool
def tumor_localization_tool(
    image_path: str,
    output_path: str = None,
    model_path: str = None,
    threshold: float = 0.5,
    save_visualization: bool = True
) -> Dict:
    """
    肿瘤定位工具 - 使用U-Net模型对CT图像进行肿瘤分割和定位
    
    功能说明：
    - 对单张CT图像进行精确的肿瘤分割
    - 生成肿瘤区域的分割掩码
    - 提取肿瘤位置、大小、边界框等信息
    - 可选生成可视化结果图像
    
    输入参数：
    - image_path: 输入CT图像路径
    - output_path: 输出掩码图像路径（可选，默认在原图旁保存）
    - model_path: U-Net模型权重文件路径（可选，使用默认模型）
    - threshold: 分割阈值，0-1之间（默认0.5）
    - save_visualization: 是否保存可视化结果（默认True）
    
    输出说明：
    - success: 是否成功完成分割
    - has_tumor: 是否检测到肿瘤
    - tumor_count: 检测到的肿瘤区域数量
    - total_area: 肿瘤总面积（像素）
    - regions: 每个肿瘤区域的详细信息列表
    - mask_path: 保存的掩码图像路径
    - visualization_path: 可视化结果图像路径
    
    注意事项：
    - 模型权重文件需要预先训练并放置在正确位置
    - 输入图像会被自动调整为256x256进行推理
    - 输出掩码会调整回原始图像尺寸
    """
```

#### 批量肿瘤定位工具（batch_tumor_localization）

批量肿瘤定位工具对目录中的所有 CT 图像进行批量分割和定位，保持原始文件夹结构，并生成批量处理报告。该工具适用于处理大量影像数据。

```python
@tool
def batch_tumor_localization(
    input_dir: str,
    output_dir: str = None,
    model_path: str = None,
    threshold: float = 0.5,
    image_extensions: List[str] = None
) -> Dict:
    """
    批量肿瘤定位工具 - 对目录中的所有CT图像进行批量分割和定位
    
    功能说明：
    - 遍历指定目录下的所有图像
    - 对每张图像进行肿瘤分割
    - 保持原始文件夹结构
    - 生成批量处理报告
    
    输出说明：
    - success: 是否成功完成批量处理
    - total_images: 处理的总图像数量
    - images_with_tumor: 检测到肿瘤的图像数量
    - images_without_tumor: 未检测到肿瘤的图像数量
    - tumor_detection_rate: 肿瘤检出率
    - sample_results: 样本结果列表（最多10个）
    """
```

#### 肿瘤定位状态查询工具（get_localization_status）

查询肿瘤定位工具的状态信息，包括模型加载状态、使用的设备（CPU/CUDA）、工具版本等。

```python
@tool
def get_localization_status() -> Dict:
    """
    获取肿瘤定位工具状态信息
    
    功能说明：
    - 检查模型是否已加载
    - 返回工具版本和配置信息
    - 诊断工具使用问题
    """
```

---

### 8. 影像组学工具（radiomics_tools.py）

影像组学工具模块提供完整的影像分析工具链：U-Net 肿瘤分割 → PyRadiomics 特征提取 → LASSO 特征筛选。该模块适用于深度影像分析和研究场景。

#### U-Net 分割工具（unet_segmentation_tool）

U-Net 分割工具对 CT 图像进行像素级分割，生成二值分割掩膜，标记肿瘤区域。该工具输出分割结果图像和统计信息。

```python
@tool
def unet_segmentation_tool(
    image_path: str,
    output_dir: str = None,
    model_path: str = None,
    confidence_threshold: float = 0.5
) -> Dict:
    """
    U-Net 肿瘤分割工具 - 对CT图像进行精确的肿瘤区域分割
    
    功能说明：
    - 使用训练好的 U-Net 模型对CT图像进行像素级分割
    - 生成二值分割掩膜（Mask），标记肿瘤区域
    - 输出分割结果图像和统计信息
    
    输出说明：
    - success: 是否成功完成分割
    - mask_path: 分割掩膜保存路径
    - tumor_area: 肿瘤区域面积（像素数）
    - tumor_ratio: 肿瘤区域占比
    - bounding_box: 肿瘤边界框坐标
    
    注意事项：
    - 需要预先训练好的 U-Net 模型文件
    - 输入图像应为灰度或RGB格式
    - 推荐使用 GPU 加速（CPU模式较慢）
    """
```

#### PyRadiomics 特征提取工具（radiomics_feature_extraction_tool）

PyRadiomics 特征提取工具从 CT 图像和对应的分割掩膜中提取影像组学特征，支持多种特征类别：形状、纹理、灰度、小波变换等。该工具输出约 1500 维特征向量，用于后续机器学习分析。

```python
@tool
def radiomics_feature_extraction_tool(
    image_path: str,
    mask_path: str,
    output_dir: str = None,
    feature_classes: List[str] = None
) -> Dict:
    """
    PyRadiomics 影像组学特征提取工具 - 提取高维影像组学特征
    
    功能说明：
    - 从CT图像和对应的分割掩膜中提取影像组学特征
    - 支持多种特征类别：形状、纹理、灰度、小波变换等
    - 输出约1500维特征向量，用于后续机器学习分析
    
    输入参数：
    - image_path: CT图像文件路径
    - mask_path: 分割掩膜文件路径（U-Net输出）
    - feature_classes: 特征类别列表（可选，默认提取所有类别）
      可选: ['shape', 'firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
    
    输出说明：
    - success: 是否成功提取特征
    - feature_count: 提取的特征数量
    - features: 特征字典 {feature_name: value}
    - feature_categories: 特征分类统计
    - output_file: 特征保存路径（JSON格式）
    
    注意事项：
    - 需要安装 pyradiomics 和 SimpleITK
    - 图像和掩膜尺寸必须一致
    - 特征提取耗时较长（约10-30秒/图像）
    """
```

#### LASSO 特征筛选工具（lasso_feature_selection_tool）

LASSO 特征筛选工具使用 LASSO（Least Absolute Shrinkage and Selection Operator）算法，从约 1500 维影像组学特征中筛选出最重要的 Top-K 特征。该工具提供特征重要性评分和统计参考值。

```python
@tool
def lasso_feature_selection_tool(
    features_dict: Dict[str, float],
    top_k: int = 20,
    alpha: float = None,
    output_dir: str = None
) -> Dict:
    """
    LASSO 特征筛选工具 - 从高维特征中筛选最重要的特征
    
    功能说明：
    - 使用 LASSO 算法从约1500维影像组学特征中筛选出最重要的 Top-K 特征
    - 提供特征重要性评分和统计参考值
    
    输入参数：
    - features_dict: 特征字典 {feature_name: value}
    - top_k: 筛选的特征数量（默认20）
    - alpha: LASSO 正则化参数（可选，自动交叉验证）
    - output_dir: 结果保存目录（可选）
    
    输出说明：
    - success: 是否成功筛选
    - selected_features: 筛选出的 Top-K 特征列表
    - feature_importance: 特征重要性得分
    - statistics: 特征统计信息（均值、标准差、范围）
    
    注意事项：
    - 需要安装 scikit-learn
    - 单个样本无法进行 LASSO 训练，会基于特征方差进行筛选
    - 建议积累多个样本数据后进行真正的 LASSO 筛选
    """
```

#### 完整影像组学分析工具（comprehensive_radiomics_analysis）

完整影像组学分析工具一键执行完整的影像分析流程：U-Net 分割 → PyRadiomics 特征提取 → LASSO 特征筛选。该工具适用于单张 CT 图像的深度分析。

```python
@tool
def comprehensive_radiomics_analysis(
    image_path: str,
    output_dir: str = None,
    top_k_features: int = 20
) -> Dict:
    """
    完整影像组学分析工具 - 一键执行完整的影像分析流程
    
    功能说明：
    - 自动执行：U-Net分割 → PyRadiomics特征提取 → LASSO特征筛选
    - 生成完整的影像组学分析报告
    - 适用于单张CT图像的深度分析
    
    输入参数：
    - image_path: CT图像文件路径
    - output_dir: 输出目录（可选）
    - top_k_features: 筛选的特征数量（默认20）
    
    输出说明：
    - success: 是否成功完成分析
    - segmentation: 分割结果
    - radiomics: 特征提取结果
    - feature_selection: 特征筛选结果
    - summary: 分析总结
    
    注意事项：
    - 完整流程耗时较长（约30-60秒/图像）
    - 建议在GPU环境运行
    - 需要安装所有依赖：torch, opencv-python, pyradiomics, SimpleITK, scikit-learn
    """
```

---

### 9. 病理 CLAM 工具（pathology_clam_tools.py）

病理 CLAM 工具模块基于 CLAM（Clustering-constrained Attention Multiple Instance Learning）深度学习模型，提供病理全切片图像（WSI）的分析功能。该模块支持 .svs、.tif、.ndpi 等全切片图像格式，能够自动进行组织分割、特征提取、分类预测和注意力热力图生成。

#### 病理切片分类工具（pathology_slide_classify）

病理切片分类工具对全切片图像进行肿瘤/正常分类，生成注意力热力图和高关注区域切片。该工具自动执行完整的 CLAM 分析流程：组织分割 → 切片提取 → 特征提取 → 模型推理 → 热力图生成。

```python
@tool
def pathology_slide_classify(
    slide_path: str,
    output_dir: str = None,
    model_path: str = None,
    generate_heatmap: bool = True,
    extract_topk: bool = True,
    topk: int = 10
) -> Dict:
    """
    病理切片分类工具 - 使用 CLAM 模型对全切片图像进行肿瘤/正常分类
    
    功能说明：
    - 对 .svs 等格式的病理全切片图像进行 AI 分析
    - 自动分割组织区域并提取特征
    - 使用注意力机制定位可疑区域
    - 输出分类结果、热力图和高关注区域切片
    
    输入参数：
    - slide_path: 病理切片文件路径（支持 .svs, .tif, .tiff, .ndpi 格式）
    - output_dir: 输出目录路径（可选）
    - model_path: CLAM 模型路径（可选，使用默认模型）
    - generate_heatmap: 是否生成注意力热力图（默认 True）
    - extract_topk: 是否提取高关注区域切片（默认 True）
    - topk: 提取的高关注区域数量（默认 10）
    
    输出说明：
    - success: 是否成功完成分析
    - slide_id: 切片标识符
    - prediction: 预测结果（TUMOR/NORMAL）
    - tumor_probability: 肿瘤概率
    - normal_probability: 正常概率
    - confidence: 置信度
    - heatmap_path: 热力图文件路径
    - topk_patches_dir: Top-K 切片目录
    - report_path: 分析报告路径
    
    注意事项：
    - 首次运行需要下载 ResNet50 预训练权重
    - 处理大型切片文件可能需要较长时间（几分钟到十几分钟）
    - 需要 openslide-python 库来读取 .svs 文件
    - GPU 可以显著加速处理过程
    """
```

#### 快速病理检查工具（quick_pathology_check）

快速病理检查工具对切片进行快速分析，仅返回分类结果，不生成热力图和切片。该工具适用于大批量筛选场景。

```python
@tool
def quick_pathology_check(slide_path: str) -> Dict:
    """
    快速病理检查工具 - 快速检测切片是否可能包含肿瘤
    
    功能说明：
    - 对病理切片进行快速分析
    - 仅返回分类结果，不生成热力图和切片
    - 适用于大批量筛选场景
    
    输出说明：
    - success: 是否成功
    - prediction: 预测结果（TUMOR/NORMAL）
    - tumor_probability: 肿瘤概率
    - confidence: 置信度
    """
```

#### 病理 CLAM 状态查询工具（get_pathology_clam_status）

查询病理 CLAM 工具的状态信息，包括模型加载状态、依赖检查、GPU 可用性等。

```python
@tool
def get_pathology_clam_status() -> Dict:
    """
    获取病理 CLAM 工具状态信息
    
    功能说明：
    - 检查模型是否已加载
    - 检查依赖是否已安装
    - 返回工具版本和配置信息
    
    输出说明：
    - model_loaded: 模型是否已加载
    - model_path: 当前/默认模型路径
    - model_exists: 模型文件是否存在
    - dependencies: 依赖检查结果
    - missing_dependencies: 缺失的依赖列表
    - gpu_available: GPU 是否可用
    - supported_formats: 支持的切片格式
    """
```

#### 综合病理分析工具（perform_comprehensive_pathology_analysis）

高级聚合工具，输入患者 ID，自动查找并分析所有病理切片，返回汇总诊断报告。

```python
@tool
def perform_comprehensive_pathology_analysis(patient_id: str) -> Dict:
    """
    高级聚合工具：输入患者 ID，自动执行完整的病理切片分析
    
    功能说明：
    - 输入患者 ID，自动查找对应的病理切片文件
    - 对所有切片执行 CLAM 分析
    - 返回汇总的诊断报告
    
    触发条件：
    - 用户需要病理诊断、组织病变分析、癌症分型
    - 触发关键词：病理分析、切片诊断、组织学检查、病理报告
    """
```

---

### 10. 卡片格式化工具（card_formatter.py）

卡片格式化工具模块负责将数据库原始数据转换为前端可渲染的 Card 格式。该模块是纯展示逻辑，不包含任何意图识别或数据库查询功能。

#### CardFormatter 类

CardFormatter 类提供多种格式化方法，将不同类型的数据转换为标准的前端卡片格式。

```python
class CardFormatter:
    """UI 卡片数据生成器"""
```

#### 患者卡片格式化（format_patient_card）

将原始病例数据转换为标准 Patient Card 格式，适用于前端弹窗展示。该方法从虚拟数据库获取病例信息，并构造前端需要的标准结构。

```python
def format_patient_card(self, patient_id: int) -> Dict[str, Any]:
    """
    将原始病例数据转换为标准 Patient Card 格式 (前端弹窗用)
    
    返回：
        - type: "patient_card"
        - patient_id: 患者编号
        - data: 包含 diagnosis_block, staging_block, patient_info, raw_data
        - text_summary: 文本摘要
    """
```

#### 影像卡片格式化（format_imaging_card）

将影像查询结果转换为标准 Imaging Card 格式，适用于展示影像样本信息。

```python
def format_imaging_card(self, imaging_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    将影像查询结果转换为标准 Imaging Card 格式
    (从 atomic tool get_patient_imaging 的结果转换而来)
    """
```

#### 肿瘤筛选结果格式化（format_tumor_screening_result）

将肿瘤检测结果转换为标准 Tumor Screening Result 格式，包含检测结论、置信度、检测结果等。

```python
def format_tumor_screening_result(self, screening_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    将肿瘤检测结果转换为标准 Tumor Screening Result 格式
    (从 perform_comprehensive_tumor_check 的结果转换而来)
    """
```

#### 完整肿瘤检测格式化（format_comprehensive_tumor_detection）

将完整的肿瘤检测结果转换为前端 Tumor Detection Card 格式，支持批量图片显示和 base64 编码。

```python
def format_comprehensive_tumor_detection(
    self, 
    screening_result: Dict[str, Any], 
    include_images: bool = True
) -> Dict[str, Any]:
    """
    将完整的肿瘤检测结果转换为前端 Tumor Detection Card 格式
    (从 perform_comprehensive_tumor_check 的结果转换而来，支持批量图片显示)
    
    Args:
        screening_result: perform_comprehensive_tumor_check 返回的原始结果
        include_images: 是否读取并包含图片的 base64 数据
    """
```

---

## 工具工厂函数

Tools 模块提供了多个工厂函数用于批量获取工具实例，这些函数封装了工具初始化的复杂性，直接返回可供 LLM 调用的工具列表。

```python
def list_tools() -> list:
    """
    返回完整工具注册表（包含临床工具和 RAG 基础工具）
    包含联网搜索能力，可以实时获取最新资料。
    包含肿瘤筛选、肿瘤定位、影像组学等AI分析工具。
    """
```

```python
def list_tools_with_web_search() -> list:
    """
    返回包含联网搜索能力的完整工具集
    """
```

```python
def list_all_tools() -> list:
    """
    返回所有可用工具，包括：
    - 临床工具
    - RAG 工具（所有变体）
    - 联网搜索工具
    - 数据库工具
    - 肿瘤筛选工具
    - 肿瘤定位工具
    - 影像组学工具
    """
```

```python
def get_all_rag_tools() -> list[BaseTool]:
    """获取所有 RAG 工具"""
```

```python
def get_enhanced_rag_tools() -> list[BaseTool]:
    """获取增强版 RAG 工具集（推荐使用）"""
```

```python
def get_clinical_web_search_tools() -> list[BaseTool]:
    """获取临床相关的联网搜索工具集"""
```

```python
def get_database_tools() -> List[Any]:
    """获取所有数据库工具"""
```

```python
def get_tumor_screening_tools():
    """获取肿瘤筛选工具集"""
```

```python
def get_tumor_localization_tools():
    """获取肿瘤定位工具集"""
```

```python
def list_radiomics_tools():
    """获取影像组学工具集"""
```

```python
def get_pathology_clam_tools():
    """获取病理 CLAM 工具集"""
```

---

## 使用示例

### 导入和使用工具

```python
from src.tools import (
    list_tools,
    list_all_tools,
    list_clinical_tools,
)
from src.tools.clinical_tools import (
    PatientHistoryParserTool,
    PathologyParserTool,
    VolumeCTSegmentorTool,
)
from src.tools.rag_tools import (
    ClinicalGuidelineSearchTool,
    TreatmentSearchTool,
)
from src.tools.web_search_tools import (
    WebSearchTool,
    ClinicalEvidenceSearchTool,
)
from src.tools.database_tools import (
    get_patient_case_info,
    get_patient_imaging,
    search_cases,
)
from src.tools.tumor_screening_tools import (
    perform_comprehensive_tumor_check,
    quick_tumor_check,
)
from src.tools.tumor_localization_tools import (
    tumor_localization_tool,
    batch_tumor_localization,
)
from src.tools.radiomics_tools import (
    unet_segmentation_tool,
    radiomics_feature_extraction_tool,
    lasso_feature_selection_tool,
    comprehensive_radiomics_analysis,
)
from src.tools.pathology_clam_tools import (
    pathology_slide_classify,
    quick_pathology_check,
    get_pathology_clam_status,
)
from src.tools.card_formatter import (
    CardFormatter,
    formatter,
)
```

### 调用临床工具

```python
# 解析患者病史
history_text = """
患者男性，62岁，因间断便血2个月入院。
肠镜检查提示直肠距肛门8cm见一肿物，大小约3×4cm，表面溃烂。
活检病理示中分化腺癌。
既往体健，否认肿瘤家族史。
吸烟史30年，每日20支。
"""

parser = PatientHistoryParserTool()
result = parser.invoke({"history_text": history_text})
print(result)
# 输出: {'chief_complaint': '间断便血2个月', 'tumor_location': 'rectum', ...}
```

```python
# 分析 CT 报告
ct_report = """
腹部CT示：肝脏形态大小正常，肝右叶见一低密度占位，大小约2.5×3cm，
边界清，增强扫描呈环形强化。脾脏、胰腺未见异常。
腹腔未见肿大淋巴结，未见腹水。
"""

ct_tool = VolumeCTSegmentorTool()
result = ct_tool.invoke({"ct_report_text": ct_report})
print(result)
# 输出: {'m_stage': 'M1a', 'metastasis_sites': ['liver'], 'liver_lesion': True, ...}
```

### 调用数据库工具

```python
# 查询患者病历
case_info = get_patient_case_info.invoke({"patient_id": 93})
print(case_info)
# 输出: {'patient_id': 93, 'gender': '男', 'age': 62, 'tumor_location': '直肠', ...}

# 搜索病例
results = search_cases.invoke({
    "tumor_location": "直肠",
    "ct_stage": "3",
    "limit": 5
})
print(f"找到 {len(results)} 个符合条件的病例")
```

### 调用肿瘤检测工具

```python
# 执行完整肿瘤检测
detection_result = perform_comprehensive_tumor_check.invoke({"patient_id": "93"})
print(detection_result)
# 输出: {'patient_id': '093', 'total_images': 50, 'images_with_tumor': 12, ...}

# 快速检测单张图像
quick_result = quick_tumor_check.invoke({"image_path": "/path/to/ct_image.png"})
print(quick_result)
# 输出: {'has_tumor': True, 'confidence': 0.8765, 'bounding_boxes': [...]}
```

### 调用影像组学工具

```python
# 完整影像组学分析
analysis_result = comprehensive_radiomics_analysis.invoke({
    "image_path": "/path/to/ct_image.png",
    "top_k_features": 20
})
print(analysis_result)
# 输出: {'success': True, 'segmentation': {...}, 'radiomics': {...}, ...}
```

### 调用病理 CLAM 工具

```python
# 病理切片分类
result = pathology_slide_classify.invoke({
    "slide_path": "/path/to/slide.svs",
    "generate_heatmap": True,
    "extract_topk": True,
    "topk": 10
})
print(result)
# 输出: {'success': True, 'prediction': 'TUMOR', 'tumor_probability': 0.8765, 
#        'heatmap_path': '/output/heatmaps/slide_heatmap.jpg', ...}

# 快速病理检查
quick_result = quick_pathology_check.invoke({
    "slide_path": "/path/to/slide.svs"
})
print(quick_result)
# 输出: {'success': True, 'prediction': 'TUMOR', 'tumor_probability': 0.8765, ...}

# 查看工具状态
status = get_pathology_clam_status.invoke({})
print(status)
# 输出: {'model_loaded': True, 'gpu_available': True, 'dependencies': {...}, ...}
```

### 调用卡片格式化工具

```python
# 格式化患者卡片
patient_card = formatter.format_patient_card(93)
print(patient_card)
# 输出: {'type': 'patient_card', 'patient_id': 93, 'data': {...}, ...}

# 格式化肿瘤检测结果
tumor_card = formatter.format_comprehensive_tumor_detection(detection_result)
print(tumor_card)
# 输出: {'type': 'tumor_detection_card', 'patient_id': '093', 'data': {...}, ...}
```

### 调用 RAG 工具

```python
# 检索治疗方案
treatment_tool = TreatmentSearchTool()
result = treatment_tool.invoke({
    "query": "III期结肠癌术后辅助化疗方案",
    "disease": "结肠癌",
    "top_k": 6
})

# 返回格式示例：
# [REF_1] [[Source:CSCO_2024.pdf|Page:45]]
# 对于III期结肠癌患者，推荐使用XELOX方案（奥沙利铂+卡培他滨）...
# 
# <retrieved_metadata>[{"ref_id": "REF_1", "source": "CSCO_2024.pdf", "page": 45, ...}]</retrieved_metadata>
```

### 调用联网搜索工具

```python
# 搜索最新临床证据
search_tool = ClinicalEvidenceSearchTool()
result = search_tool.invoke({
    "topic": "MSI-H结直肠癌免疫治疗",
    "disease": "结直肠癌",
    "treatment": "PD-1抑制剂"
})
print(result)
```

### 批量获取工具集

```python
# 获取所有工具（包含临床、RAG、联网搜索、数据库、AI分析）
all_tools = list_all_tools()

# 仅获取临床和 RAG 工具（不含联网搜索）
tools = list_tools()

# 仅获取临床工具
clinical_only = list_clinical_tools()
```

---

## 架构设计

### 工具调用流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LangGraph 工作流                             │
│  assessment_nodes / clinical_nodes / decision_nodes                  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM 智能体 (Agent)                                │
│  - 分析用户请求                                                      │
│  - 决定调用哪些工具                                                  │
│  - 解析工具返回结果                                                  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│  临床工具(6个)│       │  RAG 工具(8个) │       │数据库工具(7个)│
│ - PatientHistory│       │ - ClinicalGuideline│       │ - get_case_by_id│
│ - Pathology   │       │ - TreatmentSearch│       │ - get_imaging   │
│ - CTStage     │       │ - StagingSearch   │       │ - search_cases  │
│ - MRIStager   │       │ - DrugInfoSearch  │       │ - get_statistics│
│ - PolypDetect │       │ - GuidelineSource│       │ - list_folders  │
│ - Molecular   │       │ - HybridSearch   │       │ - random_case   │
└───────────────┘       │ - ListGuidelineTOC│       └───────────────┘
        │               │ - ReadGuidelineChap│
        │               └───────────────┘
        │                       │
        ▼                       ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│肿瘤筛选工具(4个)│       │肿瘤定位工具(3个)│       │影像组学工具(4个)│
│ - Screening   │       │ - Localization │       │ - U-Net Seg    │
│ - QuickCheck  │       │ - BatchLocate  │       │ - PyRadiomics  │
│ - Status      │       │ - Status       │       │ - LASSO        │
│ - Comprehensive│       └───────────────┘       │ - Comprehensive│
└───────────────┘               │               └───────────────┘
        │                       │                       │
        │               ┌───────────────┐               │
        │               │病理CLAM工具(4个)│               │
        │               │ - SlideClassify│               │
        │               │ - QuickCheck   │               │
        │               │ - Status       │               │
        │               │ - Comprehensive│               │
        │               └───────────────┘               │
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
                    ┌─────────────────────────┐
                    │   联网搜索工具 (5个)     │
                    │  - WebSearch            │
                    │  - ClinicalEvidence     │
                    │  - DrugInfoOnline       │
                    │  - GuidelineUpdates     │
                    │  - LatestResearch       │
                    └─────────────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │       返回结构化数据     │
                    │       或检索结果文本     │
                    │     → CardFormatter     │
                    │       转换为前端格式     │
                    └─────────────────────────┘
```

### 设计决策

**LangChain BaseTool 标准实现**：所有工具都继承自 BaseTool 类并实现 `_run()` 和可选的 `_arun()` 方法。这一设计确保工具可以无缝集成到 LangGraph 的 Agent 节点中，支持同步和异步两种调用模式。工具的 name、description、args_schema 字段遵循 LangChain 规范，使 LLM 能够自动理解工具用途和参数格式。

**Pydantic 输入模式约束**：RAG 工具、联网搜索工具和数据库工具使用 Pydantic BaseModel 定义输入模式（args_schema），通过 Field 类的 description 属性提供参数说明，通过 ge/le/min_length/pattern 等约束保证输入参数的有效性。LLM 根据这些约束自动生成符合规范的参数，避免参数错误导致的工具调用失败。

**混合解析策略**：临床工具中的病史解析采用混合策略，平衡处理速度和解析质量。规则引擎处理简单明确病例，LLM 处理复杂模糊病例，智能路由确保每种病例都使用最优策略。自动回退机制确保即使 LLM 不可用也能返回基本结果。

**双通道输出格式**：RAG 工具的输出同时服务于 LLM 和下游代码。文本通道使用可读的锚点格式供 LLM 引用，数据通道使用 JSON 格式供代码解析。这种设计避免了 LLM 解析结构化数据的困难，同时保留了引用溯源能力。

**延迟导入策略**：深度学习工具（肿瘤筛选、肿瘤定位、影像组学）采用延迟导入模式，torch、ultralytics、pyradiomics 等依赖在首次调用时才加载。这一设计避免了模块加载时因依赖缺失导致的启动失败，支持渐进式初始化。

**模型懒初始化**：肿瘤筛选、肿瘤定位、影像组学工具中的模型采用单例模式和懒加载，首次调用时才加载模型文件并移动到计算设备（CPU/GPU）。后续调用直接复用已加载的模型，避免重复加载的开销。

**前端格式化分离**：CardFormatter 类独立于业务逻辑，专注于将结构化数据转换为前端可渲染的格式。这种分离设计使得数据获取（数据库工具、AI 分析工具）和展示逻辑清晰解耦，便于前端适配和维护。

---

## 依赖说明

### 核心依赖

| 包名 | 用途 | 最低版本 |
|------|------|----------|
| langchain | 工具框架基础 | 0.1.x |
| langchain-core | 核心工具类定义 | 0.1.x |
| pydantic | 输入模式定义 | 2.x |

### 间接依赖

| 包名 | 用途 | 触发条件 |
|------|------|----------|
| src.rag | RAG 检索能力 | 使用 RAG 工具时 |
| src.services | LLM 和搜索服务 | 使用临床工具的 LLM 增强解析或联网搜索工具时 |
| torch | 深度学习框架 | 使用肿瘤定位、影像组学工具时 |
| ultralytics | YOLO 目标检测 | 使用肿瘤筛选工具时 |
| pyradiomics | 影像组学特征提取 | 使用影像组学工具时 |
| SimpleITK | 医学图像处理 | 使用影像组学工具时 |
| scikit-learn | 机器学习 | 使用 LASSO 特征筛选时 |
| opencv-python | 图像处理 | 使用肿瘤筛选、定位、影像组学工具时 |
| openslide-python | WSI 读取 | 使用病理 CLAM 工具时 |
| h5py | HDF5 文件处理 | 使用病理 CLAM 工具时 |
| timm | 预训练模型 | 使用病理 CLAM 工具时 |
| pyyaml | YAML 配置 | 使用病理 CLAM 工具时 |

### 可选依赖

| 包名 | 用途 | 说明 |
|------|------|------|
| openai | OpenAI API | 联网搜索工具默认使用 |
| torch | PyTorch | GPU 加速推荐 |
| cuda toolkit | CUDA 支持 | GPU 加速必需 |

---

## 常见问题

### 1. RAG 工具返回"未运行 ingestion"

确保已执行 `python -m src.rag.ingest` 将 `data/guidelines/*.pdf` 入库到 `chroma_db/` 目录。该命令会解析 PDF 文件、进行分块和向量化存储。如果已执行过 ingestion，检查 `chroma_db/` 目录是否存在且包含数据文件。

### 2. 病史解析结果不完整

病史解析结果中包含 `parsing_method` 和 `confidence_score` 字段。如果 `parsing_method` 为 `rule_based` 且 `confidence_score` 较低（<0.7），说明使用的是规则解析，信息可能不完整。确保 LLM 配置正确（API Key 和模型可用），复杂病史会自动调用 LLM 增强解析。

### 3. CT/MRI 分期工具漏检转移灶

分期工具基于文本关键词检测，可能因报告表述方式不同而漏检。工具返回的 `metastasis_sites` 字段会列出检测到的转移部位，如果为空但影像报告描述可疑，建议人工复核或使用联网搜索工具查询该病例的具体情况。

### 4. 联网搜索工具无结果

确认使用的是支持网络搜索的模型（如 gpt-4o-search-preview）。某些专业话题可能确实没有公开资料，工具会明确返回"没有找到相关资料"。可以尝试简化搜索关键词或使用更通用的表述。

### 5. 工具调用参数错误

RAG 工具和联网搜索工具定义了严格的输入模式（args_schema），LLM 需要根据 description 生成正确参数。常见错误包括：必填参数缺失、参数值超出约束范围（如 top_k > 15）、参数格式错误。查看工具的 description 和 args_schema 定义可了解正确用法。

### 6. 深度学习工具无法加载模型

肿瘤筛选、肿瘤定位、影像组学工具需要 torch、ultralytics、pyradiomics 等依赖。确保已安装相应依赖：
```bash
pip install torch torchvision
pip install ultralytics
pip install pyradiomics SimpleITK
pip install scikit-learn
pip install opencv-python
```

模型文件需要在正确的路径：
- 肿瘤筛选模型：`src/tools/tool/Tumor_Detection/best.pt`
- 肿瘤定位模型：`src/tools/tool/Tumor_Localization/checkpoint_epoch_last.pth`

### 7. GPU 不可用时性能较慢

深度学习工具默认优先使用 GPU 加速。如果检测到 CUDA 会自动使用 GPU，否则使用 CPU。CPU 模式下推理速度较慢，建议在支持 CUDA 的环境中运行。

### 8. 数据库工具返回空结果

确保虚拟数据库已正确初始化，数据文件存在于 `data/Case Database/` 目录。数据库工具基于静态数据集实现，检查数据集格式是否符合预期。

### 9. 病理 CLAM 工具无法读取切片

病理 CLAM 工具依赖 openslide-python 读取全切片图像。确保已正确安装：

**Windows:**
1. 从 https://openslide.org/download/ 下载 Windows 二进制文件
2. 解压并将 bin 目录添加到 PATH
3. `pip install openslide-python`

**Linux:**
```bash
sudo apt-get install openslide-tools
pip install openslide-python
```

**macOS:**
```bash
brew install openslide
pip install openslide-python
```

其他依赖：
```bash
pip install h5py timm pyyaml pandas
```

模型文件路径：`src/tools/tool/Pathological_Slide_Classification/CLAM_Tool/s_4_checkpoint.pt`

### 10. 病理分析耗时较长

全切片图像文件通常较大（数 GB），处理时间取决于：
- 切片文件大小
- 计算设备（GPU 明显快于 CPU）
- 内存/显存容量

建议在 GPU 环境运行，处理时间通常为 5-15 分钟/切片。

---

## 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| 1.0.0 | 2024-12 | 初始版本发布，基础工具和临床工具 |
| 1.1.0 | 2024-12 | 增加 RAG 工具集，支持混合检索 |
| 1.2.0 | 2024-12 | 增加联网搜索工具集 |
| 1.3.0 | 2024-12 | 增强临床工具：病史解析支持 LLM 增强、CT/MRI 分期工具优化 |
| 1.4.0 | 2024-12 | RAG 工具增加双通道输出格式、引用锚点功能 |
| 2.0.0 | 2026-01 | 增加数据库工具集，支持虚拟病例查询 |
| 2.1.0 | 2026-01 | 增加肿瘤筛选工具集（基于 YOLO） |
| 2.2.0 | 2026-01 | 增加肿瘤定位工具集（基于 U-Net） |
| 2.3.0 | 2026-01 | 增加影像组学工具集（U-Net + PyRadiomics + LASSO） |
| 2.4.0 | 2026-01 | 增加卡片格式化工具，支持前端数据展示 |
| 2.5.0 | 2026-01 | 增加病理 CLAM 工具集（全切片分类 + 热力图 + Top-K 区域提取） |
| 3.0.0 | 2026-01 | 代码重构：移除不存在的 intelligent_query_tool.py，文档与实际代码对齐 |

---

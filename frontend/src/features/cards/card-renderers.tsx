import { useState, type ReactNode } from "react";

import type { JsonObject, JsonValue } from "../../app/api/types";

export type CardPromptHandler = (prompt: string, context?: Record<string, unknown>) => void;

type CardRendererContext = {
  cardType: string;
  payload: JsonObject;
  onPromptRequest?: CardPromptHandler;
  isInteractive?: boolean;
};

const TRIAGE_RISK_LABELS: Record<string, string> = {
  low: "低风险",
  medium: "中风险",
  high: "高风险",
};

const TRIAGE_DISPOSITION_LABELS: Record<string, string> = {
  observe: "观察随访",
  routine_gi_clinic: "常规消化门诊",
  urgent_gi_clinic: "尽快消化门诊",
  emergency: "急诊就医",
  enter_crc_flow: "进入 CRC 临床评估",
};

const EMPTY_IMAGING_PREVIEW_MESSAGE = "暂无影像预览。";
const EMPTY_TUMOR_PREVIEW_MESSAGE = "暂无阳性样本预览。";
const EMPTY_PATHOLOGY_SLIDE_PREVIEW_MESSAGE = "暂无切片预览。";
const EMPTY_RADIOMICS_PREVIEW_MESSAGE = "暂无分析样本预览。";

function asObject(value: unknown): JsonObject | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as JsonObject;
}

function asObjectArray(value: unknown): JsonObject[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.map((item) => asObject(item)).filter((item): item is JsonObject => item !== null);
}

function asString(value: unknown): string | null {
  if (typeof value === "string" && value.trim()) {
    return value.trim();
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return null;
}

function asNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return null;
}

function readValue(source: JsonObject | null, key: string): JsonValue | unknown {
  return source?.[key];
}

function booleanLabel(value: unknown): string | null {
  if (value === true) {
    return "是";
  }
  if (value === false) {
    return "否";
  }
  return null;
}

function fieldText(value: unknown, options?: { booleanAsLabel?: boolean; suffix?: string }): string | null {
  const fieldObject = asObject(value);
  if (fieldObject) {
    const fieldMeta = asObject(readValue(fieldObject, "field_meta"));
    const metaDisplay = asString(readValue(fieldMeta, "display"));
    if (metaDisplay) {
      return metaDisplay;
    }

    const directDisplay = asString(readValue(fieldObject, "display"));
    if (directDisplay) {
      return directDisplay;
    }

    const candidateKeys = ["value", "raw_value", "raw", "canonical_value", "actual_value", "data"];
    for (const key of candidateKeys) {
      if (Object.prototype.hasOwnProperty.call(fieldObject, key)) {
        const candidate = readValue(fieldObject, key);
        if (candidate !== undefined) {
          return fieldText(candidate, options);
        }
      }
    }

    return null;
  }

  if (typeof value === "boolean") {
    return options?.booleanAsLabel === false ? String(value) : booleanLabel(value);
  }

  const text = asString(value);
  if (!text) {
    return null;
  }

  if (options?.suffix && !text.endsWith(options.suffix)) {
    return `${text}${options.suffix}`;
  }

  return text;
}

function fieldMetaDisplay(payload: JsonObject, section: string, field: string): string | null {
  const fieldMetaRoot = asObject(readValue(payload, "field_meta"));
  const sectionMeta = asObject(readValue(fieldMetaRoot, section));
  const fieldMeta = asObject(readValue(sectionMeta, field));
  const display = asString(readValue(fieldMeta, "display"));
  if (display) {
    return display;
  }

  const dottedKeyMeta = asObject(readValue(fieldMetaRoot, `${section}.${field}`));
  const dottedDisplay = asString(readValue(dottedKeyMeta, "display"));
  if (dottedDisplay) {
    return dottedDisplay;
  }

  const dottedKeyValue = asString(readValue(fieldMetaRoot, `${section}.${field}`));
  return dottedKeyValue;
}

function patientCardFieldText(
  payload: JsonObject,
  section: string,
  field: string,
  value: unknown,
  preferFieldMeta: boolean,
  options?: { booleanAsLabel?: boolean; suffix?: string },
): string | null {
  const display = preferFieldMeta ? fieldMetaDisplay(payload, section, field) : null;
  if (display) {
    return display;
  }
  return fieldText(value, options);
}

function triageRiskLabel(value: unknown): string | null {
  const raw = asString(value);
  if (!raw) {
    return null;
  }
  return TRIAGE_RISK_LABELS[raw] ?? raw;
}

function triageDispositionLabel(value: unknown): string | null {
  const raw = asString(value);
  if (!raw) {
    return null;
  }
  return TRIAGE_DISPOSITION_LABELS[raw] ?? raw;
}

function renderMetaItems(items: Array<{ label: string; value: string | number | null | undefined }>) {
  const visibleItems = items.filter((item) => item.value !== null && item.value !== undefined && String(item.value) !== "");
  if (visibleItems.length === 0) {
    return null;
  }

  return (
    <dl className="workspace-definition-list workspace-definition-list-compact">
      {visibleItems.map((item) => (
        <div key={item.label}>
          <dt>{item.label}</dt>
          <dd>{String(item.value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function renderPromptButtons(prompts: string[], onPromptRequest?: CardPromptHandler, labels?: string[]) {
  if (!onPromptRequest || prompts.length === 0) {
    return null;
  }

  return (
    <div className="workspace-action-row">
      {prompts.map((prompt, index) => (
        <button
          key={prompt}
          type="button"
          className="workspace-secondary-button workspace-action-button"
          onClick={() => onPromptRequest(prompt)}
        >
          {labels?.[index] ?? prompt}
        </button>
      ))}
    </div>
  );
}

function renderDisclosure(title: string, payload: JsonObject) {
  return (
    <details className="workspace-card-disclosure">
      <summary>{title}</summary>
      <pre>{JSON.stringify(payload, null, 2)}</pre>
    </details>
  );
}

function previewImageSrc(image: JsonObject): string | null {
  const base64 = asString(image.image_base64);
  if (base64) {
    return `data:image/png;base64,${base64}`;
  }
  return asString(image.image_url);
}

function previewImagesFromPayload(payload: JsonObject): JsonObject[] {
  const data = asObject(payload.data) ?? payload;
  let images = asObjectArray(readValue(data, "images"));
  if (images.length === 0) {
    images = asObjectArray(readValue(data, "sample_images_with_tumor"));
  }
  if (images.length === 0) {
    images = asObjectArray(readValue(data, "analyzed_images"));
  }
  return images.filter((item) => Boolean(previewImageSrc(item)));
}

function ImagingPreviewGallery({ images }: { images: JsonObject[] }) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const selectedImage = images[selectedIndex] ?? images[0];
  const selectedSource = selectedImage ? previewImageSrc(selectedImage) : null;

  if (!selectedSource) {
    return null;
  }

  return (
    <div className="workspace-card-section">
      <strong>影像预览</strong>
      <div className="workspace-image-preview-frame">
        <img
          src={selectedSource}
          alt={asString(selectedImage.image_name) ?? "影像预览"}
          className="workspace-image-preview-main"
        />
      </div>
      <div className="workspace-image-strip">
        {images.map((image, index) => {
          const imageSource = previewImageSrc(image);
          if (!imageSource) {
            return null;
          }

          const imageName = asString(image.image_name) ?? `影像 ${index + 1}`;
          const active = index === selectedIndex;
          return (
            <button
              key={`${imageName}-${index}`}
              type="button"
              className={active ? "workspace-image-chip workspace-image-chip-active" : "workspace-image-chip"}
              onClick={() => setSelectedIndex(index)}
            >
              <img src={imageSource} alt={imageName} className="workspace-image-chip-thumb" />
              <span>{imageName}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function renderPreviewSection(images: JsonObject[], emptyMessage: string) {
  return images.length > 0 ? (
    <ImagingPreviewGallery images={images.slice(0, 8)} />
  ) : (
    <div className="workspace-card-section">
      <strong>影像预览</strong>
      <p className="workspace-copy workspace-copy-tight">{emptyMessage}</p>
    </div>
  );
}

function cardSummary(payload: JsonObject): string | null {
  const candidates = ["text_summary", "summary", "recommendation", "assessment", "note", "details"];
  for (const key of candidates) {
    const value = asString(payload[key]);
    if (value) {
      return value;
    }
  }
  return null;
}

function renderValueList(value: unknown) {
  if (!Array.isArray(value)) {
    return null;
  }

  const items = value.map((item) => asString(item)).filter((item): item is string => Boolean(item));
  if (items.length === 0) {
    return null;
  }

  return (
    <ul className="workspace-list">
      {items.map((item) => (
        <li key={item} className="workspace-list-item">
          {item}
        </li>
      ))}
    </ul>
  );
}

function renderMedicalCard(payload: JsonObject, onPromptRequest?: CardPromptHandler) {
  const data = asObject(payload.data);
  const patientSummary = asObject(readValue(data, "patient_summary"));
  const diagnosisBlock = asObject(readValue(data, "diagnosis_block"));
  const stagingBlock = asObject(readValue(data, "staging_block"));
  const drafts = asObjectArray(readValue(data, "treatment_draft"));

  const diagnosis = asString(readValue(diagnosisBlock, "confirmed"));
  const stage = asString(readValue(stagingBlock, "clinical_stage"));
  const risk = asString(readValue(stagingBlock, "risk_status"));

  const quickSuggestions: string[] = [];
  if (diagnosis) {
    quickSuggestions.push(`${diagnosis}的标准一线治疗方案是什么？`);
  }
  if (stage?.includes("IV")) {
    quickSuggestions.push("针对晚期结直肠癌有哪些靶向药物选择？");
  } else if (stage) {
    quickSuggestions.push("这个分期术后是否需要辅助化疗？");
  }
  quickSuggestions.push("帮我解读报告中的关键异常指标。");

  return (
    <>
      <div className="workspace-card-section">
        <p className="workspace-card-kicker">医疗总览</p>
        <p className="workspace-copy workspace-copy-tight">
          {cardSummary(payload) ?? asString(readValue(patientSummary, "chief_complaint")) ?? "暂无医疗总览摘要。"}
        </p>
        {renderMetaItems([
          { label: "诊断", value: diagnosis },
          { label: "分期", value: stage },
          { label: "风险", value: risk },
          { label: "年龄", value: asString(readValue(patientSummary, "age")) },
          { label: "性别", value: asString(readValue(patientSummary, "gender")) },
        ])}
      </div>
      {drafts.length > 0 ? (
        <div className="workspace-card-section">
          <strong>治疗草案</strong>
          <ul className="workspace-list">
            {drafts.map((item, index) => (
              <li key={asString(item.name) ?? `draft-${index}`} className="workspace-list-item">
                <strong>{asString(item.name) ?? `方案 ${index + 1}`}</strong>
                <p className="workspace-copy workspace-copy-tight">
                  {asString(item.details) ?? asString(item.status) ?? "暂无细节。"}
                </p>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {renderPromptButtons(quickSuggestions, onPromptRequest)}
      {renderDisclosure("查看原始数据", payload)}
    </>
  );
}

function renderPatientCard(payload: JsonObject, onPromptRequest?: CardPromptHandler) {
  const data = asObject(payload.data);
  const patientInfo = asObject(readValue(data, "patient_info"));
  const diagnosisBlock = asObject(readValue(data, "diagnosis_block"));
  const stagingBlock = asObject(readValue(data, "staging_block"));
  const historyBlock = asObject(readValue(data, "history_block"));
  const cardMeta = asObject(readValue(payload, "card_meta"));
  const sourceMode = asString(readValue(cardMeta, "source_mode"));
  const isSelfReport = sourceMode === "patient_self_report";
  const patientId = asString(payload.patient_id) ?? asString(readValue(data, "patient_id")) ?? "N/A";
  const rawRiskFactors = readValue(historyBlock, "risk_factors");
  const riskFactors = Array.isArray(rawRiskFactors)
    ? rawRiskFactors.map((item) => asString(item)).filter((item): item is string => Boolean(item))
    : [];
  const riskFactorsDisplay = isSelfReport ? fieldMetaDisplay(payload, "history_block", "risk_factors") : null;

  const prompts = [
    `为病人 ${patientId} 生成治疗方案`,
    `查询病人 #${patientId} 的影像资料`,
    `为病人 ${patientId} 撰写当日病程记录`,
  ];
  const labels = ["生成治疗方案", "查询影像资料", "撰写病程记录"];

  const patientInfoItems = [
    { label: "性别", value: patientCardFieldText(payload, "patient_info", "gender", readValue(patientInfo, "gender"), isSelfReport) },
    { label: "年龄", value: patientCardFieldText(payload, "patient_info", "age", readValue(patientInfo, "age"), isSelfReport, { suffix: "岁" }) },
    { label: "ECOG", value: patientCardFieldText(payload, "patient_info", "ecog", readValue(patientInfo, "ecog"), isSelfReport) },
    { label: "CEA", value: patientCardFieldText(payload, "patient_info", "cea", readValue(patientInfo, "cea"), isSelfReport) },
  ];
  const diagnosisItems = [
    { label: "确诊", value: patientCardFieldText(payload, "diagnosis_block", "confirmed", readValue(diagnosisBlock, "confirmed"), isSelfReport) },
    { label: "原发部位", value: patientCardFieldText(payload, "diagnosis_block", "primary_site", readValue(diagnosisBlock, "primary_site"), isSelfReport) },
    { label: "MMR", value: patientCardFieldText(payload, "diagnosis_block", "mmr_status", readValue(diagnosisBlock, "mmr_status"), isSelfReport) },
    { label: "临床分期", value: patientCardFieldText(payload, "staging_block", "clinical_stage", readValue(stagingBlock, "clinical_stage"), isSelfReport) },
    { label: "cT", value: patientCardFieldText(payload, "staging_block", "ct_stage", readValue(stagingBlock, "ct_stage"), isSelfReport) },
    { label: "cN", value: patientCardFieldText(payload, "staging_block", "cn_stage", readValue(stagingBlock, "cn_stage"), isSelfReport) },
    { label: "cM", value: patientCardFieldText(payload, "staging_block", "cm_stage", readValue(stagingBlock, "cm_stage"), isSelfReport) },
  ];
  const historyItems = [
    { label: "主诉", value: patientCardFieldText(payload, "history_block", "chief_complaint", readValue(historyBlock, "chief_complaint"), isSelfReport) },
    { label: "症状持续时间", value: patientCardFieldText(payload, "history_block", "symptom_duration", readValue(historyBlock, "symptom_duration"), isSelfReport) },
    { label: "家族史", value: patientCardFieldText(payload, "history_block", "family_history", readValue(historyBlock, "family_history"), isSelfReport, { booleanAsLabel: true }) },
    { label: "家族史详情", value: patientCardFieldText(payload, "history_block", "family_history_details", readValue(historyBlock, "family_history_details"), isSelfReport) },
    { label: "病理活检确认", value: patientCardFieldText(payload, "history_block", "biopsy_confirmed", readValue(historyBlock, "biopsy_confirmed"), isSelfReport, { booleanAsLabel: true }) },
    { label: "活检详情", value: patientCardFieldText(payload, "history_block", "biopsy_details", readValue(historyBlock, "biopsy_details"), isSelfReport) },
    { label: "危险因素", value: riskFactorsDisplay ?? (riskFactors.length > 0 ? riskFactors.join("、") : null) },
  ];

  return (
    <>
      <div className="workspace-card-section">
        <p className="workspace-card-kicker">患者画像</p>
        <strong className="workspace-card-heading">{`患者 #${patientId}`}</strong>
        {renderMetaItems(patientInfoItems)}
      </div>
      {isSelfReport || diagnosisBlock ? (
        <div className="workspace-card-section">
          <strong>诊断信息</strong>
          {renderMetaItems(diagnosisItems)}
        </div>
      ) : null}
      {isSelfReport || historyBlock ? (
        <div className="workspace-card-section">
          <strong>基础病史</strong>
          {renderMetaItems(historyItems)}
        </div>
      ) : null}
      {!isSelfReport ? renderPromptButtons(prompts, onPromptRequest, labels) : null}
      {renderDisclosure("查看原始数据", payload)}
    </>
  );
}

function renderImagingVisualCard(payload: JsonObject) {
  const data = asObject(payload.data) ?? payload;
  const folderName = asString(readValue(data, "folder_name")) ?? asString(readValue(data, "patient_id")) ?? "未知";
  const totalImages = asNumber(readValue(data, "total_images")) ?? asObjectArray(readValue(data, "images")).length;
  const previewImages = previewImagesFromPayload(payload);
  const previewCount = previewImages.length;
  const summary = cardSummary(payload) ?? `影像样本：患者 ${folderName}，共 ${totalImages} 张影像`;

  return (
    <>
      <div className="workspace-card-section">
        <p className="workspace-card-kicker">影像样本</p>
        <strong className="workspace-card-heading">{`患者 ${folderName}`}</strong>
        <p className="workspace-copy workspace-copy-tight">{summary}</p>
        {renderMetaItems([
          { label: "影像总数", value: totalImages > 0 ? `共 ${totalImages} 张影像` : null },
          { label: "预览样本", value: previewCount > 0 ? `${previewCount} 张` : null },
          { label: "来源目录", value: folderName },
        ])}
      </div>
      {renderPreviewSection(previewImages, EMPTY_IMAGING_PREVIEW_MESSAGE)}
      {renderDisclosure("查看原始数据", payload)}
    </>
  );
}

function renderPathologySlideVisualCard(payload: JsonObject) {
  const data = asObject(payload.data);
  const folderName = asString(readValue(data, "folder_name")) ?? "未知";
  const totalImages = asNumber(readValue(data, "total_images")) ?? asObjectArray(readValue(data, "images")).length;
  const previewSize = asNumber(readValue(data, "preview_size"));
  const previewImages = previewImagesFromPayload(payload);

  return (
    <>
      <div className="workspace-card-section">
        <p className="workspace-card-kicker">病理切片</p>
        <strong className="workspace-card-heading">{`患者 ${folderName}`}</strong>
        <p className="workspace-copy workspace-copy-tight">
          {cardSummary(payload) ?? "已整理病理切片预览，可按需继续查看。"}
        </p>
        {renderMetaItems([
          { label: "切片总数", value: totalImages > 0 ? `${totalImages} 张` : null },
          { label: "预览样本", value: previewImages.length > 0 ? `${previewImages.length} 张` : null },
          { label: "预览尺寸", value: previewSize ? `${previewSize}px` : null },
          { label: "来源目录", value: folderName },
        ])}
      </div>
      {renderPreviewSection(previewImages, EMPTY_PATHOLOGY_SLIDE_PREVIEW_MESSAGE)}
      {renderDisclosure("查看原始数据", payload)}
    </>
  );
}

function renderTumorDetectionVisualCard(payload: JsonObject, onPromptRequest?: CardPromptHandler) {
  const data = asObject(payload.data);
  const previewImages = previewImagesFromPayload(payload);
  const patientId =
    asString(payload.patient_id) ??
    asString(readValue(data, "patient_id")) ??
    asString(readValue(data, "folder_name")) ??
    "N/A";

  const prompts = [
    `查看患者 ${patientId} 的肿瘤检测原始数据`,
    `生成患者 ${patientId} 的肿瘤检测总结`,
  ];
  const labels = ["查看原始数据", "生成检测总结"];

  return (
    <>
      <div className="workspace-card-section">
        <p className="workspace-card-kicker">肿瘤检测</p>
        <strong className="workspace-card-heading">{`患者 ${patientId}`}</strong>
        <p className="workspace-copy workspace-copy-tight">
          {cardSummary(payload) ?? "已完成影像肿瘤检测，可继续查看详细评估。"}
        </p>
        {renderMetaItems([
          { label: "影像总数", value: asNumber(readValue(data, "total_images")) },
          { label: "检出阳性", value: asNumber(readValue(data, "images_with_tumor")) },
          { label: "阳性比例", value: asString(readValue(data, "tumor_detection_rate")) },
          { label: "最高置信度", value: asString(readValue(data, "max_confidence")) ?? asNumber(readValue(data, "max_confidence")) },
        ])}
      </div>
      {renderPreviewSection(previewImages, EMPTY_TUMOR_PREVIEW_MESSAGE)}
      {renderPromptButtons(prompts, onPromptRequest, labels)}
      {renderDisclosure("查看原始数据", payload)}
    </>
  );
}

function renderPathologyCard(payload: JsonObject) {
  const data = asObject(payload.data);
  const analysisMode = asString(readValue(data, "analysis_mode"));
  const patientId = asString(readValue(data, "patient_id")) ?? "N/A";
  const results = asObjectArray(readValue(data, "results"));

  return (
    <>
      <div className="workspace-card-section">
        <p className="workspace-card-kicker">病理报告</p>
        <strong className="workspace-card-heading">{`患者 ${patientId}`}</strong>
        <p className="workspace-copy workspace-copy-tight">
          {cardSummary(payload) ?? "病理切片分析结果已生成。"}
        </p>
        {renderMetaItems([
          { label: "分析模式", value: analysisMode },
          { label: "预测结果", value: asString(readValue(data, "prediction")) ?? asString(readValue(data, "overall_diagnosis")) },
          { label: "肿瘤概率", value: asString(readValue(data, "tumor_probability")) ?? asNumber(readValue(data, "tumor_probability")) },
          { label: "置信度", value: asString(readValue(data, "confidence")) ?? asNumber(readValue(data, "confidence")) },
          { label: "已分析切片", value: results.length > 0 ? `${results.length} 张` : asNumber(readValue(data, "slides_analyzed")) },
        ])}
      </div>
      {renderDisclosure("查看原始数据", payload)}
    </>
  );
}

function renderRadiomicsVisualCard(payload: JsonObject) {
  const data = asObject(payload.data);
  const patientId = asString(payload.patient_id) ?? asString(readValue(data, "patient_id")) ?? "N/A";
  const topFeatures = asObjectArray(readValue(data, "top_features"));
  const previewImages = previewImagesFromPayload(payload);

  return (
    <>
      <div className="workspace-card-section">
        <p className="workspace-card-kicker">影像组学报告</p>
        <strong className="workspace-card-heading">{`患者 ${patientId}`}</strong>
        <p className="workspace-copy workspace-copy-tight">
          {cardSummary(payload) ?? "已完成影像组学分析。"}
        </p>
        {renderMetaItems([
          { label: "分析模式", value: asString(readValue(data, "analysis_mode")) },
          { label: "影像总数", value: asNumber(readValue(data, "total_images")) },
          { label: "检出阳性", value: asNumber(readValue(data, "images_with_tumor")) },
          { label: "已分析影像", value: asNumber(readValue(data, "analyzed_images_count")) },
          { label: "预览样本", value: previewImages.length > 0 ? `${previewImages.length} 张` : null },
          { label: "Top 特征", value: topFeatures.length > 0 ? `${topFeatures.length} 项` : null },
        ])}
      </div>
      {renderPreviewSection(previewImages, EMPTY_RADIOMICS_PREVIEW_MESSAGE)}
      {renderDisclosure("查看原始数据", payload)}
    </>
  );
}

function renderDecisionCard(payload: JsonObject) {
  const data = asObject(payload.data) ?? payload;
  const summary =
    asString(data.patient_summary) ?? asString(data.summary) ?? cardSummary(data) ?? "已生成治疗决策摘要。";
  const plans = asObjectArray(data.treatment_plan);
  const followUp = asObjectArray(data.follow_up_plan);
  const goals = Array.isArray(data.treatment_goals) ? data.treatment_goals : [];
  const considerations = Array.isArray(data.key_considerations) ? data.key_considerations : [];

  return (
    <>
      <div className="workspace-card-section">
        <p className="workspace-card-kicker">治疗决策</p>
        <p className="workspace-copy workspace-copy-tight">{summary}</p>
        {renderMetaItems([{ label: "分期结论", value: asString(data.staging_conclusion) ?? asString(data.staging) }])}
      </div>
      {goals.length > 0 ? (
        <div className="workspace-card-section">
          <strong>治疗目标</strong>
          <ul className="workspace-list">
            {goals.map((goal, index) => (
              <li key={`goal-${index}`} className="workspace-list-item">
                {String(goal)}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {plans.length > 0 ? (
        <div className="workspace-card-section">
          <strong>治疗计划</strong>
          <ul className="workspace-list">
            {plans.map((plan, index) => (
              <li key={`plan-${index}`} className="workspace-list-item">
                <strong>{asString(plan.phase) ?? asString(plan.title) ?? asString(plan.name) ?? `阶段 ${index + 1}`}</strong>
                <p className="workspace-copy workspace-copy-tight">
                  {asString(plan.regimen) ?? asString(plan.content) ?? asString(plan.details) ?? "暂无说明。"}
                </p>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {followUp.length > 0 ? (
        <div className="workspace-card-section">
          <strong>随访计划</strong>
          <ul className="workspace-list">
            {followUp.map((item, index) => (
              <li key={`follow-${index}`} className="workspace-list-item">
                {asString(item.period) ?? asString(item.frequency) ?? asString(item.items) ?? JSON.stringify(item)}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {considerations.length > 0 ? (
        <div className="workspace-card-section">
          <strong>关键考虑点</strong>
          <ul className="workspace-list">
            {considerations.map((item, index) => (
              <li key={`consideration-${index}`} className="workspace-list-item">
                {String(item)}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {renderDisclosure("查看原始数据", payload)}
    </>
  );
}

function renderTriageCard(payload: JsonObject) {
  const symptomSnapshot = asObject(payload.symptom_snapshot);
  const summary = cardSummary(payload);
  const riskLevel = triageRiskLabel(payload.risk_level);
  const disposition = triageDispositionLabel(payload.disposition);
  const chiefSymptoms =
    asString(payload.chief_symptoms) ??
    asString(symptomSnapshot?.chief_symptoms) ??
    (Array.isArray(symptomSnapshot?.chief_symptoms)
      ? symptomSnapshot.chief_symptoms
          .map((item) => asString(item))
          .filter((item): item is string => Boolean(item))
          .join("、")
      : null);
  const suggestedTests = renderValueList(payload.suggested_tests);

  return (
    <>
      {summary ? <p className="workspace-copy workspace-copy-tight">{summary}</p> : null}
      {renderMetaItems([
        { label: "风险等级", value: riskLevel },
        { label: "建议去向", value: disposition },
        { label: "主诉症状", value: chiefSymptoms },
      ])}
      {suggestedTests ? (
        <div className="workspace-card-section">
          <strong>建议检查</strong>
          {suggestedTests}
        </div>
      ) : null}
      {renderDisclosure("查看原始数据", payload)}
    </>
  );
}

function renderGenericCard(payload: JsonObject) {
  const summary = cardSummary(payload);
  if (summary) {
    return (
      <>
        <p className="workspace-copy workspace-copy-tight">{summary}</p>
        {renderDisclosure("查看原始数据", payload)}
      </>
    );
  }
  return renderDisclosure("查看原始数据", payload);
}

export function cardTitle(cardType: string, payload: JsonObject): string {
  if (typeof payload.title === "string" && payload.title) {
    return payload.title;
  }

  const typeLabels: Record<string, string> = {
    medical_card: "医疗总览",
    patient_card: "患者画像",
    imaging_card: "影像样本",
    tumor_detection_card: "肿瘤检测",
    tumor_screening_result: "肿瘤筛查",
    pathology_card: "病理报告",
    pathology_slide_card: "病理切片",
    radiomics_report_card: "影像组学报告",
    decision_card: "治疗决策",
    triage_card: "门诊分诊",
    patient_summary: "患者摘要",
    tumor_board: "肿瘤讨论",
  };

  return typeLabels[cardType] ?? cardType.replace(/_/g, " ");
}

export function renderCardContent({ cardType, payload, onPromptRequest }: CardRendererContext): ReactNode {
  switch (cardType) {
    case "medical_card":
      return renderMedicalCard(payload, onPromptRequest);
    case "patient_card":
      return renderPatientCard(payload, onPromptRequest);
    case "imaging_card":
      return renderImagingVisualCard(payload);
    case "tumor_detection_card":
    case "tumor_screening_result":
      return renderTumorDetectionVisualCard(payload, onPromptRequest);
    case "pathology_card":
      return renderPathologyCard(payload);
    case "pathology_slide_card":
      return renderPathologySlideVisualCard(payload);
    case "radiomics_report_card":
      return renderRadiomicsVisualCard(payload);
    case "decision_card":
      return renderDecisionCard(payload);
    case "triage_card":
      return renderTriageCard(payload);
    default:
      return renderGenericCard(payload);
  }
}

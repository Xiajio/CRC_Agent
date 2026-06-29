import { useState, type ReactNode } from "react";

import type { JsonObject, JsonValue } from "../../app/api/types";
import { Button } from "../../components/ui";

export type CardPromptHandler = (prompt: string, context?: Record<string, unknown>) => void;

export type CardPatientContext = {
  registry_patient_id?: unknown;
  case_database_patient_id?: unknown;
};

type CardRendererContext = {
  cardType: string;
  payload: JsonObject;
  onPromptRequest?: CardPromptHandler;
  isInteractive?: boolean;
  patientContext?: CardPatientContext | null;
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

function asDecisionFollowUpItems(...values: unknown[]): string[] {
  for (const value of values) {
    if (!Array.isArray(value)) {
      continue;
    }

    const items = value
      .map((item) => {
        const directText = asString(item);
        if (directText) {
          return directText;
        }

        const objectItem = asObject(item);
        if (!objectItem) {
          return null;
        }

        return (
          asString(objectItem.period) ??
          asString(objectItem.frequency) ??
          asString(objectItem.items) ??
          asString(objectItem.title) ??
          asString(objectItem.content) ??
          JSON.stringify(objectItem)
        );
      })
      .filter((item): item is string => Boolean(item));

    if (items.length > 0) {
      return items;
    }
  }

  return [];
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

function confidenceRatio(value: unknown): number | null {
  if (typeof value === "number") {
    if (!Number.isFinite(value) || value < 0) {
      return null;
    }
    return value <= 1 ? value : value / 100;
  }

  if (typeof value !== "string") {
    return null;
  }

  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }

  const hasPercentSuffix = trimmed.endsWith("%");
  const numericText = hasPercentSuffix ? trimmed.slice(0, -1).trim() : trimmed;
  const parsed = Number(numericText);
  if (!Number.isFinite(parsed) || parsed < 0) {
    return null;
  }

  return hasPercentSuffix || parsed > 1 ? parsed / 100 : parsed;
}

function formatRatioAsPercent(ratio: number): string {
  const rounded = Math.round((ratio * 100 + Number.EPSILON) * 10) / 10;
  const formatted = Number.isInteger(rounded) ? rounded.toFixed(0) : rounded.toFixed(1);
  return `${formatted}%`;
}

function confidenceDisplay(value: unknown): string | null {
  const ratio = confidenceRatio(value);
  if (ratio !== null) {
    return formatRatioAsPercent(ratio);
  }

  return asString(value);
}

function needsManualReview(value: unknown): boolean {
  return value === true || (typeof value === "string" && value.trim().toLowerCase() === "true");
}

function isBelowConfidenceThreshold(confidenceValue: unknown, thresholdValue: unknown): boolean {
  const confidence = confidenceRatio(confidenceValue);
  const threshold = confidenceRatio(thresholdValue);
  return confidence !== null && threshold !== null && confidence < threshold;
}

function readValue(source: JsonObject | null, key: string): JsonValue | unknown {
  return source?.[key];
}

function readPatientContextValue(
  payload: JsonObject,
  data: JsonObject | null,
  key: keyof CardPatientContext,
  fallback?: CardPatientContext | null,
): unknown {
  const payloadValue = readValue(payload, key);
  if (payloadValue !== null && payloadValue !== undefined) {
    return payloadValue;
  }

  const dataValue = readValue(data, key);
  if (dataValue !== null && dataValue !== undefined) {
    return dataValue;
  }

  return fallback?.[key];
}

function patientPromptContext(
  payload: JsonObject,
  fallback?: CardPatientContext | null,
): Record<string, unknown> | undefined {
  const data = asObject(payload.data);
  const registryPatientId = asNumber(readPatientContextValue(payload, data, "registry_patient_id", fallback));
  const caseDatabasePatientId = asString(readPatientContextValue(payload, data, "case_database_patient_id", fallback));
  const context: Record<string, unknown> = {};

  if (registryPatientId !== null) {
    context.registry_patient_id = registryPatientId;
  }
  if (caseDatabasePatientId !== null) {
    context.case_database_patient_id = caseDatabasePatientId;
  }

  return Object.keys(context).length > 0 ? context : undefined;
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
    <dl className="clinical-definition-list clinical-definition-list-compact">
      {visibleItems.map((item) => (
        <div key={item.label} className="clinical-definition-item">
          <dt>{item.label}</dt>
          <dd>{String(item.value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function renderConfidenceReviewNotice(show: boolean) {
  if (!show) {
    return null;
  }

  return (
    <div className="clinical-confidence-alert" role="status">
      <span className="clinical-confidence-badge">需人工复核</span>
      <span>模型置信度低于阈值或已被系统标记，请复核原始影像、切片与模型输出。</span>
    </div>
  );
}

function renderPromptButtons(
  prompts: string[],
  onPromptRequest?: CardPromptHandler,
  labels?: string[],
  context?: Record<string, unknown>,
) {
  if (!onPromptRequest || prompts.length === 0) {
    return null;
  }

  return (
    <div className="clinical-action-row clinical-action-row-prompts">
      {prompts.map((prompt, index) => (
        <Button
          key={prompt}
          type="button"
          variant="secondary"
          size="sm"
          onClick={() => {
            if (context) {
              onPromptRequest(prompt, context);
              return;
            }
            onPromptRequest(prompt);
          }}
        >
          {labels?.[index] ?? prompt}
        </Button>
      ))}
    </div>
  );
}

function renderDisclosure(title: string, payload: JsonObject) {
  return (
    <details className="clinical-card-disclosure">
      <summary>{title}</summary>
      <pre>{JSON.stringify(payload, null, 2)}</pre>
    </details>
  );
}

function previewImageSrc(image: JsonObject): string | null {
  const base64 = asString(image.image_base64);
  if (base64) {
    const mimeType = asString(image.image_mime_type) ?? "image/png";
    return `data:${mimeType};base64,${base64}`;
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
    <div className="clinical-card-section">
      <strong>影像预览</strong>
      <div className="clinical-image-preview-frame">
        <img
          src={selectedSource}
          alt={asString(selectedImage.image_name) ?? "影像预览"}
          className="clinical-image-preview-main"
        />
      </div>
      <div className="clinical-image-strip">
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
              className={active ? "clinical-image-chip clinical-image-chip-active" : "clinical-image-chip"}
              onClick={() => setSelectedIndex(index)}
            >
              <img src={imageSource} alt={imageName} className="clinical-image-chip-thumb" />
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
    <div className="clinical-card-section">
      <strong>影像预览</strong>
      <p className="clinical-copy clinical-copy-tight">{emptyMessage}</p>
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
    <ul className="clinical-list">
      {items.map((item) => (
        <li key={item} className="clinical-list-item">
          {item}
        </li>
      ))}
    </ul>
  );
}

function renderMedicalCard(
  payload: JsonObject,
  onPromptRequest?: CardPromptHandler,
  patientContext?: CardPatientContext | null,
) {
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
  const context = patientPromptContext(payload, patientContext);

  return (
    <>
      <div className="clinical-card-section">
        <p className="clinical-card-kicker">医疗总览</p>
        <p className="clinical-copy clinical-copy-tight">
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
        <div className="clinical-card-section">
          <strong>治疗草案</strong>
          <ul className="clinical-list">
            {drafts.map((item, index) => (
              <li key={asString(item.name) ?? `draft-${index}`} className="clinical-list-item">
                <strong>{asString(item.name) ?? `方案 ${index + 1}`}</strong>
                <p className="clinical-copy clinical-copy-tight">
                  {asString(item.details) ?? asString(item.status) ?? "暂无细节。"}
                </p>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {renderPromptButtons(quickSuggestions, onPromptRequest, undefined, context)}
      {renderDisclosure("查看原始数据", payload)}
    </>
  );
}

function renderPatientCard(
  payload: JsonObject,
  onPromptRequest?: CardPromptHandler,
  patientContext?: CardPatientContext | null,
) {
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
  const context = patientPromptContext(payload, patientContext);

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
    { label: "症状归类", value: patientCardFieldText(payload, "history_block", "symptom_focus", readValue(historyBlock, "symptom_focus"), isSelfReport) },
    { label: "症状持续时间", value: patientCardFieldText(payload, "history_block", "symptom_duration", readValue(historyBlock, "symptom_duration"), isSelfReport) },
    { label: "家族史", value: patientCardFieldText(payload, "history_block", "family_history", readValue(historyBlock, "family_history"), isSelfReport, { booleanAsLabel: true }) },
    { label: "家族史详情", value: patientCardFieldText(payload, "history_block", "family_history_details", readValue(historyBlock, "family_history_details"), isSelfReport) },
    { label: "病理活检确认", value: patientCardFieldText(payload, "history_block", "biopsy_confirmed", readValue(historyBlock, "biopsy_confirmed"), isSelfReport, { booleanAsLabel: true }) },
    { label: "活检详情", value: patientCardFieldText(payload, "history_block", "biopsy_details", readValue(historyBlock, "biopsy_details"), isSelfReport) },
    { label: "危险因素", value: riskFactorsDisplay ?? (riskFactors.length > 0 ? riskFactors.join("、") : null) },
  ];

  return (
    <div className="clinical-card-stack">
      <div className="clinical-card-section clinical-card-section-stack">
        <p className="clinical-card-kicker clinical-card-kicker-primary">患者画像</p>
        <strong className="clinical-card-heading clinical-card-heading-offset">{`患者 #${patientId}`}</strong>
        {renderMetaItems(patientInfoItems)}
      </div>
      {isSelfReport || diagnosisBlock ? (
        <div className="clinical-card-section clinical-card-section-bordered">
          <strong className="clinical-card-section-title">诊断信息</strong>
          {renderMetaItems(diagnosisItems)}
        </div>
      ) : null}
      {isSelfReport || historyBlock ? (
        <div className="clinical-card-section clinical-card-section-bordered">
          <strong className="clinical-card-section-title">基础病史</strong>
          {renderMetaItems(historyItems)}
        </div>
      ) : null}
      {!isSelfReport ? renderPromptButtons(prompts, onPromptRequest, labels, context) : null}
      {renderDisclosure("查看原始数据", payload)}
    </div>
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
      <div className="clinical-card-section">
        <p className="clinical-card-kicker">影像样本</p>
        <strong className="clinical-card-heading">{`患者 ${folderName}`}</strong>
        <p className="clinical-copy clinical-copy-tight">{summary}</p>
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
      <div className="clinical-card-section">
        <p className="clinical-card-kicker">病理切片</p>
        <strong className="clinical-card-heading">{`患者 ${folderName}`}</strong>
        <p className="clinical-copy clinical-copy-tight">
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

function renderTumorDetectionVisualCard(
  payload: JsonObject,
  onPromptRequest?: CardPromptHandler,
  patientContext?: CardPatientContext | null,
) {
  const data = asObject(payload.data);
  const previewImages = previewImagesFromPayload(payload);
  const patientId =
    asString(payload.patient_id) ??
    asString(readValue(data, "patient_id")) ??
    asString(readValue(data, "folder_name")) ??
    "N/A";
  const maxConfidenceValue = readValue(data, "max_confidence");
  const confidenceThresholdValue = readValue(data, "confidence_threshold");
  const showReviewNotice =
    needsManualReview(readValue(data, "needs_review")) ||
    isBelowConfidenceThreshold(maxConfidenceValue, confidenceThresholdValue);

  const prompts = [
    `查看患者 ${patientId} 的肿瘤检测原始数据`,
    `生成患者 ${patientId} 的肿瘤检测总结`,
  ];
  const labels = ["查看原始数据", "生成检测总结"];
  const context = patientPromptContext(payload, patientContext);

  return (
    <>
      <div className="clinical-card-section">
        <p className="clinical-card-kicker">肿瘤检测</p>
        <strong className="clinical-card-heading">{`患者 ${patientId}`}</strong>
        <p className="clinical-copy clinical-copy-tight">
          {cardSummary(payload) ?? "已完成影像肿瘤检测，可继续查看详细评估。"}
        </p>
        {renderMetaItems([
          { label: "影像总数", value: asNumber(readValue(data, "total_images")) },
          { label: "检出阳性", value: asNumber(readValue(data, "images_with_tumor")) },
          { label: "阳性比例", value: asString(readValue(data, "tumor_detection_rate")) },
          { label: "最高置信度", value: confidenceDisplay(maxConfidenceValue) },
          { label: "置信度阈值", value: confidenceDisplay(confidenceThresholdValue) },
        ])}
        {renderConfidenceReviewNotice(showReviewNotice)}
      </div>
      {renderPreviewSection(previewImages, EMPTY_TUMOR_PREVIEW_MESSAGE)}
      {renderPromptButtons(prompts, onPromptRequest, labels, context)}
      {renderDisclosure("查看原始数据", payload)}
    </>
  );
}

function renderPathologyCard(payload: JsonObject) {
  const data = asObject(payload.data);
  const analysisMode = asString(readValue(data, "analysis_mode"));
  const patientId = asString(readValue(data, "patient_id")) ?? "N/A";
  const results = asObjectArray(readValue(data, "results"));
  const tumorProbabilityValue = readValue(data, "tumor_probability");
  const confidenceValue = readValue(data, "confidence");
  const confidenceThresholdValue = readValue(data, "confidence_threshold");
  const showReviewNotice =
    needsManualReview(readValue(data, "needs_review")) ||
    isBelowConfidenceThreshold(confidenceValue, confidenceThresholdValue);

  return (
    <>
      <div className="clinical-card-section">
        <p className="clinical-card-kicker">病理报告</p>
        <strong className="clinical-card-heading">{`患者 ${patientId}`}</strong>
        <p className="clinical-copy clinical-copy-tight">
          {cardSummary(payload) ?? "病理切片分析结果已生成。"}
        </p>
        {renderMetaItems([
          { label: "分析模式", value: analysisMode },
          { label: "预测结果", value: asString(readValue(data, "prediction")) ?? asString(readValue(data, "overall_diagnosis")) },
          { label: "肿瘤概率", value: confidenceDisplay(tumorProbabilityValue) },
          { label: "模型置信度", value: confidenceDisplay(confidenceValue) },
          { label: "置信度阈值", value: confidenceDisplay(confidenceThresholdValue) },
          { label: "已分析切片", value: results.length > 0 ? `${results.length} 张` : asNumber(readValue(data, "slides_analyzed")) },
        ])}
        {renderConfidenceReviewNotice(showReviewNotice)}
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
      <div className="clinical-card-section">
        <p className="clinical-card-kicker">影像组学报告</p>
        <strong className="clinical-card-heading">{`患者 ${patientId}`}</strong>
        <p className="clinical-copy clinical-copy-tight">
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
  const followUp = asDecisionFollowUpItems(data.follow_up_plan, data.follow_up);
  const goals = Array.isArray(data.treatment_goals) ? data.treatment_goals : [];
  const considerations = Array.isArray(data.key_considerations) ? data.key_considerations : [];

  return (
    <>
      <div className="clinical-card-section">
        <p className="clinical-card-kicker">治疗决策</p>
        <p className="clinical-copy clinical-copy-tight">{summary}</p>
        {renderMetaItems([{ label: "分期结论", value: asString(data.staging_conclusion) ?? asString(data.staging) }])}
      </div>
      {goals.length > 0 ? (
        <div className="clinical-card-section">
          <strong>治疗目标</strong>
          <ul className="clinical-list">
            {goals.map((goal, index) => (
              <li key={`goal-${index}`} className="clinical-list-item">
                {String(goal)}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {plans.length > 0 ? (
        <div className="clinical-card-section">
          <strong>治疗计划</strong>
          <ul className="clinical-list">
            {plans.map((plan, index) => (
              <li key={`plan-${index}`} className="clinical-list-item">
                <strong>
                  {asString(plan.phase) ?? asString(plan.title) ?? asString(plan.step) ?? asString(plan.name) ?? `阶段 ${index + 1}`}
                </strong>
                <p className="clinical-copy clinical-copy-tight">
                  {asString(plan.regimen) ?? asString(plan.content) ?? asString(plan.rationale) ?? asString(plan.reasoning) ?? asString(plan.details) ?? "暂无说明。"}
                </p>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {followUp.length > 0 ? (
        <div className="clinical-card-section">
          <strong>随访计划</strong>
          <ul className="clinical-list">
            {followUp.map((item, index) => (
              <li key={`follow-${index}`} className="clinical-list-item">
                {item}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {considerations.length > 0 ? (
        <div className="clinical-card-section">
          <strong>关键考虑点</strong>
          <ul className="clinical-list">
            {considerations.map((item, index) => (
              <li key={`consideration-${index}`} className="clinical-list-item">
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
  const symptomFocus = asString(payload.symptom_focus) ?? asString(symptomSnapshot?.symptom_focus);
  const suggestedTests = renderValueList(payload.suggested_tests);

  return (
    <>
      {summary ? <p className="clinical-copy clinical-copy-tight">{summary}</p> : null}
      {renderMetaItems([
        { label: "风险等级", value: riskLevel },
        { label: "建议去向", value: disposition },
        { label: "主诉症状", value: chiefSymptoms },
        { label: "症状归类", value: symptomFocus },
      ])}
      {suggestedTests ? (
        <div className="clinical-card-section">
          <strong>建议检查</strong>
          {suggestedTests}
        </div>
      ) : null}
      {renderDisclosure("查看原始数据", payload)}
    </>
  );
}

function renderExportableMaterialCard(payload: JsonObject) {
  const title = asString(payload.title) ?? "患者资料";
  const markdown = asString(payload.markdown) ?? "";
  const filename = asString(payload.suggested_filename) ?? "patient-material";
  const preview = markdown.replace(/^#+\s*/gm, "").trim().slice(0, 180);

  function downloadMarkdown() {
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${filename}.md`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return (
    <div className="clinical-card-stack">
      <div className="clinical-card-section">
        <strong className="clinical-card-heading">{title}</strong>
        <p className="clinical-copy clinical-copy-tight">{preview || "资料已生成。"}</p>
      </div>
      <div className="clinical-action-row">
        <Button type="button" variant="secondary" size="sm" onClick={downloadMarkdown}>
          下载 Markdown
        </Button>
      </div>
      {renderDisclosure("查看原始数据", payload)}
    </div>
  );
}

function renderGenericCard(payload: JsonObject) {
  const summary = cardSummary(payload);
  if (summary) {
    return (
      <>
        <p className="clinical-copy clinical-copy-tight">{summary}</p>
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

export function renderCardContent({ cardType, payload, onPromptRequest, patientContext }: CardRendererContext): ReactNode {
  switch (cardType) {
    case "medical_card":
      return renderMedicalCard(payload, onPromptRequest, patientContext);
    case "patient_card":
      return renderPatientCard(payload, onPromptRequest, patientContext);
    case "imaging_card":
      return renderImagingVisualCard(payload);
    case "tumor_detection_card":
    case "tumor_screening_result":
      return renderTumorDetectionVisualCard(payload, onPromptRequest, patientContext);
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
    case "exportable_material_card":
      return renderExportableMaterialCard(payload);
    default:
      return renderGenericCard(payload);
  }
}

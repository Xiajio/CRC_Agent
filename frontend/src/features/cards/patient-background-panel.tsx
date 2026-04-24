import type { JsonObject, JsonValue } from "../../app/api/types";

type PatientBackgroundPanelProps = {
  title?: string;
  emptyMessage?: string;
  cards: Record<string, JsonObject>;
};

type FieldItem = {
  label: string;
  value: string | null;
};

const PATIENT_CARD_TYPE = "patient_card";

function asObject(value: unknown): JsonObject | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as JsonObject;
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
        return fieldText(readValue(fieldObject, key), options);
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

  return asString(readValue(fieldMetaRoot, `${section}.${field}`));
}

function knownText(value: string | null): string | null {
  if (!value) {
    return null;
  }
  const normalized = value.trim().toLowerCase();
  if (
    normalized === "n/a" ||
    normalized === "na" ||
    normalized === "none" ||
    normalized === "null" ||
    normalized.includes("待确认") ||
    normalized.includes("未确认") ||
    normalized.includes("未知") ||
    normalized.includes("未提供")
  ) {
    return null;
  }
  return value;
}

function patientFieldText(
  payload: JsonObject,
  section: string,
  field: string,
  value: unknown,
  preferFieldMeta: boolean,
  options?: { booleanAsLabel?: boolean; suffix?: string },
): string | null {
  const display = preferFieldMeta ? fieldMetaDisplay(payload, section, field) : null;
  return knownText(display ?? fieldText(value, options));
}

function listText(value: unknown): string | null {
  if (!Array.isArray(value)) {
    return knownText(fieldText(value));
  }
  const values = value.map((item) => knownText(fieldText(item))).filter((item): item is string => Boolean(item));
  return values.length > 0 ? values.join("、") : null;
}

function groupVisibleItems(items: FieldItem[]): FieldItem[] {
  return items.filter((item) => item.value !== null && item.value !== "");
}

function PatientBackgroundSection({ title, items }: { title: string; items: FieldItem[] }) {
  const visibleItems = groupVisibleItems(items);
  if (visibleItems.length === 0) {
    return null;
  }

  return (
    <div className="patient-background-section">
      <h3>{title}</h3>
      <dl className="patient-background-fields">
        {visibleItems.map((item) => (
          <div key={item.label} className="patient-background-field">
            <dt>{item.label}</dt>
            <dd>{item.value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export function PatientBackgroundPanel({
  title = "患者背景信息",
  emptyMessage = "当前暂无患者背景信息",
  cards,
}: PatientBackgroundPanelProps) {
  const patientCard = asObject(cards[PATIENT_CARD_TYPE]);
  if (!patientCard) {
    return (
      <section className="clinical-card patient-background-panel" data-testid="patient-background-panel">
        <div className="clinical-panel-header">
          <span className="clinical-panel-icon clinical-grid-icon" aria-hidden="true" />
          <h2>{title}</h2>
        </div>
        <div className="patient-background-body">
          <p className="patient-background-empty">{emptyMessage}</p>
        </div>
      </section>
    );
  }

  const data = asObject(patientCard.data);
  const patientInfo = asObject(readValue(data, "patient_info"));
  const diagnosisBlock = asObject(readValue(data, "diagnosis_block"));
  const stagingBlock = asObject(readValue(data, "staging_block"));
  const historyBlock = asObject(readValue(data, "history_block"));
  const cardMeta = asObject(readValue(patientCard, "card_meta"));
  const preferFieldMeta = asString(readValue(cardMeta, "source_mode")) === "patient_self_report";
  const patientId = asString(patientCard.patient_id) ?? asString(readValue(data, "patient_id")) ?? "";
  const patientLabel = patientId && patientId !== "current" ? `患者 #${patientId}` : "当前患者";

  const ctStage = patientFieldText(patientCard, "staging_block", "ct_stage", readValue(stagingBlock, "ct_stage"), preferFieldMeta);
  const cnStage = patientFieldText(patientCard, "staging_block", "cn_stage", readValue(stagingBlock, "cn_stage"), preferFieldMeta);
  const cmStage = patientFieldText(patientCard, "staging_block", "cm_stage", readValue(stagingBlock, "cm_stage"), preferFieldMeta);
  const tnm = [ctStage, cnStage, cmStage].filter(Boolean).join("");

  const groups = [
    {
      title: "基本资料",
      items: [
        { label: "性别", value: patientFieldText(patientCard, "patient_info", "gender", readValue(patientInfo, "gender"), preferFieldMeta) },
        { label: "年龄", value: patientFieldText(patientCard, "patient_info", "age", readValue(patientInfo, "age"), preferFieldMeta, { suffix: "岁" }) },
        { label: "ECOG", value: patientFieldText(patientCard, "patient_info", "ecog", readValue(patientInfo, "ecog"), preferFieldMeta) },
        { label: "CEA", value: patientFieldText(patientCard, "patient_info", "cea", readValue(patientInfo, "cea"), preferFieldMeta) },
      ],
    },
    {
      title: "就诊线索",
      items: [
        { label: "主诉", value: patientFieldText(patientCard, "history_block", "chief_complaint", readValue(historyBlock, "chief_complaint"), preferFieldMeta) },
        { label: "持续时间", value: patientFieldText(patientCard, "history_block", "symptom_duration", readValue(historyBlock, "symptom_duration"), preferFieldMeta) },
        { label: "家族史", value: patientFieldText(patientCard, "history_block", "family_history", readValue(historyBlock, "family_history"), preferFieldMeta, { booleanAsLabel: true }) },
        { label: "危险因素", value: listText(readValue(historyBlock, "risk_factors")) },
      ],
    },
    {
      title: "诊疗背景",
      items: [
        { label: "确诊", value: patientFieldText(patientCard, "diagnosis_block", "confirmed", readValue(diagnosisBlock, "confirmed"), preferFieldMeta) },
        { label: "原发部位", value: patientFieldText(patientCard, "diagnosis_block", "primary_site", readValue(diagnosisBlock, "primary_site"), preferFieldMeta) },
        { label: "临床分期", value: patientFieldText(patientCard, "staging_block", "clinical_stage", readValue(stagingBlock, "clinical_stage"), preferFieldMeta) },
        { label: "TNM", value: tnm || null },
        { label: "MMR", value: patientFieldText(patientCard, "diagnosis_block", "mmr_status", readValue(diagnosisBlock, "mmr_status"), preferFieldMeta) },
      ],
    },
  ];

  const allItems = groups.flatMap((group) => group.items);
  const knownCount = allItems.filter((item) => item.value).length;
  const pendingLabels = allItems.filter((item) => !item.value).map((item) => item.label);
  const pendingPreview = pendingLabels.slice(0, 6).join("、");
  const pendingSuffix = pendingLabels.length > 6 ? `等 ${pendingLabels.length} 项` : `${pendingLabels.length} 项`;
  const summary = knownCount > 0 ? `已整理 ${knownCount} 项背景信息，仍有 ${pendingLabels.length} 项待确认。` : "暂未采集到可用背景信息。";

  return (
    <section className="clinical-card patient-background-panel" data-testid="patient-background-panel">
      <div className="clinical-panel-header">
        <span className="clinical-panel-icon clinical-grid-icon" aria-hidden="true" />
        <h2>{title}</h2>
      </div>
      <div className="patient-background-body">
        <div className="patient-background-hero">
          <span className="patient-background-avatar" aria-hidden="true">患</span>
          <div>
            <strong>{patientLabel}</strong>
            <p>{summary}</p>
          </div>
        </div>

        {knownCount > 0
          ? groups.map((group) => <PatientBackgroundSection key={group.title} title={group.title} items={group.items} />)
          : null}

        {pendingLabels.length > 0 ? (
          <div className="patient-background-pending">
            <span>待补充</span>
            <p>{pendingPreview ? `${pendingPreview}${pendingLabels.length > 6 ? "等" : ""}` : pendingSuffix}</p>
          </div>
        ) : null}
      </div>
    </section>
  );
}

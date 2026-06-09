import type { JsonObject } from "../../app/api/types";
import { Card } from "../../components/ui";

type PatientProfilePanelProps = {
  patientProfile: JsonObject | null;
};

const PRIMARY_PATIENT_IDENTITY_KEYS = [
  "case_database_patient_id",
  "registry_patient_id",
  "patient_id",
];

const PATIENT_IDENTITY_DISPLAY_ORDER = [
  ...PRIMARY_PATIENT_IDENTITY_KEYS,
  "current_patient_id",
];

function formatValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "未知";
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value);
}

function fieldLabel(key: string): string {
  const labels: Record<string, string> = {
    name: "姓名",
    age: "年龄",
    diagnosis: "诊断",
    mrn: "病历号",
    gender: "性别",
    sex: "性别",
    dob: "出生日期",
    birth_date: "出生日期",
    date_of_birth: "出生日期",
    case_database_patient_id: "病例库样本ID",
    registry_patient_id: "登记患者ID",
    current_patient_id: "兼容患者ID",
    patient_id: "患者ID",
    allergies: "过敏史",
    medications: "用药",
    medications_list: "用药",
    history: "病史",
    summary: "摘要",
    status: "状态",
    stage: "阶段",
    note: "备注",
    notes: "备注",
  };

  return labels[key] ?? `字段：${key.replace(/_/g, "")}`;
}

function patientProfileEntries(patientProfile: JsonObject): Array<[string, unknown]> {
  const hasPrimaryIdentity = PRIMARY_PATIENT_IDENTITY_KEYS.some(
    (key) => patientProfile[key] !== undefined && patientProfile[key] !== null,
  );

  return Object.entries(patientProfile)
    .filter(([key]) => !(key === "current_patient_id" && hasPrimaryIdentity))
    .sort(([left], [right]) => {
      const leftIndex = PATIENT_IDENTITY_DISPLAY_ORDER.indexOf(left);
      const rightIndex = PATIENT_IDENTITY_DISPLAY_ORDER.indexOf(right);

      if (leftIndex >= 0 && rightIndex >= 0) {
        return leftIndex - rightIndex;
      }
      if (leftIndex >= 0) {
        return -1;
      }
      if (rightIndex >= 0) {
        return 1;
      }
      return 0;
    });
}

export function PatientProfilePanel({ patientProfile }: PatientProfilePanelProps) {
  return (
    <Card variant="clinical-panel">
      <h2>患者画像</h2>
      {patientProfile ? (
        <dl className="clinical-detail-list">
          {patientProfileEntries(patientProfile).map(([key, value]) => (
            <div key={key} className="clinical-detail-row">
              <dt>{fieldLabel(key)}</dt>
              <dd>{formatValue(value)}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="clinical-copy">等待患者信息加载</p>
      )}
    </Card>
  );
}

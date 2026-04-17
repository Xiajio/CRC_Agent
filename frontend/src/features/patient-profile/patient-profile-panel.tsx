import type { JsonObject } from "../../app/api/types";

type PatientProfilePanelProps = {
  patientProfile: JsonObject | null;
};

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
    current_patient_id: "当前患者ID",
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

export function PatientProfilePanel({ patientProfile }: PatientProfilePanelProps) {
  return (
    <div className="workspace-card">
      <h2>患者画像</h2>
      {patientProfile ? (
        <dl className="workspace-detail-list">
          {Object.entries(patientProfile).map(([key, value]) => (
            <div key={key} className="workspace-detail-row">
              <dt>{fieldLabel(key)}</dt>
              <dd>{formatValue(value)}</dd>
            </div>
          ))}
        </dl>
      ) : (
        <p className="workspace-copy">等待患者信息加载</p>
      )}
    </div>
  );
}

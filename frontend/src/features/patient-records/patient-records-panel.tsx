import type { PatientRegistryRecord } from "../../app/api/types";
import { Card } from "../../components/ui";

type PatientRecordsPanelProps = {
  records: PatientRegistryRecord[];
  isLoading: boolean;
};

function recordTypeLabel(record: PatientRegistryRecord): string {
  if (record.record_type === "crc_triage_assessment") {
    return "CRC 专项问诊";
  }
  if (record.document_type) {
    return record.document_type;
  }
  return record.record_type || "患者记录";
}

function formatRecordTime(value: string): string {
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp)) {
    return value || "时间未记录";
  }
  return new Date(timestamp).toLocaleString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function PatientRecordsPanel({ records, isLoading }: PatientRecordsPanelProps) {
  return (
    <Card as="section" variant="clinical-panel">
      <h2>历史问诊记录</h2>
      {isLoading ? <p className="clinical-copy">正在加载历史问诊记录...</p> : null}
      {!isLoading && records.length === 0 ? <p className="clinical-copy">当前暂无历史问诊记录</p> : null}
      {!isLoading && records.length > 0 ? (
        <ul className="clinical-list">
          {records.map((record) => (
            <li key={record.record_id} className="clinical-list-item">
              <strong>{recordTypeLabel(record)}</strong>
              <p className="clinical-copy clinical-copy-tight">{record.summary_text}</p>
              <p className="clinical-meta-text">{formatRecordTime(record.created_at)}</p>
            </li>
          ))}
        </ul>
      ) : null}
    </Card>
  );
}

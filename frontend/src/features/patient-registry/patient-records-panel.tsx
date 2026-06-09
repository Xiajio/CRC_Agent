import type { PatientRegistryRecord } from "../../app/api/types";
import { Card } from "../../components/ui";

type PatientRecordsPanelProps = {
  records: PatientRegistryRecord[];
  isLoading: boolean;
};

function recordMeta(record: PatientRegistryRecord): string {
  const segments = [
    record.document_type,
    record.ingest_decision,
    record.source,
  ].filter((value) => typeof value === "string" && value.trim().length > 0);
  return segments.length > 0 ? segments.join(" / ") : "暂无元数据";
}

export function PatientRecordsPanel({ records, isLoading }: PatientRecordsPanelProps) {
  return (
    <Card as="section" variant="clinical-panel">
      <h2 style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <span style={{ fontSize: "1.2rem" }}>📁</span> 患者病历记录
      </h2>
      {isLoading ? <p className="clinical-copy" style={{ color: "var(--color-primary)" }}>正在加载病历记录...</p> : null}
      {!isLoading && records.length === 0 ? (
        <p className="clinical-copy">暂无病历记录。</p>
      ) : null}
      {records.length > 0 ? (
        <ul className="clinical-list" style={{ gap: "10px" }}>
          {records.map((record) => (
            <li key={record.record_id} className="clinical-list-item">
              <strong style={{ color: "var(--color-primary)" }}>{`记录 #${record.record_id}`}</strong>
              <p className="clinical-copy clinical-copy-tight">{record.summary_text}</p>
              <p className="clinical-meta-text" style={{ marginTop: "8px" }}>{recordMeta(record)}</p>
            </li>
          ))}
        </ul>
      ) : null}
    </Card>
  );
}

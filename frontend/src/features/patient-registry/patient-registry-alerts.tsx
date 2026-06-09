import type { PatientRegistryAlert } from "../../app/api/types";
import { Card } from "../../components/ui";

type PatientRegistryAlertsPanelProps = {
  alerts: PatientRegistryAlert[];
  isLoading: boolean;
};

function alertLabel(alert: PatientRegistryAlert): string {
  if (alert.record_id !== null && alert.record_id !== undefined) {
    return `${alert.kind} / 记录 #${alert.record_id}`;
  }
  return alert.kind;
}

export function PatientRegistryAlertsPanel({ alerts, isLoading }: PatientRegistryAlertsPanelProps) {
  return (
    <Card as="section" variant="clinical-panel">
      <h2 style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <span style={{ fontSize: "1.2rem" }}>⚠️</span> 患者库预警
      </h2>
      {isLoading ? <p className="clinical-copy" style={{ color: "var(--color-primary)" }}>正在加载预警信息...</p> : null}
      {!isLoading && alerts.length === 0 ? (
        <p className="clinical-copy">暂无预警信息。</p>
      ) : null}
      {alerts.length > 0 ? (
        <ul className="clinical-list" style={{ gap: "10px" }}>
          {alerts.map((alert, index) => (
            <li key={`${alert.kind}-${alert.record_id ?? "none"}-${index}`} className="clinical-list-item" style={{ borderLeft: "4px solid var(--color-warning)" }}>
              <strong>{alertLabel(alert)}</strong>
              <p className="clinical-copy clinical-copy-tight">{alert.message}</p>
            </li>
          ))}
        </ul>
      ) : null}
    </Card>
  );
}

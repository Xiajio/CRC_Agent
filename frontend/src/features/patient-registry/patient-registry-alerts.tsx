import type { PatientRegistryAlert } from "../../app/api/types";

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
    <section className="workspace-card">
      <h2 style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <span style={{ fontSize: "1.2rem" }}>⚠️</span> 患者库预警
      </h2>
      {isLoading ? <p className="workspace-copy" style={{ color: "#8e4a55" }}>正在加载预警信息...</p> : null}
      {!isLoading && alerts.length === 0 ? (
        <p className="workspace-copy">暂无预警信息。</p>
      ) : null}
      {alerts.length > 0 ? (
        <ul className="workspace-list" style={{ gap: "10px" }}>
          {alerts.map((alert, index) => (
            <li key={`${alert.kind}-${alert.record_id ?? "none"}-${index}`} className="workspace-list-item" style={{ borderLeft: "4px solid #a35d68" }}>
              <strong>{alertLabel(alert)}</strong>
              <p className="workspace-copy workspace-copy-tight">{alert.message}</p>
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}

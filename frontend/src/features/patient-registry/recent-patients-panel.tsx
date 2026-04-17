import type { PatientRegistryItem } from "../../app/api/types";

type RecentPatientsPanelProps = {
  title?: string;
  emptyMessage?: string;
  items: PatientRegistryItem[];
  previewedPatientId: number | null;
  isLoading: boolean;
  error: string | null;
  onPreviewPatient: (patientId: number) => void;
};

function patientSummary(item: PatientRegistryItem): string {
  const segments = [item.tumor_location, item.clinical_stage, item.mmr_status]
    .filter((value) => typeof value === "string" && value.trim().length > 0);
  return segments.length > 0 ? segments.join(" / ") : "暂无摘要";
}

export function RecentPatientsPanel({
  title = "👥 最近患者",
  emptyMessage = "暂无最近患者记录。",
  items,
  previewedPatientId,
  isLoading,
  error,
  onPreviewPatient,
}: RecentPatientsPanelProps) {
  return (
    <section className="workspace-card" data-testid="recent-patients-panel">
      <h2 style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <span style={{ fontSize: "1.2rem" }}></span> {title}
      </h2>
      {error ? <p className="workspace-copy workspace-copy-alert">{error}</p> : null}
      {isLoading ? <p className="workspace-copy" style={{ color: "#8e4a55" }}>正在加载最近患者...</p> : null}
      {!isLoading && items.length === 0 ? <p className="workspace-copy">{emptyMessage}</p> : null}
      {items.length > 0 ? (
        <div style={{ maxHeight: "calc(100vh - 200px)", overflowY: "auto", paddingRight: "8px", marginRight: "-8px" }}>
          <ul className="workspace-list" style={{ gap: "10px", marginTop: "10px" }}>
            {items.map((item) => {
              const isPreviewed = previewedPatientId === item.patient_id;
              return (
                <li key={item.patient_id}>
                  <button
                    type="button"
                    className={`workspace-list-item ${isPreviewed ? "workspace-step-current" : ""}`}
                    onClick={() => onPreviewPatient(item.patient_id)}
                    aria-label={`preview patient ${item.patient_id}`}
                    aria-pressed={isPreviewed}
                    style={{
                      width: "100%",
                      display: "flex",
                      flexDirection: "column",
                      gap: "12px",
                      padding: "16px",
                      textAlign: "left",
                      transition: "all 0.2s ease",
                      border: isPreviewed ? "1px solid rgba(142, 74, 85, 0.3)" : "1px solid rgba(165, 73, 83, 0.12)",
                      boxShadow: isPreviewed ? "0 4px 12px rgba(142, 74, 85, 0.08)" : "none",
                    }}
                  >
                    <div>
                      <strong style={{ color: isPreviewed ? "#8e4a55" : "inherit", fontSize: "1.05rem" }}>
                        {`患者 #${item.patient_id}`}
                      </strong>
                      <p className="workspace-copy workspace-copy-tight" style={{ fontSize: "0.85rem", marginTop: "6px" }}>
                        {patientSummary(item)}
                      </p>
                    </div>
                    <span className="workspace-meta" style={{ fontSize: "0.85rem" }}>
                      {isPreviewed ? "✅ 正在预览" : "👀 预览患者"}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}
    </section>
  );
}

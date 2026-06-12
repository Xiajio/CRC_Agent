import type { PatientRegistryItem } from "../../app/api/types";
import { Card } from "../../components/ui";

type RecentPatientsPanelProps = {
  title?: string;
  emptyMessage?: string;
  items: PatientRegistryItem[];
  previewedPatientId: number | null;
  isLoading: boolean;
  isLoadingPreview: boolean;
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
  isLoadingPreview,
  error,
  onPreviewPatient,
}: RecentPatientsPanelProps) {
  return (
    <Card as="section" variant="clinical-panel" data-testid="recent-patients-panel">
      <h2 className="recent-patients-heading">{title}</h2>
      {error ? <p className="clinical-copy clinical-copy-alert">{error}</p> : null}
      {isLoading ? <p className="clinical-copy recent-patients-loading">正在加载最近患者...</p> : null}
      {!isLoading && items.length === 0 ? <p className="clinical-copy">{emptyMessage}</p> : null}
      {items.length > 0 ? (
        <div className="recent-patients-scroll">
          <ul className="clinical-list recent-patients-list">
            {items.map((item) => {
              const isPreviewed = previewedPatientId === item.patient_id;
              return (
                <li key={item.patient_id}>
                  <button
                    type="button"
                    className={`clinical-list-item recent-patient-button ${
                      isPreviewed ? "clinical-step-current recent-patient-button-active" : ""
                    }`}
                    onClick={() => onPreviewPatient(item.patient_id)}
                    disabled={isLoadingPreview}
                    aria-label={`preview patient ${item.patient_id}`}
                    aria-pressed={isPreviewed}
                  >
                    <div>
                      <strong className="recent-patient-title">
                        {`患者 #${item.patient_id}`}
                      </strong>
                      <p className="clinical-copy clinical-copy-tight recent-patient-summary">
                        {patientSummary(item)}
                      </p>
                    </div>
                    <span className="clinical-meta-text recent-patient-status">
                      {isPreviewed ? "正在预览" : "预览患者"}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}
    </Card>
  );
}

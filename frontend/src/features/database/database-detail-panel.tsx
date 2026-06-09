import type { DatabaseCaseDetailResponse } from "../../app/api/types";
import { Card } from "../../components/ui";
import { cardTitle, renderCardContent, type CardPromptHandler } from "../cards/card-renderers-extended";

interface DatabaseDetailPanelProps {
  detail: DatabaseCaseDetailResponse | null;
  onPromptRequest?: CardPromptHandler;
}

export function DatabaseDetailPanel({ detail, onPromptRequest }: DatabaseDetailPanelProps) {
  if (!detail) {
    return (
      <Card>
        <h2>{"\u60a3\u8005\u8be6\u60c5"}</h2>
        <p className="clinical-copy">
          {"\u4ece\u75c5\u4f8b\u5217\u8868\u4e2d\u9009\u4e2d\u4e00\u4f4d\u60a3\u8005\u540e\uff0c\u8fd9\u91cc\u4f1a\u590d\u7528\u5de5\u4f5c\u53f0\u5361\u7247\u5c55\u793a\u5168\u90e8\u8be6\u60c5\u3002"}
        </p>
      </Card>
    );
  }

  const patientSampleId = String(detail.patient_id).padStart(3, "0");

  return (
    <div className="database-detail-stack">
      <Card>
        <div className="database-section-heading database-section-heading-inline">
          <h2>{`\u60a3\u8005 #${patientSampleId}`}</h2>
          <div className="database-badge-row">
            <span className="database-pill">{detail.available_data.case_info ? "Case" : "No Case"}</span>
            <span className="database-pill">{detail.available_data.imaging ? "Imaging" : "No Imaging"}</span>
            <span className="database-pill">
              {detail.available_data.pathology_slides ? "Pathology" : "No Pathology"}
            </span>
          </div>
        </div>
      </Card>
      {Object.entries(detail.cards).map(([cardType, payload]) => (
        <Card key={cardType}>
          <h2>{cardTitle(cardType, payload)}</h2>
          {renderCardContent({ cardType, payload, onPromptRequest })}
        </Card>
      ))}
    </div>
  );
}

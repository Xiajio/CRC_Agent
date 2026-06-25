import type { FrontendMessage } from "../../app/api/types";
import { Button, Card } from "../../components/ui";
import type { CardPatientContext, CardPromptHandler } from "../cards/card-renderers-extended";
import {
  buildDoctorReportDraftPrompt,
  buildDoctorReportPromptContext,
  DOCTOR_REPORT_DRAFT_ACTIONS,
  latestReportDraftFromMessages,
} from "./doctor-report-draft-utils";

export type DoctorReportDraftViewProps = {
  registryPatientId: number | null;
  caseDatabasePatientId: string | null;
  messages: FrontendMessage[];
  isStreaming: boolean;
  disabled: boolean;
  patientContext?: CardPatientContext | null;
  onReportPromptRequest?: CardPromptHandler;
};

function reportPatientContext(props: DoctorReportDraftViewProps) {
  const context: CardPatientContext = {
    ...(props.patientContext ?? {}),
  };

  if (props.registryPatientId !== null) {
    context.registry_patient_id = props.registryPatientId;
  }
  if (props.caseDatabasePatientId !== null) {
    context.case_database_patient_id = props.caseDatabasePatientId;
  }

  return buildDoctorReportPromptContext(context);
}

function contextLabel(registryPatientId: number | undefined, caseDatabasePatientId: string | undefined) {
  const registry = registryPatientId ? `P-${registryPatientId}` : "未绑定";
  const sample = caseDatabasePatientId ?? "未绑定";
  return `登记号 ${registry} / 病例样本 ${sample}`;
}

export function DoctorReportDraftView(props: DoctorReportDraftViewProps) {
  const promptContext = reportPatientContext(props);
  const hasPatientContext = Boolean(promptContext.registry_patient_id || promptContext.case_database_patient_id);
  const canGenerate = Boolean(props.onReportPromptRequest) && hasPatientContext && !props.disabled && !props.isStreaming;
  const latestDraft = latestReportDraftFromMessages(props.messages);
  const canExport = Boolean(latestDraft?.text) && !props.disabled && !props.isStreaming;

  function handlePrint() {
    if (typeof window !== "undefined" && typeof window.print === "function") {
      window.print();
    }
  }

  return (
    <main className="clinical-report-draft-dashboard" data-testid="doctor-report-draft-view">
      <Card as="section" padding="none" className="clinical-card clinical-report-draft-card">
        <div className="clinical-panel-header">
          <span className="clinical-panel-icon" aria-hidden="true" />
          <h2>报告草稿</h2>
        </div>
        <div className="clinical-report-draft-body">
          <p className="clinical-report-draft-context">
            {contextLabel(promptContext.registry_patient_id, promptContext.case_database_patient_id)}
          </p>
          <div className="clinical-report-draft-actions">
            {DOCTOR_REPORT_DRAFT_ACTIONS.map((action) => (
              <Button
                key={action.key}
                type="button"
                variant={canGenerate ? "primary" : "secondary"}
                disabled={!canGenerate}
                title={hasPatientContext ? action.summary : "需要先绑定登记患者或病例样本"}
                onClick={() => {
                  if (!canGenerate || !props.onReportPromptRequest) {
                    return;
                  }
                  props.onReportPromptRequest(buildDoctorReportDraftPrompt(action), promptContext);
                }}
              >
                {action.title}
              </Button>
            ))}
          </div>
        </div>
      </Card>

      <Card as="section" padding="none" className="clinical-card clinical-report-draft-card clinical-report-draft-preview">
        <div className="clinical-panel-header">
          <span className="clinical-panel-icon" aria-hidden="true" />
          <h2>最新草稿</h2>
          <Button type="button" size="sm" variant="secondary" disabled={!canExport} onClick={handlePrint}>
            导出 PDF
          </Button>
        </div>
        <div className="clinical-report-draft-print-area">
          {latestDraft ? (
            <pre className="clinical-report-draft-text">{latestDraft.text}</pre>
          ) : (
            <div className="clinical-multimodal-empty-state">
              <p>暂无报告草稿</p>
            </div>
          )}
        </div>
      </Card>
    </main>
  );
}

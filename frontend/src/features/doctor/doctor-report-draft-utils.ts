import type { FrontendMessage } from "../../app/api/types";
import type { CardPatientContext } from "../cards/card-renderers-extended";

export type DoctorReportDraftActionKey = "case_summary" | "consultation_report" | "handoff_note";

export type DoctorReportDraftAction = {
  key: DoctorReportDraftActionKey;
  title: string;
  summary: string;
  promptTitle: string;
};

export type DoctorReportPromptContext = {
  registry_patient_id?: number;
  case_database_patient_id?: string;
};

export type ReportDraftPreview = {
  cursor: string;
  text: string;
};

const REPORT_DRAFT_REQUIREMENTS = [
  "请使用 Markdown 输出，适合作为医生复核后的文书草稿。",
  "请按以下结构组织：基本信息、资料来源、主诉与病史、检查摘要、影像与病理、分期与评估、诊疗建议、缺失资料、人工复核提示。",
  "即使资料缺失，也必须输出病例/报告草稿模板，并把缺失内容写入“缺失资料/待核实”章节；已有资料填入对应章节。",
  "只基于当前对话、绑定患者、病例样本、医疗卡片和已检索证据；不确定的内容请明确写入缺失资料或待核实。",
  "不要只输出缺失提醒、补充资料请求或单段说明；草稿主体必须始终存在。",
  "不要写成最终签署文书，不要替代医生判断。",
].join("\n");

export const DOCTOR_REPORT_DRAFT_ACTIONS: readonly DoctorReportDraftAction[] = [
  {
    key: "case_summary",
    title: "病例摘要草稿",
    summary: "汇总当前患者资料和对话中的关键病情。",
    promptTitle: "病例摘要草稿",
  },
  {
    key: "consultation_report",
    title: "会诊报告草稿",
    summary: "生成面向会诊记录的结构化报告草稿。",
    promptTitle: "会诊报告草稿",
  },
  {
    key: "handoff_note",
    title: "交接记录草稿",
    summary: "生成便于交接班或转诊沟通的摘要。",
    promptTitle: "交接记录草稿",
  },
] as const;

function readInteger(value: unknown): number | null {
  if (typeof value === "number" && Number.isInteger(value) && value > 0) {
    return value;
  }
  if (typeof value !== "string") {
    return null;
  }
  const normalized = value.trim();
  if (!/^\d+$/.test(normalized)) {
    return null;
  }
  const parsed = Number(normalized);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

function readCaseDatabasePatientId(value: unknown): string | null {
  if (typeof value === "number" && Number.isInteger(value) && value > 0) {
    return String(value).padStart(3, "0");
  }
  if (typeof value !== "string") {
    return null;
  }
  const normalized = value.trim();
  if (!normalized) {
    return null;
  }
  return /^\d+$/.test(normalized) ? normalized.padStart(3, "0") : normalized;
}

export function buildDoctorReportPromptContext(
  patientContext: CardPatientContext | null | undefined,
): DoctorReportPromptContext {
  if (!patientContext) {
    return {};
  }

  const context: DoctorReportPromptContext = {};
  const registryPatientId = readInteger(patientContext.registry_patient_id);
  const caseDatabasePatientId = readCaseDatabasePatientId(patientContext.case_database_patient_id);

  if (registryPatientId !== null) {
    context.registry_patient_id = registryPatientId;
  }
  if (caseDatabasePatientId !== null) {
    context.case_database_patient_id = caseDatabasePatientId;
  }

  return context;
}

export function buildDoctorReportDraftPrompt(action: DoctorReportDraftAction): string {
  return `请生成${action.promptTitle}。\n${REPORT_DRAFT_REQUIREMENTS}`;
}

function messageText(content: unknown): string | null {
  return typeof content === "string" && content.trim() ? content.trim() : null;
}

export function latestReportDraftFromMessages(messages: FrontendMessage[]): ReportDraftPreview | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.type !== "ai") {
      continue;
    }

    const text = messageText(message.content);
    if (text) {
      return {
        cursor: message.cursor,
        text,
      };
    }
  }

  return null;
}

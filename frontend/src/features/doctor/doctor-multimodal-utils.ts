import type { JsonObject } from "../../app/api/types";
import type { CardPatientContext } from "../cards/card-renderers-extended";

export type MultimodalCardGroupKey = "imaging" | "pathology" | "radiomics";

export type MultimodalCardGroup = {
  key: MultimodalCardGroupKey;
  title: string;
  summary: string;
  cards: Array<{
    cardType: string;
    payload: JsonObject;
  }>;
};

export type MultimodalPromptContext = {
  registry_patient_id?: number;
  case_database_patient_id?: string;
};

export type MultimodalActionKey = "imaging_review" | "pathology_review" | "case_summary" | "handoff_note";

export type MultimodalAction = {
  key: MultimodalActionKey;
  title: string;
  summary: string;
  prompt: string;
  contextRequirement: "case_database_patient_id" | "registry_patient_id";
};

type MultimodalCardEntry = {
  cardType: string;
  payload: JsonObject;
};

type MultimodalActionState = MultimodalAction & {
  disabled: boolean;
};

const MULTIMODAL_CARD_GROUPS: Record<
  MultimodalCardGroupKey,
  Omit<MultimodalCardGroup, "key" | "cards">
> = {
  imaging: {
    title: "影像",
    summary: "整理影像样本与肿瘤检测卡片。",
  },
  pathology: {
    title: "病理",
    summary: "整理病理报告与病理切片卡片。",
  },
  radiomics: {
    title: "影像组学",
    summary: "整理影像组学分析卡片。",
  },
};

const MULTIMODAL_CARD_TYPES: Record<string, MultimodalCardGroupKey> = {
  imaging_card: "imaging",
  tumor_detection_card: "imaging",
  tumor_screening_result: "imaging",
  pathology_card: "pathology",
  pathology_slide_card: "pathology",
  radiomics_report_card: "radiomics",
};

function asObject(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }

  return value as Record<string, unknown>;
}

function asCardEntry(value: unknown, cardTypeFallback?: string): MultimodalCardEntry | null {
  const card = asObject(value);
  if (!card) {
    return null;
  }

  const cardType = typeof card.cardType === "string" && card.cardType.trim()
    ? card.cardType.trim()
    : typeof card.card_type === "string" && card.card_type.trim()
      ? card.card_type.trim()
      : cardTypeFallback?.trim();
  const payload = asObject(card.payload) ?? card;

  if (!cardType) {
    return null;
  }

  return {
    cardType,
    payload: payload as JsonObject,
  };
}

function readInteger(value: unknown): number | null {
  if (typeof value === "number" && Number.isInteger(value)) {
    return value;
  }

  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isInteger(parsed)) {
      return parsed;
    }
  }

  return null;
}

function readCaseDatabasePatientId(value: unknown): string | null {
  if (typeof value === "number" && Number.isInteger(value) && value >= 0) {
    return String(value).padStart(3, "0");
  }

  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isInteger(parsed) && parsed >= 0) {
      return String(parsed).padStart(3, "0");
    }
  }

  return null;
}

function buildMultimodalCardGroup(key: MultimodalCardGroupKey, cards: MultimodalCardEntry[]): MultimodalCardGroup {
  return {
    key,
    title: MULTIMODAL_CARD_GROUPS[key].title,
    summary: MULTIMODAL_CARD_GROUPS[key].summary,
    cards,
  };
}

export const MULTIMODAL_ACTIONS: readonly MultimodalAction[] = [
  {
    key: "imaging_review",
    title: "影像分析",
    summary: "基于病例样本整理影像要点。",
    prompt: "请结合病例样本整理影像要点，并给出临床可执行的结论。",
    contextRequirement: "case_database_patient_id",
  },
  {
    key: "pathology_review",
    title: "病理分析",
    summary: "基于病例样本整理病理要点。",
    prompt: "请结合病例样本整理病理要点，并给出临床可执行的结论。",
    contextRequirement: "case_database_patient_id",
  },
  {
    key: "case_summary",
    title: "病例摘要",
    summary: "基于登记患者信息生成病例摘要。",
    prompt: "请结合登记患者信息生成病例摘要。",
    contextRequirement: "registry_patient_id",
  },
  {
    key: "handoff_note",
    title: "交接说明",
    summary: "基于登记患者信息生成交接说明。",
    prompt: "请结合登记患者信息生成交接说明。",
    contextRequirement: "registry_patient_id",
  },
] as const;

export function groupMultimodalCards(
  cards: Array<{ cardType: string; payload: JsonObject }> | Record<string, JsonObject>,
): MultimodalCardGroup[] {
  const entries: MultimodalCardEntry[] = Array.isArray(cards)
    ? cards
        .map((card) => asCardEntry(card))
        .filter((card): card is MultimodalCardEntry => card !== null)
    : Object.entries(cards).map(([cardType, payload]) => asCardEntry({ cardType, payload }, cardType)).filter((card): card is MultimodalCardEntry => card !== null);

  const groupedCards = new Map<MultimodalCardGroupKey, MultimodalCardEntry[]>();
  for (const groupKey of Object.keys(MULTIMODAL_CARD_GROUPS) as MultimodalCardGroupKey[]) {
    groupedCards.set(groupKey, []);
  }

  for (const card of entries) {
    const groupKey = MULTIMODAL_CARD_TYPES[card.cardType];
    if (!groupKey) {
      continue;
    }

    groupedCards.get(groupKey)?.push(card);
  }

  return (Object.keys(MULTIMODAL_CARD_GROUPS) as MultimodalCardGroupKey[])
    .map((groupKey) => buildMultimodalCardGroup(groupKey, groupedCards.get(groupKey) ?? []))
    .filter((group) => group.cards.length > 0);
}

export function buildMultimodalPromptContext(patientContext: CardPatientContext | null | undefined): MultimodalPromptContext {
  if (!patientContext) {
    return {};
  }

  const context: MultimodalPromptContext = {};

  const registryPatientId = readInteger(patientContext.registry_patient_id);
  if (registryPatientId !== null) {
    context.registry_patient_id = registryPatientId;
  }

  const caseDatabasePatientId = readCaseDatabasePatientId(patientContext.case_database_patient_id);
  if (caseDatabasePatientId !== null) {
    context.case_database_patient_id = caseDatabasePatientId;
  }

  return context;
}

export function buildMultimodalActionState(
  action: MultimodalAction,
  context: MultimodalPromptContext | null | undefined,
): MultimodalActionState {
  const hasAnyPatientContext = Boolean(context && (context.registry_patient_id !== undefined || context.case_database_patient_id !== undefined));
  const hasRegistryPatient = typeof context?.registry_patient_id === "number";
  const hasCaseDatabasePatient = typeof context?.case_database_patient_id === "string" && context.case_database_patient_id.length > 0;

  let disabled = true;
  if (hasAnyPatientContext) {
    disabled = action.contextRequirement === "case_database_patient_id" ? !hasCaseDatabasePatient : !hasRegistryPatient;
  }

  return {
    ...action,
    disabled,
  };
}

export function buildMultimodalPrompt(action: MultimodalAction): string {
  return action.prompt;
}

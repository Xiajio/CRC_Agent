export type PatientWorkspaceTab = "assistant" | "crc_triage" | "profile" | "upload";

export type PatientWorkspaceNavItem = {
  key: PatientWorkspaceTab;
  label: string;
  disabled?: boolean;
};

export const PATIENT_ASSISTANT_TAB: PatientWorkspaceTab = "assistant";
export const PATIENT_CRC_TRIAGE_TAB: PatientWorkspaceTab = "crc_triage";
export const PATIENT_PROFILE_TAB: PatientWorkspaceTab = "profile";
export const PATIENT_UPLOAD_TAB: PatientWorkspaceTab = "upload";

export const PATIENT_WORKSPACE_NAV_ITEMS: PatientWorkspaceNavItem[] = [
  { key: PATIENT_ASSISTANT_TAB, label: "\u95ee\u52a9\u624b" },
  { key: PATIENT_CRC_TRIAGE_TAB, label: "\u4e13\u9879\u95ee\u8bca" },
  { key: PATIENT_PROFILE_TAB, label: "\u6211\u7684\u8d44\u6599" },
  { key: PATIENT_UPLOAD_TAB, label: "\u4e0a\u4f20\u62a5\u544a" },
];

export function isPatientWorkspaceTab(value: string): value is PatientWorkspaceTab {
  return (
    value === PATIENT_ASSISTANT_TAB
    || value === PATIENT_CRC_TRIAGE_TAB
    || value === PATIENT_PROFILE_TAB
    || value === PATIENT_UPLOAD_TAB
  );
}

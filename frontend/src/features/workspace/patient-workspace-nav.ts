export type PatientWorkspaceTab = "profile" | "upload";

export type PatientWorkspaceNavItem = {
  key: PatientWorkspaceTab;
  label: string;
  disabled?: boolean;
};

export const PATIENT_PROFILE_TAB: PatientWorkspaceTab = "profile";
export const PATIENT_UPLOAD_TAB: PatientWorkspaceTab = "upload";

export const PATIENT_WORKSPACE_NAV_ITEMS: PatientWorkspaceNavItem[] = [
  { key: PATIENT_PROFILE_TAB, label: "\u8d44\u6599\u586b\u5199" },
  { key: PATIENT_UPLOAD_TAB, label: "\u4e0a\u4f20" },
];

export function isPatientWorkspaceTab(value: string): value is PatientWorkspaceTab {
  return value === PATIENT_PROFILE_TAB || value === PATIENT_UPLOAD_TAB;
}

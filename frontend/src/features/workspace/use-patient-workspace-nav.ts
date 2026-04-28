import { useState } from "react";

import {
  PATIENT_PROFILE_TAB,
  PATIENT_WORKSPACE_NAV_ITEMS,
  type PatientWorkspaceTab,
  isPatientWorkspaceTab,
} from "./patient-workspace-nav";

export { PATIENT_PROFILE_TAB, PATIENT_UPLOAD_TAB } from "./patient-workspace-nav";
export type { PatientWorkspaceNavItem, PatientWorkspaceTab } from "./patient-workspace-nav";

export function usePatientWorkspaceNav() {
  const [activeTab, setActiveTab] = useState<PatientWorkspaceTab>(PATIENT_PROFILE_TAB);

  function selectTab(key: string) {
    if (isPatientWorkspaceTab(key)) {
      setActiveTab(key);
    }
  }

  function resetTab() {
    setActiveTab(PATIENT_PROFILE_TAB);
  }

  return {
    activeTab,
    navItems: PATIENT_WORKSPACE_NAV_ITEMS,
    selectTab,
    resetTab,
  };
}

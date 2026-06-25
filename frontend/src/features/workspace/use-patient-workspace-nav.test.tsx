import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
  PATIENT_ASSISTANT_TAB,
  PATIENT_CRC_TRIAGE_TAB,
  PATIENT_PROFILE_TAB,
  PATIENT_UPLOAD_TAB,
  usePatientWorkspaceNav,
} from "./use-patient-workspace-nav";

describe("usePatientWorkspaceNav", () => {
  it("starts on the assistant tab and exposes production nav items", () => {
    const { result } = renderHook(() => usePatientWorkspaceNav());

    expect(result.current.activeTab).toBe(PATIENT_ASSISTANT_TAB);
    expect(result.current.navItems.map((item) => item.key)).toEqual([
      PATIENT_ASSISTANT_TAB,
      PATIENT_CRC_TRIAGE_TAB,
      PATIENT_PROFILE_TAB,
      PATIENT_UPLOAD_TAB,
    ]);
    expect(result.current.navItems.map((item) => item.label)).toEqual([
      "问助手",
      "专项问诊",
      "我的资料",
      "上传报告",
    ]);
  });

  it("accepts supported patient tabs and ignores unsupported keys", () => {
    const { result } = renderHook(() => usePatientWorkspaceNav());

    act(() => result.current.selectTab(PATIENT_CRC_TRIAGE_TAB));
    expect(result.current.activeTab).toBe(PATIENT_CRC_TRIAGE_TAB);

    act(() => result.current.selectTab(PATIENT_UPLOAD_TAB));
    expect(result.current.activeTab).toBe(PATIENT_UPLOAD_TAB);

    act(() => result.current.selectTab("symptoms"));
    expect(result.current.activeTab).toBe(PATIENT_UPLOAD_TAB);

    act(() => result.current.resetTab());
    expect(result.current.activeTab).toBe(PATIENT_ASSISTANT_TAB);
  });
});

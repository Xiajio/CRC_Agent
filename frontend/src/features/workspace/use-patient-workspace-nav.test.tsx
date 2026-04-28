import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import {
  PATIENT_PROFILE_TAB,
  PATIENT_UPLOAD_TAB,
  usePatientWorkspaceNav,
} from "./use-patient-workspace-nav";

describe("usePatientWorkspaceNav", () => {
  it("starts on the profile tab and exposes production nav items", () => {
    const { result } = renderHook(() => usePatientWorkspaceNav());

    expect(result.current.activeTab).toBe(PATIENT_PROFILE_TAB);
    expect(result.current.navItems.map((item) => item.key)).toEqual([
      PATIENT_PROFILE_TAB,
      PATIENT_UPLOAD_TAB,
    ]);
  });

  it("accepts supported patient tabs and ignores unsupported keys", () => {
    const { result } = renderHook(() => usePatientWorkspaceNav());

    act(() => result.current.selectTab(PATIENT_UPLOAD_TAB));
    expect(result.current.activeTab).toBe(PATIENT_UPLOAD_TAB);

    act(() => result.current.selectTab("symptoms"));
    expect(result.current.activeTab).toBe(PATIENT_UPLOAD_TAB);

    act(() => result.current.resetTab());
    expect(result.current.activeTab).toBe(PATIENT_PROFILE_TAB);
  });
});

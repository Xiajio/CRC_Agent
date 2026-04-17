import { useState } from "react";
import type { Dispatch, SetStateAction } from "react";

export type DoctorTab = "consultation" | "database";
export type DoctorDatabaseSource = "historical_case_base" | "patient_registry";

export type DoctorViewState = {
  activeDoctorTab: DoctorTab;
  setActiveDoctorTab: Dispatch<SetStateAction<DoctorTab>>;
  activeDatabaseSource: DoctorDatabaseSource;
  setActiveDatabaseSource: Dispatch<SetStateAction<DoctorDatabaseSource>>;
};

export function useDoctorViewState(): DoctorViewState {
  const [activeDoctorTab, setActiveDoctorTab] = useState<DoctorTab>("consultation");
  const [activeDatabaseSource, setActiveDatabaseSource] =
    useState<DoctorDatabaseSource>("historical_case_base");

  return {
    activeDoctorTab,
    setActiveDoctorTab,
    activeDatabaseSource,
    setActiveDatabaseSource,
  };
}

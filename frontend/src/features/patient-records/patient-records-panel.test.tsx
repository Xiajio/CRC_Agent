import { render, screen } from "@testing-library/react";

import { PatientRecordsPanel } from "./patient-records-panel";

test("renders patient triage records in the profile tab", () => {
  render(
    <PatientRecordsPanel
      records={[
        {
          record_id: 8,
          patient_id: 101,
          asset_id: 1,
          record_type: "crc_triage_assessment",
          document_type: "crc_triage_assessment",
          ingest_decision: "record_only",
          snapshot_contributed: false,
          conflict_detected: false,
          summary_text: "建议尽快消化专科评估。",
          source: "patient_generated",
          created_at: "2026-06-25T08:00:00Z",
        },
      ]}
      isLoading={false}
    />,
  );

  expect(screen.getByText("历史问诊记录")).toBeInTheDocument();
  expect(screen.getByText("建议尽快消化专科评估。")).toBeInTheDocument();
  expect(screen.getByText("CRC 专项问诊")).toBeInTheDocument();
});

test("renders an empty patient records state", () => {
  render(<PatientRecordsPanel records={[]} isLoading={false} />);

  expect(screen.getByText("当前暂无历史问诊记录")).toBeInTheDocument();
});

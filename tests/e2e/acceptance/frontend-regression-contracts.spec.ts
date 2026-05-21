import { expect, test, type Page, type Route } from "@playwright/test";

const PATIENT_SESSION_ID = "patient-session";
const DOCTOR_SESSION_ID = "doctor-session";
const DOCTOR_THINKING = "DOCTOR_INTERNAL_REASONING_VISIBLE";
const PATIENT_THINKING = "PATIENT_INTERNAL_REASONING_HIDDEN";
const TREATMENT_PLAN_LABEL = "\u751f\u6210\u6cbb\u7597\u65b9\u6848";

type Scene = "patient" | "doctor";

function baseSnapshot(overrides: Record<string, unknown> = {}) {
  return {
    snapshot_version: 1,
    messages: [],
    messages_total: 0,
    messages_next_before_cursor: null,
    cards: [],
    roadmap: [],
    findings: {},
    patient_profile: null,
    patient_identity: null,
    stage: null,
    assessment_draft: null,
    case_database_patient_id: null,
    registry_patient_id: null,
    current_patient_id: null,
    references: [],
    plan: [],
    critic: null,
    safety_alert: null,
    uploaded_assets: {},
    context_maintenance: null,
    context_state: null,
    ...overrides,
  };
}

function sessionResponse(scene: Scene, snapshotOverrides: Record<string, unknown> = {}) {
  const sessionId = scene === "patient" ? PATIENT_SESSION_ID : DOCTOR_SESSION_ID;
  return {
    session_id: sessionId,
    thread_id: `${scene}-thread`,
    scene,
    patient_id: scene === "doctor" ? 7 : null,
    snapshot_version: 1,
    snapshot: baseSnapshot(snapshotOverrides),
    runtime: {
      runner_mode: "fixture",
      fixture_case: "frontend-regression-contracts",
    },
  };
}

const doctorPatientCard = {
  card_type: "patient_card",
  payload: {
    type: "patient_card",
    patient_id: "093",
    data: {
      patient_id: "093",
      patient_info: {
        age: 61,
        gender: "\u7537",
      },
      diagnosis_block: {
        confirmed: "CRC",
        primary_site: "rectum",
        mmr_status: "pMMR",
      },
      staging_block: {
        clinical_stage: "III",
      },
    },
  },
  source_channel: "state",
};

function responseJson(payload: unknown) {
  return {
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(payload),
  };
}

async function fulfillSse(route: Route, events: unknown[]) {
  await route.fulfill({
    status: 200,
    contentType: "text/event-stream",
    headers: {
      "Cache-Control": "no-cache",
    },
    body: events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""),
  });
}

async function installWorkspaceMocks(page: Page) {
  await page.addInitScript(() => {
    window.localStorage.clear();
  });

  await page.route("**/api/sessions", async (route) => {
    const request = route.request();
    if (request.method() !== "POST") {
      await route.fallback();
      return;
    }

    const body = request.postDataJSON() as { scene?: Scene };
    if (body.scene === "doctor") {
      await route.fulfill(responseJson(sessionResponse("doctor", {
        registry_patient_id: 7,
        case_database_patient_id: "093",
        cards: [doctorPatientCard],
      })));
      return;
    }

    await route.fulfill(responseJson(sessionResponse("patient")));
  });

  await page.route(/\/api\/sessions\/[^/?]+(?:\?.*)?$/, async (route) => {
    const url = route.request().url();
    if (route.request().method() !== "GET") {
      await route.fallback();
      return;
    }

    if (url.includes(`/${DOCTOR_SESSION_ID}`)) {
      await route.fulfill(responseJson(sessionResponse("doctor", {
        registry_patient_id: 7,
        case_database_patient_id: "093",
        cards: [doctorPatientCard],
      })));
      return;
    }

    await route.fulfill(responseJson(sessionResponse("patient")));
  });

  await page.route(`**/api/sessions/${PATIENT_SESSION_ID}/messages/stream`, async (route) => {
    await fulfillSse(route, [
      {
        type: "message.done",
        role: "assistant",
        content: "Patient-facing final answer.",
        thinking: PATIENT_THINKING,
        message_id: "patient-answer-1",
      },
      {
        type: "done",
        thread_id: "patient-thread",
        run_id: "patient-run-1",
        snapshot_version: 2,
      },
    ]);
  });

  await page.route(`**/api/sessions/${DOCTOR_SESSION_ID}/messages/stream`, async (route) => {
    await fulfillSse(route, [
      {
        type: "message.done",
        role: "assistant",
        content: "Doctor-facing final answer.",
        thinking: DOCTOR_THINKING,
        message_id: "doctor-answer-1",
      },
      {
        type: "done",
        thread_id: "doctor-thread",
        run_id: "doctor-run-1",
        snapshot_version: 2,
      },
    ]);
  });

  await page.route("**/api/patient-registry/patients/recent?limit=20", async (route) => {
    await route.fulfill(responseJson({ items: [], total: 0 }));
  });
  await page.route("**/api/patient-registry/patients/7", async (route) => {
    await route.fulfill(responseJson({
      patient_id: 7,
      status: "active",
      created_at: "2026-05-13T00:00:00Z",
      updated_at: "2026-05-13T00:00:00Z",
      age: 61,
      gender: "\u7537",
      tumor_location: "rectum",
      mmr_status: "pMMR",
      clinical_stage: "III",
    }));
  });
  await page.route("**/api/patient-registry/patients/7/records", async (route) => {
    await route.fulfill(responseJson({ items: [] }));
  });
  await page.route("**/api/patient-registry/patients/7/alerts", async (route) => {
    await route.fulfill(responseJson({ items: [] }));
  });
  await page.route("**/api/database/stats", async (route) => {
    await route.fulfill(responseJson({
      total_cases: 0,
      gender_distribution: {},
      tumor_location_distribution: {},
      ct_stage_distribution: {},
      mmr_status_distribution: {},
    }));
  });
  await page.route("**/api/database/cases/search", async (route) => {
    await route.fulfill(responseJson({
      items: [],
      total: 0,
      page: 1,
      page_size: 20,
      applied_filters: {},
      warnings: [],
    }));
  });
}

async function openWorkspace(page: Page) {
  await installWorkspaceMocks(page);
  await page.goto("/");
  await expect(page.getByTestId("conversation-panel")).toBeVisible();
}

async function submitCurrentComposer(page: Page, prompt: string) {
  const composer = page.getByTestId("conversation-panel").locator("textarea");
  await composer.fill(prompt);
  await composer.press("Enter");
}

test("keeps thinking hidden for patient replies while doctor replies disclose it", async ({ page }) => {
  await openWorkspace(page);

  await submitCurrentComposer(page, "patient prompt");
  await expect(page.getByText("Patient-facing final answer.")).toBeVisible();
  await expect(page.locator(".clinical-thinking-disclosure")).toHaveCount(0);
  await expect(page.getByText(PATIENT_THINKING)).toHaveCount(0);

  await page.getByRole("button", { name: "医生场景" }).click();
  await expect(page.getByRole("button", { name: "患者场景" })).toBeVisible();

  await submitCurrentComposer(page, "doctor prompt");
  await expect(page.getByText("Doctor-facing final answer.")).toBeVisible();
  const thinkingDisclosure = page.locator(".clinical-thinking-disclosure");
  await expect(thinkingDisclosure).toHaveCount(1);
  await thinkingDisclosure.locator("summary").click();
  await expect(thinkingDisclosure).toContainText(DOCTOR_THINKING);
});

test("sends split identity context when doctor card prompt is clicked", async ({ page }) => {
  await openWorkspace(page);
  await page.getByRole("button", { name: "医生场景" }).click();
  await expect(page.getByRole("button", { name: "患者场景" })).toBeVisible();

  const streamRequest = page.waitForRequest((request) =>
    request.method() === "POST"
    && request.url().includes(`/api/sessions/${DOCTOR_SESSION_ID}/messages/stream`),
  );

  await page.getByRole("button", { name: TREATMENT_PLAN_LABEL }).click();
  const payload = (await streamRequest).postDataJSON() as {
    context?: Record<string, unknown>;
  };

  expect(payload.context).toEqual({
    registry_patient_id: 7,
    case_database_patient_id: "093",
  });
});

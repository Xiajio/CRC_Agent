import { expect, test, type Page, type Route } from "@playwright/test";

type Scene = "patient" | "doctor";

const ACTIVE_SCENE_STORAGE_KEY = "langg.workspace.active-scene";
const PATIENT_SESSION_ID = "visual-patient-session";
const DOCTOR_SESSION_ID = "visual-doctor-session";

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
    patient_id: scene === "doctor" ? 93 : null,
    snapshot_version: 1,
    snapshot: baseSnapshot(snapshotOverrides),
    runtime: {
      runner_mode: "fixture",
      fixture_case: "clinical-style-architecture",
    },
  };
}

function responseJson(payload: unknown) {
  return {
    status: 200,
    contentType: "application/json",
    body: JSON.stringify(payload),
  };
}

async function installVisualMocks(page: Page) {
  await page.route("**/api/sessions", async (route) => {
    const request = route.request();
    if (request.method() !== "POST") {
      await route.fallback();
      return;
    }

    const body = request.postDataJSON() as { scene?: Scene };
    if (body.scene === "doctor") {
      await route.fulfill(responseJson(sessionResponse("doctor", {
        registry_patient_id: 93,
        case_database_patient_id: "093",
        cards: [
          {
            card_type: "patient_card",
            payload: {
              type: "patient_card",
              patient_id: "093",
              data: {
                patient_id: "093",
                patient_info: { age: 61, gender: "male" },
                diagnosis_block: { confirmed: "CRC", primary_site: "rectum", mmr_status: "pMMR" },
                staging_block: { clinical_stage: "III" },
              },
            },
            source_channel: "state",
          },
        ],
      })));
      return;
    }

    await route.fulfill(responseJson(sessionResponse("patient")));
  });

  await page.route(/\/api\/sessions\/[^/?]+(?:\?.*)?$/, async (route) => {
    const request = route.request();
    if (request.method() !== "GET") {
      await route.fallback();
      return;
    }

    const url = request.url();
    if (url.includes(`/${DOCTOR_SESSION_ID}`)) {
      await route.fulfill(responseJson(sessionResponse("doctor", {
        registry_patient_id: 93,
        case_database_patient_id: "093",
      })));
      return;
    }

    await route.fulfill(responseJson(sessionResponse("patient")));
  });

  await page.route("**/api/patient-registry/patients/recent?limit=20", async (route) => {
    await route.fulfill(responseJson({ items: [], total: 0 }));
  });
  await page.route("**/api/patient-registry/patients/93", async (route) => {
    await route.fulfill(responseJson({
      patient_id: 93,
      status: "active",
      created_at: "2026-06-01T00:00:00Z",
      updated_at: "2026-06-01T00:00:00Z",
      age: 61,
      gender: "male",
      tumor_location: "rectum",
      mmr_status: "pMMR",
      clinical_stage: "III",
    }));
  });
  await page.route("**/api/patient-registry/patients/93/records", async (route) => {
    await route.fulfill(responseJson({ items: [] }));
  });
  await page.route("**/api/patient-registry/patients/93/alerts", async (route) => {
    await route.fulfill(responseJson({ items: [] }));
  });
  await page.route("**/api/database/stats", async (route) => {
    await route.fulfill(responseJson({
      total_cases: 24,
      age_statistics: { min: 42, max: 78, mean: 61.4 },
      gender_distribution: { male: 14, female: 10 },
      tumor_location_distribution: { rectum: 15, sigmoid: 6, ascending: 3 },
      ct_stage_distribution: { III: 12, II: 8, IV: 4 },
      mmr_status_distribution: { pMMR: 18, dMMR: 6 },
    }));
  });
  await page.route("**/api/database/cases/search", async (route) => {
    await route.fulfill(responseJson({
      items: [
        {
          patient_id: 93,
          age: 61,
          gender: "male",
          tumor_location: "rectum",
          ct_stage: "III",
          mmr_status: "pMMR",
          ecog_score: 1,
          chief_complaint: "intermittent hematochezia",
          updated_at: "2026-06-01T00:00:00Z",
        },
      ],
      total: 1,
      page: 1,
      page_size: 20,
      applied_filters: {},
      warnings: [],
    }));
  });
}

async function prepareVisualPage(page: Page, scene: Scene = "patient") {
  await installVisualMocks(page);
  await page.addInitScript(
    ([activeSceneStorageKey, activeScene]) => {
      window.localStorage.clear();
      window.localStorage.setItem(activeSceneStorageKey, activeScene);
    },
    [ACTIVE_SCENE_STORAGE_KEY, scene],
  );
}

async function stabilize(page: Page) {
  await page.evaluate(() => document.fonts.ready);
  await page.locator(".ui-top-nav, .clinical-top-nav").first().waitFor({ state: "visible" });
}

test("doctor workspace desktop visual baseline", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 980 });
  await prepareVisualPage(page, "doctor");
  await page.goto("/");

  await expect(page.getByTestId("doctor-scene")).toBeVisible();
  await expect(page.locator(".clinical-dashboard")).toBeVisible();
  await stabilize(page);

  await expect(page.locator(".clinical-app-shell").first()).toHaveScreenshot(
    "doctor-workspace-desktop.png",
    {
      animations: "disabled",
      maxDiffPixelRatio: 0.03,
    },
  );
});

test("patient workspace mobile visual baseline", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 900 });
  await prepareVisualPage(page, "patient");
  await page.goto("/");

  await expect(page.getByTestId("workspace-layout")).toBeVisible();
  await expect(page.getByTestId("conversation-panel")).toBeVisible();
  await stabilize(page);

  await expect(page.locator(".clinical-app-shell-patient")).toHaveScreenshot(
    "patient-workspace-mobile.png",
    {
      animations: "disabled",
      maxDiffPixelRatio: 0.03,
    },
  );
});

test("database workspace tablet visual baseline", async ({ page }) => {
  await page.setViewportSize({ width: 900, height: 1100 });
  await prepareVisualPage(page, "doctor");
  await page.goto("/database");

  await expect(page.getByTestId("panel-grid")).toBeVisible();
  await expect(page.getByTestId("database-workbench")).toBeVisible();
  await stabilize(page);

  await expect(page.locator(".ui-app-shell").first()).toHaveScreenshot(
    "database-workspace-tablet.png",
    {
      animations: "disabled",
      maxDiffPixelRatio: 0.03,
    },
  );
});

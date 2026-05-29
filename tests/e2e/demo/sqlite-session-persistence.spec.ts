import { expect, test, type Page } from "@playwright/test";
import path from "node:path";

const PATIENT_SESSION_STORAGE_KEY = "langg.workspace.patient-session-id";
const DOCTOR_SESSION_STORAGE_KEY = "langg.workspace.doctor-session-id";
const PROMPT = "sqlite persistence browser probe";
const UPLOAD_FILENAME = "demo_colonoscopy_report.pdf";

const repoRoot = path.basename(process.cwd()) === "frontend"
  ? path.resolve(process.cwd(), "..")
  : process.cwd();

type StoredSessions = {
  patient: string | null;
  doctor: string | null;
};

type UploadPayload = {
  asset_id: string | number;
  asset_url: string;
};

type SessionPayload = {
  session_id: string;
  snapshot?: {
    messages?: Array<{ content?: unknown }>;
    uploaded_assets?: Record<string, { filename?: unknown }>;
  };
};

async function readStoredSessions(page: Page): Promise<StoredSessions> {
  return page.evaluate(
    ([patientKey, doctorKey]) => ({
      patient: window.localStorage.getItem(patientKey),
      doctor: window.localStorage.getItem(doctorKey),
    }),
    [PATIENT_SESSION_STORAGE_KEY, DOCTOR_SESSION_STORAGE_KEY],
  );
}

function isApiPath(responseOrRequestUrl: string, pathname: string): boolean {
  return new URL(responseOrRequestUrl).pathname === pathname;
}

function assetRequestOptions() {
  const bearerToken = (process.env.VITE_API_BEARER_TOKEN ?? process.env.API_BEARER_TOKEN ?? "").trim();
  return bearerToken ? { headers: { Authorization: `Bearer ${bearerToken}` } } : undefined;
}

test("restores a sqlite-backed browser session after reload without the recovery banner", async ({ page }) => {
  test.skip(
    process.env.SESSION_STORE_BACKEND !== "sqlite",
    "requires SESSION_STORE_BACKEND=sqlite so the browser reload exercises the SQLite session store",
  );

  await page.goto("/");
  await expect(page.getByTestId("conversation-panel")).toBeVisible();
  await expect(page.getByTestId("session-recovery-banner")).toHaveCount(0);

  const storedBefore = await readStoredSessions(page);
  expect(storedBefore.patient).toMatch(/^sess_/);
  expect(storedBefore.doctor).toMatch(/^sess_/);

  const patientSessionId = storedBefore.patient!;
  const streamResponsePromise = page.waitForResponse((response) =>
    response.request().method() === "POST"
    && isApiPath(response.url(), `/api/sessions/${patientSessionId}/messages/stream`),
  );
  await page.getByTestId("conversation-input").fill(PROMPT);
  await page.getByTestId("conversation-input").press("Enter");
  await streamResponsePromise;
  await expect(page.getByTestId("conversation-input")).toBeEnabled();
  await expect(page.getByText(PROMPT)).toBeVisible();

  const uploadResponsePromise = page.waitForResponse((response) =>
    response.request().method() === "POST"
    && isApiPath(response.url(), `/api/sessions/${patientSessionId}/uploads`),
  );
  await page
    .getByTestId("upload-input")
    .setInputFiles(path.join(repoRoot, "tests", "fixtures", "demo_uploads", UPLOAD_FILENAME));
  const uploadResponse = await uploadResponsePromise;
  expect(uploadResponse.ok()).toBeTruthy();
  const uploadPayload = await uploadResponse.json() as UploadPayload;
  const assetId = String(uploadPayload.asset_id);
  await expect(page.getByText(UPLOAD_FILENAME)).toBeVisible();

  const createdSessionsAfterReload: string[] = [];
  page.on("request", (request) => {
    if (request.method() === "POST" && isApiPath(request.url(), "/api/sessions")) {
      createdSessionsAfterReload.push(request.postData() ?? "");
    }
  });

  const apiBaseUrl = process.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";
  const restoredSessionPromise = page.waitForResponse((response) =>
    response.request().method() === "GET"
    && isApiPath(response.url(), `/api/sessions/${patientSessionId}`),
  );
  await page.reload();
  const restoredSessionResponse = await restoredSessionPromise;
  expect(restoredSessionResponse.ok()).toBeTruthy();
  const restoredSessionApiResponse = await page.request.get(
    new URL(`/api/sessions/${patientSessionId}`, apiBaseUrl).toString(),
    assetRequestOptions(),
  );
  expect(restoredSessionApiResponse.ok()).toBeTruthy();
  const restoredSession = await restoredSessionApiResponse.json() as SessionPayload;

  await expect(page.getByTestId("conversation-panel")).toBeVisible();
  await expect(page.getByTestId("session-recovery-banner")).toHaveCount(0);
  await expect(page.getByText(PROMPT)).toBeVisible();
  await expect(page.getByTestId(`uploaded-asset-${assetId}`)).toBeVisible();

  expect(await readStoredSessions(page)).toEqual(storedBefore);
  expect(createdSessionsAfterReload).toEqual([]);
  expect(restoredSession.session_id).toBe(patientSessionId);
  expect(restoredSession.snapshot?.messages?.some((message) =>
    String(message.content ?? "").includes(PROMPT),
  )).toBeTruthy();
  expect(restoredSession.snapshot?.uploaded_assets?.[assetId]).toMatchObject({
    filename: UPLOAD_FILENAME,
  });

  const assetResponse = await page.request.get(
    new URL(uploadPayload.asset_url, apiBaseUrl).toString(),
    assetRequestOptions(),
  );
  expect(assetResponse.ok()).toBeTruthy();
  expect((await assetResponse.body()).byteLength).toBeGreaterThan(0);
});

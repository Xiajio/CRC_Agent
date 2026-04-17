import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "../tests/e2e/acceptance",
  outputDir: "../output/acceptance",
  fullyParallel: false,
  timeout: 60_000,
  expect: {
    timeout: 15_000,
  },
  projects: [
    {
      name: "chromium",
      use: {
        browserName: "chromium",
        channel: "msedge",
        baseURL: "http://127.0.0.1:4173",
        trace: "on-first-retry",
      },
    },
  ],
  use: {
    baseURL: "http://127.0.0.1:4173",
    trace: "on-first-retry",
  },
  webServer: [
    {
      command: "powershell -NoProfile -ExecutionPolicy Bypass -File ../scripts/start_backend_acceptance_fixture.ps1",
      url: "http://127.0.0.1:8000/openapi.json",
      reuseExistingServer: false,
      timeout: 120_000,
    },
    {
      command: "powershell -NoProfile -ExecutionPolicy Bypass -File ../scripts/start_frontend.ps1",
      url: "http://127.0.0.1:4173",
      reuseExistingServer: true,
      timeout: 120_000,
    },
  ],
});

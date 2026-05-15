import { defineConfig } from "@playwright/test";
import Module from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const configDir = path.dirname(fileURLToPath(import.meta.url));
const frontendNodeModules = path.join(configDir, "node_modules");
const nodePathEntries = (process.env.NODE_PATH ?? "").split(path.delimiter).filter(Boolean);

if (!nodePathEntries.includes(frontendNodeModules)) {
  process.env.NODE_PATH = [frontendNodeModules, ...nodePathEntries].join(path.delimiter);
  (Module as unknown as { _initPaths: () => void })._initPaths();
}

export default defineConfig({
  testDir: "../tests/e2e",
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
      command: "powershell -NoProfile -ExecutionPolicy Bypass -File ../scripts/start_backend_fixture.ps1",
      url: "http://127.0.0.1:8000/openapi.json",
      reuseExistingServer: true,
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

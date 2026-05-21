import { defineConfig } from "@playwright/test";
import Module from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const configDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(configDir, "..");
const frontendNodeModules = path.join(configDir, "node_modules");
const nodePathEntries = (process.env.NODE_PATH ?? "").split(path.delimiter).filter(Boolean);

if (!nodePathEntries.includes(frontendNodeModules)) {
  process.env.NODE_PATH = [frontendNodeModules, ...nodePathEntries].join(path.delimiter);
  (Module as unknown as { _initPaths: () => void })._initPaths();
}

const demoWebServer =
  process.env.PLAYWRIGHT_DEMO_SKIP_WEBSERVER === "1"
    ? undefined
    : [
        {
          command: "node scripts/playwright_demo_server.cjs backend",
          cwd: repoRoot,
          url: "http://127.0.0.1:8000/openapi.json",
          reuseExistingServer: true,
          timeout: 120_000,
        },
        {
          command: "node scripts/playwright_demo_server.cjs frontend",
          cwd: repoRoot,
          url: "http://127.0.0.1:4173",
          reuseExistingServer: true,
          timeout: 120_000,
        },
      ];

export default defineConfig({
  testDir: "../tests/e2e",
  outputDir: "test-results/demo",
  fullyParallel: false,
  timeout: 90_000,
  expect: {
    timeout: 20_000,
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
  webServer: demoWebServer,
});

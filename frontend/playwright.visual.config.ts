import { defineConfig } from "@playwright/test";
import Module from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const configDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(configDir, "..");
const frontendNodeModules = path.join(configDir, "node_modules");
const nodePathEntries = (process.env.NODE_PATH ?? "").split(path.delimiter).filter(Boolean);
const backendPort = process.env.LANGG_VISUAL_BACKEND_PORT || process.env.LANGG_BACKEND_PORT || "8010";
const frontendPort = process.env.LANGG_VISUAL_FRONTEND_PORT || process.env.LANGG_FRONTEND_PORT || "4176";
const visualServerEnv = {
  LANGG_BACKEND_PORT: backendPort,
  LANGG_FRONTEND_PORT: frontendPort,
  LANGG_RUNTIME_ROOT: "runtime/visual",
  GRAPH_FIXTURE_CASE: "database_case",
  VITE_DEMO_MODE: "replay",
};

if (!nodePathEntries.includes(frontendNodeModules)) {
  process.env.NODE_PATH = [frontendNodeModules, ...nodePathEntries].join(path.delimiter);
  (Module as unknown as { _initPaths: () => void })._initPaths();
}

const visualWebServer =
  process.env.PLAYWRIGHT_VISUAL_SKIP_WEBSERVER === "1"
    ? undefined
    : [
        {
          command: "node scripts/playwright_demo_server.cjs backend",
          cwd: repoRoot,
          env: visualServerEnv,
          url: `http://127.0.0.1:${backendPort}/openapi.json`,
          reuseExistingServer: false,
          timeout: 120_000,
        },
        {
          command: "node scripts/playwright_demo_server.cjs frontend",
          cwd: repoRoot,
          env: visualServerEnv,
          url: `http://127.0.0.1:${frontendPort}`,
          reuseExistingServer: false,
          timeout: 120_000,
        },
      ];

export default defineConfig({
  testDir: "../tests/e2e/visual",
  outputDir: "../output/visual",
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
        baseURL: `http://127.0.0.1:${frontendPort}`,
        trace: "on-first-retry",
      },
    },
  ],
  use: {
    baseURL: `http://127.0.0.1:${frontendPort}`,
    trace: "on-first-retry",
  },
  webServer: visualWebServer,
});

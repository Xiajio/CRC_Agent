import { defineConfig } from "@playwright/test";
import Module from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const configDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(configDir, "..");
const frontendNodeModules = path.join(configDir, "node_modules");
const nodePathEntries = (process.env.NODE_PATH ?? "").split(path.delimiter).filter(Boolean);
const backendPort = process.env.LANGG_BACKEND_PORT ?? "8010";
const frontendPort = process.env.LANGG_FRONTEND_PORT ?? "4174";
const backendUrl = `http://127.0.0.1:${backendPort}`;
const frontendUrl = `http://127.0.0.1:${frontendPort}`;
const sqliteSessionWebServer =
  process.env.PLAYWRIGHT_SQLITE_SESSION_SKIP_WEBSERVER === "1"
    ? undefined
    : [
        {
          command: "node scripts/playwright_demo_server.cjs backend",
          cwd: repoRoot,
          url: `${backendUrl}/openapi.json`,
          reuseExistingServer: false,
          timeout: 120_000,
        },
        {
          command: "node scripts/playwright_demo_server.cjs frontend",
          cwd: repoRoot,
          url: frontendUrl,
          reuseExistingServer: false,
          timeout: 120_000,
        },
      ];

if (!nodePathEntries.includes(frontendNodeModules)) {
  process.env.NODE_PATH = [frontendNodeModules, ...nodePathEntries].join(path.delimiter);
  (Module as unknown as { _initPaths: () => void })._initPaths();
}

export default defineConfig({
  testDir: "../tests/e2e/demo",
  outputDir: "test-results/sqlite-session-persistence",
  fullyParallel: false,
  timeout: 120_000,
  expect: {
    timeout: 20_000,
  },
  projects: [
    {
      name: "chromium",
      use: {
        browserName: "chromium",
        channel: "msedge",
        baseURL: frontendUrl,
        trace: "on-first-retry",
      },
    },
  ],
  use: {
    baseURL: frontendUrl,
    trace: "on-first-retry",
  },
  webServer: sqliteSessionWebServer,
});

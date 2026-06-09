const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const { setTimeout: delay } = require("node:timers/promises");

const repoRoot = path.resolve(__dirname, "..");
const frontendRoot = path.join(repoRoot, "frontend");
const serverScript = path.join(repoRoot, "scripts", "playwright_demo_server.cjs");
const playwrightCli = path.join(frontendRoot, "node_modules", "@playwright", "test", "cli.js");
const backendPort = process.env.LANGG_VISUAL_BACKEND_PORT || "8010";
const frontendPort = process.env.LANGG_VISUAL_FRONTEND_PORT || "4176";
const backendReadyUrl = `http://127.0.0.1:${backendPort}/openapi.json`;
const frontendReadyUrl = `http://127.0.0.1:${frontendPort}`;

function killProcessTree(pid) {
  if (!pid) return;

  if (process.platform === "win32") {
    const killer = spawn("taskkill", ["/pid", String(pid), "/T", "/F"], {
      stdio: "ignore",
      windowsHide: true,
    });
    killer.unref();
    return;
  }

  try {
    process.kill(pid, "SIGTERM");
  } catch {
    // The server process may already have exited.
  }
}

async function isUrlReady(url) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 1_000);
  try {
    const response = await fetch(url, { signal: controller.signal });
    await response.body?.cancel();
    return response.status < 500;
  } catch {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}

async function waitForUrl(url, timeoutMs = 120_000) {
  const deadline = Date.now() + timeoutMs;

  while (Date.now() <= deadline) {
    if (await isUrlReady(url)) {
      return;
    }
    await delay(500);
  }

  throw new Error(`Timed out waiting for ${url}`);
}

async function runPlaywright(extraArgs, env) {
  if (!fs.existsSync(playwrightCli)) {
    throw new Error(`Playwright CLI not found at ${playwrightCli}. Run npm install in frontend first.`);
  }

  const args = ["test", "--config", "playwright.visual.config.ts", ...extraArgs];

  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [playwrightCli, ...args], {
      cwd: frontendRoot,
      stdio: "inherit",
      env,
      windowsHide: true,
    });
    const timeout = setTimeout(() => {
      killProcessTree(child.pid);
      reject(new Error("Timed out waiting for Playwright exit code"));
    }, 180_000);

    child.on("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    child.on("exit", (code, signal) => {
      clearTimeout(timeout);
      resolve(code ?? (signal ? 1 : 0));
    });
  });
}

async function main() {
  if (await isUrlReady(backendReadyUrl)) {
    throw new Error(`Visual backend port is already in use: ${backendReadyUrl}`);
  }
  if (await isUrlReady(frontendReadyUrl)) {
    throw new Error(`Visual frontend port is already in use: ${frontendReadyUrl}`);
  }

  const serverEnv = {
    ...process.env,
    LANGG_BACKEND_PORT: backendPort,
    LANGG_FRONTEND_PORT: frontendPort,
    LANGG_RUNTIME_ROOT: process.env.LANGG_RUNTIME_ROOT || "runtime/visual",
    GRAPH_FIXTURE_CASE: process.env.GRAPH_FIXTURE_CASE || "database_case",
    VITE_DEMO_MODE: "replay",
  };

  const servers = [
    spawn(process.execPath, [serverScript, "backend"], {
      cwd: repoRoot,
      env: serverEnv,
      stdio: "inherit",
      windowsHide: true,
    }),
    spawn(process.execPath, [serverScript, "frontend"], {
      cwd: repoRoot,
      env: serverEnv,
      stdio: "inherit",
      windowsHide: true,
    }),
  ];

  const cleanup = () => {
    for (const server of [...servers].reverse()) {
      killProcessTree(server.pid);
      server.unref();
    }
  };

  for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
    process.on(signal, () => {
      cleanup();
      process.exit(0);
    });
  }

  try {
    await waitForUrl(backendReadyUrl);
    await waitForUrl(frontendReadyUrl);

    const status = await runPlaywright(process.argv.slice(2), {
      ...process.env,
      LANGG_BACKEND_PORT: backendPort,
      LANGG_FRONTEND_PORT: frontendPort,
      LANGG_VISUAL_BACKEND_PORT: backendPort,
      LANGG_VISUAL_FRONTEND_PORT: frontendPort,
      PLAYWRIGHT_VISUAL_SKIP_WEBSERVER: "1",
    });
    process.exitCode = status;
  } finally {
    cleanup();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

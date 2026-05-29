const { spawn, spawnSync } = require("node:child_process");
const path = require("node:path");
const { setTimeout: delay } = require("node:timers/promises");

const repoRoot = path.resolve(__dirname, "..");
const frontendRoot = path.join(repoRoot, "frontend");
const playwrightCli = path.join(frontendRoot, "node_modules", "@playwright", "test", "cli.js");
const viteCli = path.join(frontendRoot, "node_modules", "vite", "bin", "vite.js");
const pythonExe = process.env.LANGG_PYTHON || "D:\\anaconda3\\envs\\LangG\\python.exe";
const backendPort = process.env.LANGG_BACKEND_PORT || "8010";
const frontendPort = process.env.LANGG_FRONTEND_PORT || "4174";
const backendReadyUrl = `http://127.0.0.1:${backendPort}/openapi.json`;
const frontendReadyUrl = `http://127.0.0.1:${frontendPort}`;

const env = {
  ...process.env,
  LANGG_RUNTIME_ROOT: process.env.LANGG_RUNTIME_ROOT || "runtime/sqlite-session-persistence",
  LANGG_BACKEND_PORT: backendPort,
  LANGG_FRONTEND_PORT: frontendPort,
  AUTH_MODE: "none",
  GRAPH_RUNNER_MODE: "fixture",
  GRAPH_FIXTURE_CASE: process.env.GRAPH_FIXTURE_CASE || "demo_doctor_decision",
  RAG_WARMUP: "false",
  FRONTEND_ORIGINS: `http://127.0.0.1:${frontendPort}`,
  SESSION_STORE_BACKEND: "sqlite",
  SESSION_STORE_TTL_DAYS: "none",
  UPLOAD_CONVERTER_MODE: "fixture",
  VITE_API_BASE_URL: `http://127.0.0.1:${backendPort}`,
  VITE_DEMO_MODE: "replay",
};

async function stopChild(child) {
  if (!child?.pid || child.exitCode !== null || child.signalCode !== null) return;

  const exited = new Promise((resolve) => child.once("exit", resolve));

  try {
    child.kill();
  } catch {
    return;
  }

  const stopped = await Promise.race([
    exited.then(() => true),
    delay(2_000).then(() => false),
  ]);

  if (stopped || process.platform !== "win32") return;

  spawnSync("taskkill", ["/pid", String(child.pid), "/T", "/F"], {
    stdio: "ignore",
    windowsHide: true,
  });

  await Promise.race([
    exited,
    delay(2_000),
  ]);
}

function spawnBackend() {
  return spawn(pythonExe, [
    "-m",
    "uvicorn",
    "backend.app:app",
    "--host",
    "127.0.0.1",
    "--port",
    backendPort,
  ], {
    cwd: repoRoot,
    env,
    stdio: "inherit",
    windowsHide: true,
  });
}

function spawnFrontend() {
  return spawn(process.execPath, [
    viteCli,
    "--host",
    "127.0.0.1",
    "--port",
    frontendPort,
    "--strictPort",
  ], {
    cwd: frontendRoot,
    env,
    stdio: "inherit",
    windowsHide: true,
  });
}

function failFastOnExit(child, label) {
  child.on("exit", (code, signal) => {
    if (code === null && signal === null) return;
    if (code === 0 || child.killed) return;
    console.error(`[sqlite-session] ${label} exited early with ${signal ?? code}`);
    process.exitCode = code ?? 1;
  });
  child.on("error", (error) => {
    console.error(error);
    process.exitCode = 1;
  });
}

async function waitForUrl(url, timeoutMs = 120_000) {
  const deadline = Date.now() + timeoutMs;

  while (Date.now() <= deadline) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2_000);
    try {
      const response = await fetch(url, { signal: controller.signal });
      if (response.status < 500) {
        await response.body?.cancel();
        return;
      }
      await response.body?.cancel();
    } catch {
      // Retry until ready or timed out.
    } finally {
      clearTimeout(timeout);
    }
    await delay(500);
  }

  throw new Error(`Timed out waiting for ${url}`);
}

function runPlaywright() {
  const args = [
    playwrightCli,
    "test",
    "--config",
    "playwright.sqlite-session.config.ts",
    "sqlite-session-persistence.spec.ts",
  ];

  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, args, {
      cwd: frontendRoot,
      env: {
        ...env,
        PLAYWRIGHT_SQLITE_SESSION_SKIP_WEBSERVER: "1",
      },
      stdio: "inherit",
      windowsHide: true,
    });

    child.on("exit", (code, signal) => {
      resolve(code ?? (signal ? 1 : 0));
    });

    child.on("error", reject);
  });
}

async function main() {
  const servers = [spawnBackend(), spawnFrontend()];
  failFastOnExit(servers[0], "backend");
  failFastOnExit(servers[1], "frontend");

  const cleanup = async () => {
    for (const server of [...servers].reverse()) {
      await stopChild(server);
    }
  };

  for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
    process.on(signal, async () => {
      await cleanup();
      process.exit(0);
    });
  }

  try {
    console.log(`[sqlite-session] Waiting for backend ${backendReadyUrl}`);
    await waitForUrl(backendReadyUrl);
    console.log(`[sqlite-session] Waiting for frontend ${frontendReadyUrl}`);
    await waitForUrl(frontendReadyUrl);
    process.exitCode = await runPlaywright();
  } finally {
    await cleanup();
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

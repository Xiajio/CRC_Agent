const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const { setTimeout: delay } = require("node:timers/promises");

const repoRoot = path.resolve(__dirname, "..");
const frontendRoot = path.join(repoRoot, "frontend");
const serverScript = path.join(repoRoot, "scripts", "playwright_demo_server.cjs");
const playwrightCli = path.join(frontendRoot, "node_modules", "@playwright", "test", "cli.js");

function resolveFrontendBearerToken(env) {
  const viteToken = (env.VITE_API_BEARER_TOKEN || "").trim();
  if (viteToken) return viteToken;
  const apiToken = (env.API_BEARER_TOKEN || "").trim();
  return apiToken || undefined;
}

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
      // Retry until the service is ready or the deadline expires.
    } finally {
      clearTimeout(timeout);
    }
    await delay(500);
  }

  throw new Error(`Timed out waiting for ${url}`);
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

async function runPlaywright(args, options) {
  if (!fs.existsSync(playwrightCli)) {
    throw new Error(`Playwright CLI not found at ${playwrightCli}. Run npm install in frontend first.`);
  }

  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [playwrightCli, ...args], {
      ...options,
      windowsHide: true,
    });
    const timeout = setTimeout(() => {
      killProcessTree(child.pid);
      reject(new Error("Timed out waiting for Playwright exit code"));
    }, 120_000);

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
  const backendReadyUrl = "http://127.0.0.1:8000/openapi.json";
  const frontendReadyUrl = "http://127.0.0.1:4173";
  const frontendBearerToken = resolveFrontendBearerToken(process.env);
  const serverEnv = {
    ...process.env,
    LANGG_RUNTIME_ROOT: process.env.LANGG_RUNTIME_ROOT || "runtime/demo",
    VITE_DEMO_MODE: "replay",
    ...(frontendBearerToken ? { VITE_API_BEARER_TOKEN: frontendBearerToken } : {}),
  };

  const servers = [];
  if (!(await isUrlReady(backendReadyUrl))) {
    servers.push(spawn(process.execPath, [serverScript, "backend"], {
      cwd: repoRoot,
      env: serverEnv,
      stdio: "inherit",
      windowsHide: true,
    }));
  } else {
    console.log(`[demo] Reusing existing backend ${backendReadyUrl}`);
  }

  if (!(await isUrlReady(frontendReadyUrl))) {
    servers.push(spawn(process.execPath, [serverScript, "frontend"], {
      cwd: repoRoot,
      env: serverEnv,
      stdio: "inherit",
      windowsHide: true,
    }));
  } else {
    console.log(`[demo] Reusing existing frontend ${frontendReadyUrl}`);
  }

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
    console.log(`[demo] Waiting for backend ${backendReadyUrl}`);
    await waitForUrl(backendReadyUrl);
    console.log("[demo] Backend ready");
    console.log(`[demo] Waiting for frontend ${frontendReadyUrl}`);
    await waitForUrl(frontendReadyUrl);
    console.log("[demo] Frontend ready");

    const args = [
      "test",
      "--config",
      "playwright.demo.config.ts",
      "demo/demo-mode.spec.ts",
    ];
    if (process.env.DEMO_PLAYWRIGHT_HEADLESS !== "1") {
      args.push("--headed");
    }

    console.log(`[demo] Running ${process.execPath} ${playwrightCli} ${args.join(" ")}`);
    const status = await runPlaywright(args, {
      cwd: frontendRoot,
      stdio: "inherit",
      env: {
        ...process.env,
        PLAYWRIGHT_BASE_URL: process.env.PLAYWRIGHT_BASE_URL || "http://127.0.0.1:4173",
        PLAYWRIGHT_DEMO_SKIP_WEBSERVER: "1",
        VITE_DEMO_MODE: "replay",
        ...(frontendBearerToken ? { VITE_API_BEARER_TOKEN: frontendBearerToken } : {}),
      },
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

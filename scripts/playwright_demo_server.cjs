const { spawn, spawnSync } = require("node:child_process");
const path = require("node:path");

const kind = process.argv[2];
const repoRoot = path.resolve(__dirname, "..");
const powershell =
  process.platform === "win32" && process.env.SystemRoot
    ? path.join(process.env.SystemRoot, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    : "powershell";

function usage() {
  console.error("Usage: node scripts/playwright_demo_server.cjs <backend|frontend>");
  process.exit(2);
}

if (kind !== "backend" && kind !== "frontend") {
  usage();
}

const env = { ...process.env };

let args;
if (kind === "backend") {
  env.LANGG_RUNTIME_ROOT = env.LANGG_RUNTIME_ROOT || "runtime/demo";
  args = [
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    path.join(repoRoot, "scripts", "start_backend_fixture.ps1"),
    "-FixtureCase",
    env.GRAPH_FIXTURE_CASE || "demo_doctor_decision",
    "-UploadConverterFixture",
  ];
} else {
  env.VITE_DEMO_MODE = env.VITE_DEMO_MODE || "replay";
  args = [
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    path.join(repoRoot, "scripts", "start_frontend.ps1"),
  ];
}

const child = spawn(powershell, args, {
  cwd: repoRoot,
  env,
  stdio: "inherit",
  windowsHide: true,
});

let shuttingDown = false;

function killProcessTree(pid) {
  if (!pid) return;

  if (process.platform === "win32") {
    spawnSync("taskkill", ["/pid", String(pid), "/T", "/F"], { stdio: "ignore" });
    return;
  }

  try {
    process.kill(pid, "SIGTERM");
  } catch {
    // The child may already have exited.
  }
}

function shutdown() {
  if (shuttingDown) return;
  shuttingDown = true;
  killProcessTree(child.pid);
  process.exit(0);
}

for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  process.on(signal, shutdown);
}

child.on("error", (error) => {
  console.error(error);
  process.exit(1);
});

child.on("exit", (code, signal) => {
  if (shuttingDown) return;
  process.exit(code ?? (signal ? 1 : 0));
});

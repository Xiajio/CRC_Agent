const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const { spawn } = require("node:child_process");

const repoRoot = path.resolve(__dirname, "..");
const frontendDist = path.join(repoRoot, "frontend", "dist");
const { chromium } = require(path.join(repoRoot, "frontend", "node_modules", "playwright"));
const outputDir = path.join(repoRoot, "output", "browser-acceptance");
const pythonExe = "D:\\anaconda3\\envs\\LangG\\python.exe";
const backendPort = 8101;
const frontendPort = 4176;
const backendUrl = `http://127.0.0.1:${backendPort}`;
const frontendUrl = `http://127.0.0.1:${frontendPort}`;

function ensureOutputDir() {
  fs.mkdirSync(outputDir, { recursive: true });
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitForHttp(url, child, timeoutMs = 30000) {
  const started = Date.now();
  let lastError = null;
  while (Date.now() - started < timeoutMs) {
    if (child && child.exitCode !== null) {
      throw new Error(`Process exited before ${url} became ready with code ${child.exitCode}`);
    }
    try {
      const response = await fetch(url);
      if (response.ok) {
        return;
      }
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await sleep(500);
  }
  throw new Error(`Timed out waiting for ${url}: ${lastError?.message ?? "unknown error"}`);
}

function startBackend() {
  const stdout = fs.openSync(path.join(outputDir, "real-case-backend.out.log"), "w");
  const stderr = fs.openSync(path.join(outputDir, "real-case-backend.err.log"), "w");
  return spawn(
    pythonExe,
    ["-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", String(backendPort)],
    {
      cwd: repoRoot,
      env: {
        ...process.env,
        AUTH_MODE: "none",
        GRAPH_RUNNER_MODE: "fixture",
        GRAPH_FIXTURE_CASE: "real_case_human_review",
        UPLOAD_CONVERTER_MODE: "fixture",
        RAG_WARMUP: "false",
        FRONTEND_ORIGINS: frontendUrl,
        PYTHONUTF8: "1",
        PYTHONIOENCODING: "utf-8",
      },
      stdio: ["ignore", stdout, stderr],
      windowsHide: true,
    },
  );
}

function contentType(filePath) {
  if (filePath.endsWith(".html")) return "text/html; charset=utf-8";
  if (filePath.endsWith(".js")) return "text/javascript; charset=utf-8";
  if (filePath.endsWith(".css")) return "text/css; charset=utf-8";
  if (filePath.endsWith(".svg")) return "image/svg+xml";
  if (filePath.endsWith(".json")) return "application/json; charset=utf-8";
  return "application/octet-stream";
}

function readRequestBody(request) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    request.on("data", (chunk) => chunks.push(chunk));
    request.on("end", () => resolve(chunks.length > 0 ? Buffer.concat(chunks) : undefined));
    request.on("error", reject);
  });
}

async function proxyApiRequest(request, response, parsed) {
  try {
    const body = await readRequestBody(request);
    const headers = { ...request.headers };
    delete headers.host;
    const upstream = await fetch(`${backendUrl}${parsed.pathname}${parsed.search}`, {
      method: request.method,
      headers,
      body: ["GET", "HEAD"].includes(request.method || "GET") ? undefined : body,
    });
    const responseBody = Buffer.from(await upstream.arrayBuffer());
    const responseHeaders = {};
    upstream.headers.forEach((value, key) => {
      responseHeaders[key] = value;
    });
    response.writeHead(upstream.status, responseHeaders);
    response.end(responseBody);
  } catch (error) {
    response.writeHead(502, { "Content-Type": "text/plain; charset=utf-8" });
    response.end(`Proxy error: ${error.message}`);
  }
}

function startStaticServer() {
  const server = http.createServer((request, response) => {
    const parsed = new URL(request.url || "/", frontendUrl);
    if (parsed.pathname.startsWith("/api/")) {
      void proxyApiRequest(request, response, parsed);
      return;
    }

    const requestedPath = parsed.pathname === "/" ? "/index.html" : parsed.pathname;
    const decodedPath = decodeURIComponent(requestedPath);
    const filePath = path.resolve(frontendDist, `.${decodedPath}`);
    if (!filePath.startsWith(frontendDist)) {
      response.writeHead(403);
      response.end("Forbidden");
      return;
    }
    fs.readFile(filePath, (error, body) => {
      if (error) {
        response.writeHead(404);
        response.end("Not found");
        return;
      }
      response.writeHead(200, { "Content-Type": contentType(filePath) });
      response.end(body);
    });
  });
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(frontendPort, "127.0.0.1", () => resolve(server));
  });
}

async function launchBrowser() {
  try {
    return await chromium.launch({ channel: "msedge", headless: true });
  } catch {
    return chromium.launch({ headless: true });
  }
}

async function runAcceptance() {
  if (!fs.existsSync(path.join(frontendDist, "index.html"))) {
    throw new Error("frontend/dist/index.html not found. Run the frontend build first.");
  }

  ensureOutputDir();
  const backend = startBackend();
  const staticServer = await startStaticServer();
  let browser;
  try {
    await waitForHttp(`${backendUrl}/openapi.json`, backend);
    await waitForHttp(`${frontendUrl}/`);

    browser = await launchBrowser();
    const context = await browser.newContext({ viewport: { width: 1440, height: 1100 } });
    const page = await context.newPage();
    const consoleErrors = [];
    const failedResponses = [];
    page.on("console", (message) => {
      if (["error", "warning"].includes(message.type())) {
        consoleErrors.push(`${message.type()}: ${message.text()}`);
      }
    });
    page.on("pageerror", (error) => {
      consoleErrors.push(`pageerror: ${error.message}`);
    });
    page.on("response", (response) => {
      if (response.status() >= 400) {
        failedResponses.push(`${response.status()} ${response.url()}`);
      }
    });

    await page.goto(frontendUrl, { waitUntil: "networkidle", timeout: 30000 });
    await page.locator("textarea").first().waitFor({ state: "visible", timeout: 20000 });
    await page.locator("textarea").first().fill(
      "62-year-old male with biopsy-confirmed low rectal adenocarcinoma, pMMR, MRI cT3N1M0, no distant metastasis, ECOG 1. Please provide treatment recommendation.",
    );
    await page.locator("textarea").first().press("Enter");

    await page.getByText("HUMAN_REVIEW_REQUIRED").first().waitFor({ timeout: 30000 });
    await page.getByText("Recommendation retained for review").waitFor({ timeout: 30000 });
    await page.getByText("No direct references are attached to this recommendation.").waitFor({ timeout: 30000 });
    await page.getByText("Roadmap updated").first().waitFor({ timeout: 30000 });
    await page.getByText("Discuss total neoadjuvant therapy in multidisciplinary tumor board.").first().waitFor({ timeout: 30000 });
    await page.getByText("cT3N1 low rectal cancer generally requires neoadjuvant treatment before surgery.").first().waitFor({ timeout: 30000 });

    const planRows = await page.locator(".clinical-plan-row").count();
    const roadmapSteps = await page.locator(".clinical-roadmap-step").count();
    const blockedRoadmapSteps = await page.locator(".clinical-roadmap-step-blocked").count();
    const eventChips = await page.locator(".clinical-event-chip").count();
    const roadmapEventChips = await page.locator(".clinical-event-chip").filter({ hasText: "Roadmap updated" }).count();
    const warningCount = await page.getByText("HUMAN_REVIEW_REQUIRED").count();
    const finalVisible = await page.getByText("cT3N1M0").count();
    const faviconFailures = failedResponses.filter((entry) => entry.includes("favicon"));

    if (planRows < 1) throw new Error("Execution plan did not render any rows.");
    if (roadmapSteps < 1) throw new Error("Roadmap did not render any steps.");
    if (blockedRoadmapSteps < 1) throw new Error("Roadmap did not preserve a blocked review/citation step.");
    if (eventChips < 1) throw new Error("Clinical event stream did not render any events.");
    if (roadmapEventChips < 1) throw new Error("Clinical event stream did not record roadmap updates.");
    if (warningCount < 1) throw new Error("Human review warning was not visible.");
    if (finalVisible < 1) throw new Error("Final recommendation/case stage was not visible.");
    if (faviconFailures.length > 0) throw new Error(`Favicon request failed: ${faviconFailures.join("; ")}`);

    const screenshotPath = path.join(outputDir, "real-case-human-review-acceptance.png");
    await page.screenshot({ path: screenshotPath, fullPage: true });

    const result = {
      ok: true,
      frontendUrl,
      backendUrl,
      planRows,
      roadmapSteps,
      blockedRoadmapSteps,
      eventChips,
      roadmapEventChips,
      warningCount,
      failedResponses,
      consoleErrors,
      screenshotPath,
    };
    console.log(JSON.stringify(result, null, 2));
  } finally {
    if (browser) {
      await browser.close();
    }
    await new Promise((resolve) => staticServer.close(resolve));
    if (backend.exitCode === null) {
      backend.kill();
    }
  }
}

runAcceptance().catch((error) => {
  console.error(error);
  process.exit(1);
});

# Demo Mode Runbook

## Primary Replay Demo

Start the replay-first demo profile from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_demo.ps1
```

In a second terminal, run the automated browser walkthrough:

```powershell
node .\scripts\run_demo_playwright.cjs
```

The runner starts the demo backend and frontend, waits for both ports, runs `frontend/playwright.demo.config.ts` through the local `frontend/node_modules/@playwright/test/cli.js`, and then cleans up the service process trees. Set `DEMO_PLAYWRIGHT_HEADLESS=1` before the command when you want a headless verification run instead of a visible demo browser.

This demo profile sets:

- `GRAPH_RUNNER_MODE=fixture`
- `GRAPH_FIXTURE_CASE=demo_doctor_decision`
- `UPLOAD_CONVERTER_MODE=fixture`
- `VITE_DEMO_MODE=replay`
- `LANGG_RUNTIME_ROOT=runtime/demo`

## Patient Flow

1. Fill patient identity.
2. Ask about blood in stool, abdominal pain, stool shape change, and weight loss.
3. Answer the triage card with `1个月以上`.
4. Upload `tests/fixtures/demo_uploads/demo_colonoscopy_report.pdf`.
5. Review the triage recommendation and uploaded patient-background context.

## Doctor Flow

1. Open `医生场景`.
2. Open `患者数据库`.
3. Switch to `历史病例`.
4. Open historical case `093` and click `带入会诊`.
5. Ask for clinical assessment, evidence, and treatment recommendation.
6. Review roadmap, execution plan, references, critic result, and human-review warning.
7. Open `多模态`.

## Real Model Backup

Use the existing real-chain startup command:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_real.ps1 -WarmupRag
```

Real mode does not set `GRAPH_RUNNER_MODE=fixture`, `UPLOAD_CONVERTER_MODE=fixture`, `VITE_DEMO_MODE=replay`, or `LANGG_RUNTIME_ROOT=runtime/demo`.

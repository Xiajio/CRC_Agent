# Acceptance Runbook

This runbook is aligned to the current repository at `D:\YiZhu_Agnet\LangG`. It keeps the historical full-pack E2E notes, but the current handoff gate for the real clinical case is the browser acceptance script backed by the `real_case_human_review` graph fixture.

Run commands from the repository root unless a command states otherwise.

## Scope

Current primary acceptance scope:

- Fixture: `tests\fixtures\graph_ticks\real_case_human_review.json`
- Scenario: 62-year-old male, biopsy-confirmed pMMR low rectal adenocarcinoma, MRI `cT3N1M0`, no distant metastasis, ECOG 1
- Expected safety posture: `HUMAN_REVIEW_REQUIRED`
- Expected citation posture: no direct references attached; recommendation is retained for manual oncology review
- Expected UI evidence: execution plan rows, roadmap steps, at least one blocked roadmap/review step, clinical event chips, and final case-stage text

Historical full-pack scope:

- `scripts\run_e2e_full_acceptance.ps1` is retained as the old full-pack launcher.
- The current tree does not contain `tests\e2e\acceptance`, so do not treat the old Playwright full-pack command as the active handoff unless that suite is restored.

## Prerequisites

Confirm these before running the browser acceptance:

- `D:\anaconda3\envs\LangG\python.exe` exists.
- `D:\anaconda3\envs\LangG\npm.cmd` exists.
- `D:\anaconda3\envs\LangG\node.exe` exists, or `node` resolves on `PATH`.
- `frontend\node_modules\playwright` exists.
- `tests\fixtures\graph_ticks\real_case_human_review.json` exists.
- Ports `8101` and `4176` are free.
- `output\browser-acceptance\real_case_human_review\` is writable.

Build the frontend first:

```powershell
D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run build
```

## Real Case Browser Acceptance

Run the current acceptance handoff:

```powershell
D:\anaconda3\envs\LangG\node.exe scripts/run_real_case_browser_acceptance.cjs
```

If `node.exe` is already on `PATH`, this equivalent command is acceptable:

```powershell
node scripts/run_real_case_browser_acceptance.cjs
```

The script starts an isolated fixture backend on `http://127.0.0.1:8101`, serves the built frontend on `http://127.0.0.1:4176`, drives a headless browser, and validates the real-case human-review UI state.

Expected pass criteria:

- Browser reaches the workspace and submits the real rectal cancer prompt.
- `HUMAN_REVIEW_REQUIRED` is visible.
- `Recommendation retained for review` is visible.
- `No direct references are attached to this recommendation.` is visible.
- Roadmap and execution-plan panels render review-aware steps.
- At least one roadmap step remains blocked for review/citation follow-up.
- No failed favicon request is recorded.

## Evidence Locations

Real-case browser evidence:

- `output\browser-acceptance\real_case_human_review\real-case-human-review-acceptance.json`
- `output\browser-acceptance\real_case_human_review\real-case-human-review-acceptance.png`
- `output\browser-acceptance\real_case_human_review\real-case-backend.out.log`
- `output\browser-acceptance\real_case_human_review\real-case-backend.err.log`

Historical full-pack evidence, if the old pack is restored and run:

- `output\acceptance\`

## Manual Review Handoff

After browser acceptance passes:

1. Review `real-case-human-review-acceptance.json` for `ok: true`, `fixtureCase: real_case_human_review`, and nonzero plan/roadmap/event counts.
2. Review `real-case-human-review-acceptance.png` for the warning banner, retained recommendation, blocked review step, and no-direct-reference message.
3. Review backend logs for startup/runtime errors.
4. Complete `docs\superpowers\acceptance\e2e-manual-review-checklist.md`.
5. Complete either `docs\superpowers\acceptance\real-case-human-review-acceptance-report-2026-05-03.md` or the template at `docs\superpowers\acceptance\e2e-release-report-template.md`.

Required human reviewers:

- Medical reviewer: treatment wording, staging, and manual-oncology-review posture
- Product/test reviewer: visual layout, roadmap/plan/event consistency, and screenshot readability
- Safety reviewer: `HUMAN_REVIEW_REQUIRED`, no-direct-reference disclosure, and no implied autonomous release

## Blocker Policy

- If the frontend build fails, stop and record the build command and error.
- If the fixture backend does not start on port `8101`, stop and review backend logs.
- If the browser script fails any expected UI assertion, stop and preserve the JSON, screenshot, and logs.
- If the warning, blocked review step, or no-direct-reference disclosure is missing, mark the acceptance as FAIL.
- If evidence is missing, incomplete, or unreadable, do not sign off.

## Legacy Runner Reference

The historical full-pack launcher remains:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_e2e_full_acceptance.ps1 -ListOnly
```

Use it only when `tests\e2e\acceptance` has been restored and the old acceptance-support tests are present. It is not the current real-case human-review handoff.

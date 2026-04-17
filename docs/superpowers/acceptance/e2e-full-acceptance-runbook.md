# Full Acceptance Runbook

This runbook covers the acceptance gate for the full E2E pack. It assumes the repository has already been prepared with the acceptance fixtures, the dedicated Playwright config, and the backend fixture runner.

Run every command in this document from the repository root: `D:\亿铸智能体\LangG_New`.

## L0 Prerequisites

Confirm these before starting any automated run:

- `D:\anaconda3\envs\LangG\python.exe` exists.
- `D:\anaconda3\envs\LangG\npm.cmd` exists.
- `tests\fixtures\acceptance_case_db\seed.json` exists.
- `frontend\package.json` includes `test:e2e:acceptance`.
- Ports `8000` and `4173` are free.
- `runtime\` is writable so the acceptance case database can be regenerated.

Command preview only:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_e2e_full_acceptance.ps1 -ListOnly
```

`-ListOnly` is a dry listing step. It does not validate the environment; it only prints the runner commands, evidence path, and document locations.

Exact L0 checks:

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_payload_builder.py tests/backend/test_capture_graph_fixtures.py tests/backend/test_acceptance_case_db.py tests/backend/test_uploads_routes.py -v
```

```powershell
D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test:e2e:acceptance
```

If any prerequisite fails, stop and fix the environment before continuing.

## L1 Automated Checks

Run the backend acceptance-support tests first. This is the fast gate that proves the fixture plumbing, acceptance database materialization, and upload conversion path are intact.

```powershell
D:\anaconda3\envs\LangG\python.exe -m pytest tests/backend/test_payload_builder.py tests/backend/test_capture_graph_fixtures.py tests/backend/test_acceptance_case_db.py tests/backend/test_uploads_routes.py -v
```

If you only need the quickest interactive pass, use the runner in core-only mode:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_e2e_full_acceptance.ps1 -CoreOnly
```

The core-only mode runs the backend acceptance-support tests and the workspace core Playwright spec.

## L2 Automated Checks

Run the full Playwright acceptance pack after L1 passes:

```powershell
D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test:e2e:acceptance
```

If you need to rerun an individual layer, use the spec-specific commands:

```powershell
D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test:e2e:acceptance -- ../tests/e2e/acceptance/workspace-core.spec.ts
```

```powershell
D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test:e2e:acceptance -- ../tests/e2e/acceptance/database-console.spec.ts
```

```powershell
D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run test:e2e:acceptance -- ../tests/e2e/acceptance/uploads-and-cards.spec.ts
```

## Evidence Locations

- Playwright traces, screenshots, and attachments: `output\acceptance\`
- Operator docs: `docs\superpowers\acceptance\`
- Release report template: `docs\superpowers\acceptance\e2e-release-report-template.md`

## Manual Review Handoff Order

1. Review the backend acceptance-support test results.
2. Review the full Playwright acceptance results.
3. Inspect the evidence bundle in `output\acceptance\`.
4. Review the workspace core case first.
5. Review the database console case second.
6. Review the uploads and card follow-up case third.
7. Complete `docs\superpowers\acceptance\e2e-manual-review-checklist.md`.
8. Complete `docs\superpowers\acceptance\e2e-release-report-template.md`.

## Blocker Policy

- If any L0 prerequisite fails, stop immediately.
- If any backend acceptance-support test fails, stop immediately.
- If Playwright fails, do not start manual review.
- If evidence is missing, incomplete, or unreadable, stop and record the blocker.
- If wording, visuals, card semantics, preview usability, or safety presentation raise a concern, record it in the checklist and the release report before any release decision.

## Runner Reference

The one-command launcher is `scripts\run_e2e_full_acceptance.ps1`.

- `-ListOnly` prints the backend command, Playwright command, evidence path, and docs paths, then exits without running validation.
- `-CoreOnly` runs the backend acceptance-support tests plus the workspace core Playwright spec.
- `-SkipManualDocs` suppresses the long handoff reminder after automation passes.

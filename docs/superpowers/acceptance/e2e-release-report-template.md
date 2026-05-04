# Acceptance Release Report Template

Use this template for the current real-case human-review browser acceptance. The historical full-pack E2E runner may be added under "Legacy Full-Pack Notes" only when `tests\e2e\acceptance` exists and has been run.

## Environment Metadata

- Date:
- Operator:
- Branch:
- Commit:
- Repo root: `D:\YiZhu_Agnet\LangG`
- Backend interpreter: `D:\anaconda3\envs\LangG\python.exe`
- Node/npm:
- Frontend build command: `D:\anaconda3\envs\LangG\npm.cmd --prefix frontend run build`
- Browser acceptance command: `D:\anaconda3\envs\LangG\node.exe scripts/run_real_case_browser_acceptance.cjs`
- Fixture: `real_case_human_review`
- Evidence directory: `output\browser-acceptance\real_case_human_review\`

## Scenario Summary

- Patient case:
- Expected safety posture: `HUMAN_REVIEW_REQUIRED`
- Expected citation posture: no direct references attached; manual oncology review required
- Expected UI surfaces: warning, retained recommendation, no-direct-reference disclosure, execution plan, roadmap, blocked review step, clinical event stream

## Automation Summary

- Frontend build: PASS / FAIL / NOTE:
- Browser acceptance: PASS / FAIL / NOTE:
- JSON evidence: PASS / FAIL / NOTE:
- Screenshot evidence: PASS / FAIL / NOTE:
- Backend logs reviewed: PASS / FAIL / NOTE:
- Key automated observations:

## Manual Sign-Offs

- Medical wording: PASS / FAIL / NOTE:
- Visual quality: PASS / FAIL / NOTE:
- Roadmap/execution-plan semantics: PASS / FAIL / NOTE:
- Citation/no-direct-reference disclosure: PASS / FAIL / NOTE:
- Trust and safety presentation: PASS / FAIL / NOTE:

## Evidence Reviewed

- `output\browser-acceptance\real_case_human_review\real-case-human-review-acceptance.json`
- `output\browser-acceptance\real_case_human_review\real-case-human-review-acceptance.png`
- `output\browser-acceptance\real_case_human_review\real-case-backend.out.log`
- `output\browser-acceptance\real_case_human_review\real-case-backend.err.log`

## Blocker List

- None, or list each blocker with the failing command, case, and evidence path.

## Legacy Full-Pack Notes

- `scripts\run_e2e_full_acceptance.ps1` run: YES / NO
- Reason if not run:
- Evidence directory if run: `output\acceptance\`

## Final Decision

- Decision: PASS WITH HUMAN REVIEW REQUIRED / PASS WITH CONDITIONS / FAIL
- Rationale:
- Release note:

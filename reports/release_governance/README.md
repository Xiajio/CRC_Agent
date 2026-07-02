# Release Governance Reports

This directory stores Step 12 audit-only release governance artifacts.

Allowed generated artifacts:

- `intents/*.json`
- `approvals/*.json`
- `rollback_plans/*.json`
- `audit/*.jsonl`

These files record release intent, human approvals, rollback plans, and audit
events. They do not execute release, execute rollback, toggle feature flags,
mutate safety policy, update prompts, write RAG indexes, promote literature
evidence, or deploy model/tool changes.

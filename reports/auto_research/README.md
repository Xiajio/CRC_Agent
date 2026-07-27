# Auto Research Run Artifacts

This directory stores append-only, shadow-only auto-research runs.

Each run may retrieve PubMed abstracts, generate source-bound hypotheses,
perform a separate adversarial-review pass with the configured reasoning model,
draft an unexecuted study plan, and synthesize a line-by-line source-ID-bound
report for human review. These artifacts are research candidates, not clinical
facts or validated discoveries. The review pass is not an independent model or
independent scientific reviewer.

Runs do not apply prompts, rubrics, routes, templates, RAG content, safety
policy, feature flags, model training data, or patient/doctor runtime state.
They never return patient-level rows. Promotion, execution, and clinical use
remain outside this pipeline and require separate human-reviewed governance.

The API requires an explicit `deidentified=true` declaration and rejects common
obvious identifier patterns before persistence or external egress. This
heuristic is not a substitute for institutional DLP controls or operator review;
research questions must never contain patient identifiers.

Writes are serialized by Run ID within one process and published from a fully
flushed temporary file through an atomic, non-overwriting filesystem link.
Persisted request hashes and derived Run IDs are revalidated on read. The
stage topology, status/artifact combinations, one-plan-per-hypothesis mapping,
report citations, and source ledger are revalidated on both read and final
publication. The `integrity` field therefore means structurally validated; it
is not a digital signature and does not prove that an authorized person did not
edit an artifact.

Current MVP limitations: creation executes synchronously in a FastAPI threadpool
and has no durable worker, scheduler, cancellation, retry, cross-process lease,
or pagination. Multiple server processes can still duplicate upstream work,
although the append-only publication step prevents overwriting a completed Run.

## Integrity warnings and recovery

The Run list includes only artifacts that pass the complete persisted contract,
including the filename-to-`run_id` check. A rejected file remains excluded from
normal results and is reported through `integrity.affected_artifacts`, with its
relative path, filename-derived Run ID, readable persisted Run ID, and exclusion
state. The Admin UI never renames, overwrites, deletes, or quarantines a file.

Safe recovery is deliberately narrow:

1. Submit the research request again with a new idempotency key. This appends a
   new Run and leaves the affected bytes untouched.
2. If the file must be isolated, an authorized operator first preserves the
   original bytes and SHA-256, records operator, reason, and timestamp, and then
   moves it outside `reports/auto_research/runs` through an audited manual
   procedure.

Neither path writes to or rewrites clinical data. Renaming a damaged file in
place, editing its internal `run_id`, or reusing it as a normal result is not a
supported recovery action.

## Source-control boundary

`runs/` and `validation/` are deployment-local, append-only operational data and
are excluded from source control. They may contain full research questions,
retrieved abstracts, provider diagnostics, timestamps, and operator validation
records. Do not commit those artifacts, even when the request is declared
deidentified. Store and integrity tests build synthetic artifacts under pytest
temporary directories instead of reading operational data from this repository.

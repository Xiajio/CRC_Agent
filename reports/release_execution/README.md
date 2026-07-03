# Release Execution Artifacts

This directory is reserved for Step 13 controlled local release execution.

Runtime-generated files under `requests/`, `results/`, `feature_flags/`, and `audit/` are append-only execution evidence. They are created by admin-only release execution APIs and should not be edited manually.

Step 13 execution state is local and auditable. It does not call external deployment systems, store credentials, mutate clinical safety policy, mutate prompts, mutate RAG indexes, promote literature evidence, or change patient/doctor default paths.

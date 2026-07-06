# Release Monitoring Artifacts

This directory is reserved for Step 14 post-release monitoring.

Runtime-generated files under `checks/`, `acknowledgements/`, and `audit/` are append-only monitoring evidence. They are created by admin-only monitoring APIs and should not be edited manually.

Step 14 monitoring state is local and auditable. It does not call external alerting systems, execute rollback, store credentials, mutate release execution, mutate governance, mutate clinical safety policy, mutate prompts, mutate RAG indexes, promote literature evidence, or read patient/doctor runtime data.

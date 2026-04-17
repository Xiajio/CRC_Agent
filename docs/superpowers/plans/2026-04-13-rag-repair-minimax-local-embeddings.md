# RAG Repair, MiniMax Vision, and Local Embeddings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the damaged RAG index flow, switch multimodal recognition to MiniMax 2.7 compatibly, and add a local embedding backend that can use GPU when the runtime supports it.

**Architecture:** Keep the existing RAG pipeline, but harden three weak points: provider-compatible vision/model construction, vector index repair/rebuild behavior, and embedding backend pluggability. Default behavior remains API-compatible unless explicit local embedding settings are enabled.

**Tech Stack:** Python, LangChain, Chroma, PyMuPDF, Hugging Face Transformers, PyTorch, pytest.

---

### Task 1: Lock Down Failing Behaviors

**Files:**
- Modify: `tests/backend/test_llm_service.py`
- Modify: `tests/backend/test_summary_and_retriever_resilience.py`
- Create: `tests/backend/test_rag_parser_and_embeddings.py`

- [ ] **Step 1: Write failing tests for provider-compatible vision payload handling**
- [ ] **Step 2: Write failing tests for local embedding backend selection and local-env validation**
- [ ] **Step 3: Write failing tests for ingest metadata consistency and vector repair helpers**
- [ ] **Step 4: Run focused pytest commands and confirm the new tests fail for the expected reasons**

### Task 2: Implement Provider-Compatible Vision Models

**Files:**
- Modify: `src/services/llm_service.py`
- Modify: `src/rag/parser.py`
- Modify: `src/services/document_converter.py`

- [ ] **Step 1: Expose a reusable ChatOpenAI factory/patch path for provider-compatible raw models**
- [ ] **Step 2: Switch parser vision/model calls to the compatible factory**
- [ ] **Step 3: Switch document converter model creation to the compatible factory**
- [ ] **Step 4: Run the related tests and confirm MiniMax-compatible payload rewriting passes**

### Task 3: Add Local Embedding Backend Support

**Files:**
- Modify: `src/config.py`
- Modify: `src/rag/retriever.py`
- Modify: `src/rag/ingest.py`
- Modify: `.env`

- [ ] **Step 1: Add explicit embedding backend/device/model settings**
- [ ] **Step 2: Implement a local Hugging Face embedding class with auto device selection**
- [ ] **Step 3: Allow ingest/retriever env validation to pass for local backend without API credentials**
- [ ] **Step 4: Update `.env` defaults to point multimodal parsing at MiniMax 2.7 and document local embedding options**

### Task 4: Repair and Rebuild the Damaged Index

**Files:**
- Modify: `src/rag/ingest.py`
- Modify: `src/rag/__init__.py`

- [ ] **Step 1: Add a repair-oriented helper that clears caches, backs up index directories, and rebuilds cleanly**
- [ ] **Step 2: Ensure ingest writes stable vector IDs and consistent metadata fields**
- [ ] **Step 3: Run focused retriever/ingest tests to confirm repair logic stays green**
- [ ] **Step 4: Execute the rebuild against the local workspace data and verify collection health**

### Task 5: Verify End-to-End

**Files:**
- Modify: `tests/backend/test_llm_service.py`
- Modify: `tests/backend/test_summary_and_retriever_resilience.py`
- Create: `tests/backend/test_rag_parser_and_embeddings.py`

- [ ] **Step 1: Run targeted pytest for llm service, parser/embeddings, and retriever resilience**
- [ ] **Step 2: Run a local collection health check and a sample retrieval query**
- [ ] **Step 3: Record any remaining runtime limitations, especially GPU availability for local embeddings**

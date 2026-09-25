# TRACEABILITY_SPEC

## Scope
This spec defines a deterministic text-analysis pipeline for `context.html`. The pipeline must produce the same derived artifacts when run against the same input bytes, code revision, configuration, and toolchain.

## Goals
- Make every analytical output traceable to a specific byte-identical version of `context.html`.
- Guarantee replayability: the same run inputs must yield the same normalized text, chunk boundaries, findings, and hashes.
- Preserve enough provenance to let an auditor reconstruct how a conclusion was produced.
- Keep artifacts small, text-based, and easy to diff.

## Non-Goals
- Defining the semantic quality of the analysis itself.
- Prescribing a specific model, parser library, or rule engine.
- Supporting nondeterministic workflows such as sampling-based LLM calls without a fixed, recorded output cache.
- Tracking provenance for files other than `context.html`.

## Required Artifacts
Each run must emit the following artifacts under a single run directory:

- `run_manifest.json`: top-level metadata for the run.
- `input_manifest.json`: facts about `context.html` before processing.
- `normalized_text.txt`: canonical text extracted from `context.html`.
- `segments.jsonl`: ordered analysis units derived from `normalized_text.txt`.
- `findings.json`: final structured analysis output.
- `event_log.jsonl`: append-only execution log for each pipeline stage.

Recommended run directory format:

- `artifacts/runs/{run_id}/...`

## Provenance Requirements
`run_manifest.json` must record:

- `run_id`
- pipeline name and version
- code revision identifier if available (for example, git commit SHA)
- full configuration object and its hash
- operating system, runtime, and dependency versions
- locale, timezone, and declared text encoding
- deterministic seed values for any component that accepts a seed
- wall-clock start and end timestamps in ISO 8601 UTC

`input_manifest.json` must record:

- source path as provided to the pipeline
- canonical resolved path
- file size in bytes
- SHA-256 of raw `context.html` bytes
- last modified timestamp if available
- extraction method used to convert HTML to canonical text

Every record in `segments.jsonl` and every item in `findings.json` must include:

- `run_id`
- `input_sha256`
- stable segment or finding identifier
- source offsets into `normalized_text.txt`
- the rule, prompt, or algorithm version that produced the record

## Naming Conventions
- Use lowercase snake_case for artifact filenames, JSON keys, and record identifiers.
- Use zero-padded ordinal fields where ordering matters, such as `segment_index`.
- Derive `run_id` deterministically from immutable run inputs. Preferred format:
  `trace_{input_sha256_12}_{config_sha256_12}`
- Do not include human-edited labels, local timestamps, or random UUIDs in identifiers that affect replayability.

## Hashing And Logging
- Raw input hash: SHA-256 over the exact bytes of `context.html`.
- Config hash: SHA-256 over canonical JSON with sorted keys and no insignificant whitespace.
- Artifact hashes: SHA-256 over emitted file bytes after serialization.
- JSON artifacts must be serialized canonically: UTF-8, LF line endings, sorted keys, and stable list ordering.
- `event_log.jsonl` must be append-only and ordered by stage execution.
- Each log event must include `run_id`, `stage`, `started_at`, `ended_at`, `status`, input artifact hashes, and output artifact hashes.
- Logs may include diagnostics, but must not omit failed stages; failures are part of the audit trail.

## Determinism Rules
- HTML extraction, whitespace normalization, and segmentation rules must be explicitly defined and versioned.
- Iteration order over segments, rules, and outputs must be stable and not depend on filesystem order or concurrency timing.
- If a model is used, inference parameters must be fixed. Any external service response must be cached by hash or treated as non-compliant with this spec.
- Any timestamp included in replay-sensitive artifacts must be derived from inputs or excluded from hashed outputs.

## Acceptance Criteria For Auditability
A pipeline implementation satisfies this spec only if all of the following are true:

- An auditor can start from `context.html` and the run directory alone and reconstruct the full analysis path.
- Re-running the pipeline with the same raw input bytes, code revision, config, and dependencies produces byte-identical `normalized_text.txt`, `segments.jsonl`, and `findings.json`.
- Every finding can be traced to one or more exact source offsets in `normalized_text.txt`.
- Every artifact referenced in a log entry exists and matches the logged hash.
- Any failed or partial run still leaves a complete `run_manifest.json`, `input_manifest.json`, and `event_log.jsonl`.

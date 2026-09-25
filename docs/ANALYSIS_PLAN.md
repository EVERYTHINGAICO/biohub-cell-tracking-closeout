# Deterministic Analysis Plan

## Objective

Create a reproducible analysis workflow for `context.html` that can be rerun later and audited without guesswork.

## Inputs

- Source file: `context.html`
- Stable analyzer: `scripts/analyze_context.py`

## Outputs

The analyzer writes artifacts under:

- `artifacts/context_analysis/context__<sha12>/manifest.json`
- `artifacts/context_analysis/context__<sha12>/paragraphs.jsonl`
- `artifacts/context_analysis/context__<sha12>/sections.json`
- `artifacts/context_analysis/context__<sha12>/summary.md`

`<sha12>` is the first 12 hex characters of the input file SHA-256, so artifact naming is deterministic for a given file.

## Execution Steps

1. Read `context.html` as bytes and compute:
   - raw byte length
   - raw SHA-256
   - normalized text SHA-256 after CRLF to LF normalization
2. Split the file into paragraphs using blank lines as separators.
3. Classify each paragraph with ordered rules and record:
   - line start and line end
   - paragraph text hash
   - block type
   - classifier rule that fired
4. Aggregate adjacent paragraphs of the same block type into sections.
5. Compute deterministic lexical metrics per section:
   - token count
   - alphabetic token count
   - digit token count
   - punctuation token count
   - question mark count
   - absolute / modal / negation cue counts
6. Write machine-readable artifacts and a concise markdown summary.

## Block Taxonomy

- `meta_reasoning`
- `roleplay_or_persona_text`
- `dataset_description`
- `conversation_query`
- `conversation_response`
- `baseline_exposition`
- `python_code`
- `technical_heading`
- `other`

## Determinism Rules

- No timestamps in generated content.
- No randomness.
- No network calls.
- Ordered classifier rules are fixed in code.
- Artifact paths depend only on the input hash.
- All summaries must cite line ranges from the source file.

## Traceability Notes

- Every paragraph carries its own SHA-256.
- Every section records the paragraph IDs it contains.
- The manifest links each output file back to the exact input hash and analyzer version.

## Delegation Log

- Explorer agent `Lagrange` was tasked with defining segmentation rules for `context.html`.
- Worker agent `Descartes` was tasked with drafting the traceability spec in `docs/TRACEABILITY_SPEC.md`.
- The main agent owns the analyzer implementation and integration.

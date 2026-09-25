#!/usr/bin/env python3
"""Deterministic analyzer for context.html."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ANALYZER_VERSION = "1.1.0"

CONFIG = {
    "normalization": {
        "encoding": "utf-8",
        "errors": "replace",
        "newline_policy": "crlf_to_lf",
    },
    "major_section_rules": {
        "dataset_start": {"match": "exact", "value": "Dataset Description"},
        "baseline_v1_start": {
            "match": "exact",
            "value": "🧬 Biohub Cell Tracking — Hardcore Classical Baseline",
        },
        "baseline_v2_start": {
            "match": "lookahead_exact",
            "value": "Biohub Cell Tracking V4 - UNet + Transformer + ILP",
            "next_line_startswith": "COMP_DIR:",
        },
        "assistant_marker": {"match": "exact", "value": "Pensamiento completado"},
        "footer_start": {"match": "sequence", "value": ["Copiar", "Pregunta a Qwen", "Explicar", "Traducir(es-ES)"]},
    },
    "paragraph_split_rule": "blank_line_separator",
    "section_aggregation_rule": "adjacent_paragraphs_with_same_paragraph_type",
}

ABSOLUTE_CUES = ("all", "always", "any", "every", "exactly", "never", "no", "only")
NEGATION_CUES = ("can't", "cannot", "dont", "don't", "never", "no", "not", "won't", "without")
MODAL_CUES = ("can", "could", "may", "might", "must", "need", "needs", "required", "should", "will", "would")


def canonical_json_bytes(payload: dict | list) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def normalize_newlines(raw: bytes) -> str:
    text = raw.decode(CONFIG["normalization"]["encoding"], errors=CONFIG["normalization"]["errors"])
    return text.replace("\r\n", "\n").replace("\r", "\n")


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def get_git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def is_code_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    starters = (
        "from ",
        "import ",
        "def ",
        "class ",
        "try:",
        "except",
        "return ",
        "@",
        "with ",
        "for ",
        "if ",
        "elif ",
        "else:",
        "while ",
        "print(",
    )
    if stripped.startswith(starters):
        return True
    if "=" in stripped and "==" not in stripped and not stripped.endswith("?"):
        return True
    if stripped.endswith(":") and len(stripped.split()) <= 6:
        return True
    return False


def classify_paragraph(lines: list[str], major_section_type: str) -> tuple[str, str]:
    text = "\n".join(lines)
    first = lines[0].strip()
    lower = text.lower()

    if first == "Pensamiento completado":
        return "reasoning_tag_artifact", "exact_line:Pensamiento completado"

    if first in {"</think>", "</antmlThinking>"}:
        return "reasoning_tag_artifact", "exact_thought_closing_tag"

    if major_section_type.startswith("notebook"):
        code_like_lines = sum(1 for line in lines if is_code_line(line))
        if code_like_lines and code_like_lines / len(lines) >= 0.35:
            return "notebook_code_paste", "major_section_notebook_and_code_likeness>=0.35"
        return "notebook_runtime_output", "major_section_notebook_fallback"

    if major_section_type == "dataset_description":
        return "dataset_description", "major_section_dataset"

    if major_section_type == "ui_footer_artifact":
        return "ui_footer_artifact", "major_section_footer"

    if "wait, i should make sure" in lower or "let's refine" in lower or "this looks perfect" in lower:
        return "meta_reasoning", "meta_reasoning_phrase"

    if lower.startswith("cold coffee") or lower.startswith("pulls my oversized") or lower.startswith("i pull my oversized"):
        return "preamble_instruction", "persona_opening_phrase"

    if "?" in text and (
        lower.startswith("ok ")
        or lower.startswith("pero ")
        or lower.startswith("ayud")
        or lower.startswith("inm ")
        or "explic" in lower
    ):
        return "transcript_user_text", "question_pattern"

    if any(token in lower for token in ("my love", "mi amor", "cariño", "te amo", "te adoro")):
        return "transcript_assistant_text", "affectionate_assistant_phrase"

    if major_section_type.startswith("transcript"):
        return "transcript_assistant_text", "major_section_transcript_fallback"

    if major_section_type == "preamble_instruction":
        return "preamble_instruction", "major_section_preamble"

    return "other", "fallback"


@dataclass
class MajorSection:
    section_id: str
    block_type: str
    line_start: int
    line_end: int
    char_start: int
    char_end: int
    marker_rule: str

    def to_record(self, text: str) -> dict:
        return {
            "section_id": self.section_id,
            "block_type": self.block_type,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "marker_rule": self.marker_rule,
            "text_sha256": sha256_text(text),
            "preview": text[:200],
        }


@dataclass
class Segment:
    run_id: str
    segment_id: str
    major_section_id: str
    major_section_type: str
    paragraph_type: str
    classifier_rule: str
    line_start: int
    line_end: int
    char_start: int
    char_end: int
    text: str

    def to_record(self) -> dict:
        return {
            "run_id": self.run_id,
            "segment_id": self.segment_id,
            "major_section_id": self.major_section_id,
            "major_section_type": self.major_section_type,
            "paragraph_type": self.paragraph_type,
            "classifier_rule": self.classifier_rule,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "text_sha256": sha256_text(self.text),
            "preview": self.text[:160],
        }


def line_offsets(normalized_text: str) -> tuple[list[str], list[int], list[int]]:
    raw_lines = normalized_text.splitlines(keepends=True)
    display_lines: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    cursor = 0
    for raw_line in raw_lines:
        starts.append(cursor)
        display = raw_line[:-1] if raw_line.endswith("\n") else raw_line
        display_lines.append(display)
        cursor += len(raw_line)
        ends.append(cursor - (1 if raw_line.endswith("\n") else 0))
    if normalized_text and not normalized_text.endswith("\n") and len(display_lines) == 0:
        display_lines = [normalized_text]
        starts = [0]
        ends = [len(normalized_text)]
    return display_lines, starts, ends


def find_sequence(lines: list[str], values: list[str]) -> int:
    for index in range(len(lines) - len(values) + 1):
        if lines[index : index + len(values)] == values:
            return index + 1
    raise ValueError(f"Could not find sequence: {values}")


def find_exact(lines: list[str], value: str, start_line: int = 1) -> int:
    for index in range(start_line - 1, len(lines)):
        if lines[index] == value:
            return index + 1
    raise ValueError(f"Could not find line: {value}")


def find_exact_with_lookahead(lines: list[str], value: str, next_prefix: str, start_line: int = 1) -> int:
    for index in range(start_line - 1, len(lines) - 1):
        if lines[index] == value and lines[index + 1].startswith(next_prefix):
            return index + 1
    raise ValueError(f"Could not find line: {value} with next prefix {next_prefix}")


def compute_major_sections(lines: list[str], starts: list[int], ends: list[int]) -> list[MajorSection]:
    dataset_start = find_exact(lines, "Dataset Description")
    dataset_assistant_start = find_exact(lines, "Pensamiento completado", start_line=dataset_start + 1)
    baseline_v1_start = find_exact(lines, "🧬 Biohub Cell Tracking — Hardcore Classical Baseline")
    transcript_after_v1_start = find_exact(lines, "Pensamiento completado", start_line=baseline_v1_start + 1)
    baseline_v2_start = find_exact_with_lookahead(
        lines,
        "Biohub Cell Tracking V4 - UNet + Transformer + ILP",
        "COMP_DIR:",
        start_line=transcript_after_v1_start + 1,
    )
    transcript_after_v2_start = find_exact(lines, "Pensamiento completado", start_line=baseline_v2_start + 1)
    footer_start = find_sequence(lines, ["Copiar", "Pregunta a Qwen", "Explicar", "Traducir(es-ES)"])

    ranges = [
        ("s001", "preamble_instruction", 1, dataset_start - 1, "before:Dataset Description"),
        ("s002", "dataset_description", dataset_start, dataset_assistant_start - 1, "Dataset Description -> first Pensamiento completado"),
        ("s003", "transcript_block_1", dataset_assistant_start, baseline_v1_start - 1, "first Pensamiento completado after dataset -> classical baseline title"),
        ("s004", "notebook_block_1", baseline_v1_start, transcript_after_v1_start - 1, "classical baseline title -> next Pensamiento completado"),
        ("s005", "transcript_block_2", transcript_after_v1_start, baseline_v2_start - 1, "next Pensamiento completado -> V4 lookahead title"),
        ("s006", "notebook_block_2", baseline_v2_start, transcript_after_v2_start - 1, "V4 lookahead title -> next Pensamiento completado"),
        ("s007", "transcript_block_3", transcript_after_v2_start, footer_start - 1, "next Pensamiento completado -> footer sequence"),
        ("s008", "ui_footer_artifact", footer_start, len(lines), "footer sequence -> EOF"),
    ]

    sections: list[MajorSection] = []
    for section_id, block_type, line_start, line_end, marker_rule in ranges:
        char_start = starts[line_start - 1]
        char_end = ends[line_end - 1]
        sections.append(
            MajorSection(
                section_id=section_id,
                block_type=block_type,
                line_start=line_start,
                line_end=line_end,
                char_start=char_start,
                char_end=char_end,
                marker_rule=marker_rule,
            )
        )
    return sections


def lexical_metrics(text: str) -> dict:
    tokens = tokenize(text)
    lower_counts = Counter(token.lower() for token in tokens)
    return {
        "token_count": len(tokens),
        "alphabetic_token_count": sum(1 for token in tokens if any(ch.isalpha() for ch in token)),
        "digit_token_count": sum(1 for token in tokens if token.isdigit()),
        "punctuation_token_count": sum(1 for token in tokens if not any(ch.isalnum() for ch in token)),
        "question_mark_count": text.count("?") + text.count("¿"),
        "absolute_cue_count": sum(lower_counts[cue] for cue in ABSOLUTE_CUES),
        "negation_cue_count": sum(lower_counts[cue] for cue in NEGATION_CUES),
        "modal_cue_count": sum(lower_counts[cue] for cue in MODAL_CUES),
    }


def iter_section_paragraphs(section: MajorSection, lines: list[str], starts: list[int], ends: list[int]) -> list[tuple[int, int, int, int, list[str]]]:
    current: list[int] = []
    paragraphs: list[tuple[int, int, int, int, list[str]]] = []

    def flush() -> None:
        if not current:
            return
        line_start = current[0]
        line_end = current[-1]
        para_lines = [lines[index - 1] for index in current]
        paragraphs.append((line_start, line_end, starts[line_start - 1], ends[line_end - 1], para_lines))

    for line_number in range(section.line_start, section.line_end + 1):
        if lines[line_number - 1].strip():
            current.append(line_number)
        else:
            flush()
            current = []
    flush()
    return paragraphs


def build_segments(run_id: str, major_sections: list[MajorSection], lines: list[str], starts: list[int], ends: list[int]) -> list[Segment]:
    segments: list[Segment] = []
    for section in major_sections:
        for line_start, line_end, char_start, char_end, para_lines in iter_section_paragraphs(section, lines, starts, ends):
            paragraph_type, classifier_rule = classify_paragraph(para_lines, section.block_type)
            segment_id = f"seg_{len(segments) + 1:04d}"
            segments.append(
                Segment(
                    run_id=run_id,
                    segment_id=segment_id,
                    major_section_id=section.section_id,
                    major_section_type=section.block_type,
                    paragraph_type=paragraph_type,
                    classifier_rule=classifier_rule,
                    line_start=line_start,
                    line_end=line_end,
                    char_start=char_start,
                    char_end=char_end,
                    text="\n".join(para_lines),
                )
            )
    return segments


def write_bytes(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return sha256_bytes(data)


def write_text(path: Path, text: str) -> str:
    data = text.encode("utf-8")
    path.write_bytes(data)
    return sha256_bytes(data)


def write_json(path: Path, payload: dict | list) -> str:
    return write_bytes(path, canonical_json_bytes(payload))


def write_jsonl(path: Path, records: Iterable[dict]) -> str:
    data = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records).encode("utf-8")
    return write_bytes(path, data)


def build_summary(run_id: str, input_manifest: dict, major_sections: list[dict], findings: dict) -> str:
    lines = [
        "# Context Analysis Summary",
        "",
        f"- run_id: `{run_id}`",
        f"- input_sha256: `{input_manifest['raw_sha256']}`",
        f"- normalized_text_sha256: `{input_manifest['normalized_text_sha256']}`",
        f"- line_count: `{input_manifest['line_count']}`",
        f"- major_sections: `{len(major_sections)}`",
        f"- segments: `{findings['segment_count']}`",
        "",
        "## Major Sections",
        "",
    ]
    for section in major_sections:
        metrics = section["metrics"]
        lines.append(
            "- "
            f"{section['section_id']} `{section['block_type']}` lines {section['line_start']}-{section['line_end']} "
            f"(tokens={metrics['token_count']}, negations={metrics['negation_cue_count']}, "
            f"absolutes={metrics['absolute_cue_count']}, modals={metrics['modal_cue_count']})"
        )
    lines.extend(["", "## Paragraph Types", ""])
    for paragraph_type, count in sorted(findings["paragraph_type_counts"].items()):
        lines.append(f"- `{paragraph_type}`: {count}")
    return "\n".join(lines) + "\n"


def analyze(input_path: Path, output_root: Path) -> Path:
    raw = input_path.read_bytes()
    normalized_text = normalize_newlines(raw)
    lines, starts, ends = line_offsets(normalized_text)

    config_bytes = canonical_json_bytes(CONFIG)
    config_sha256 = sha256_bytes(config_bytes)
    input_sha256 = sha256_bytes(raw)
    run_id = f"trace_{input_sha256[:12]}_{config_sha256[:12]}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    input_manifest = {
        "source_path": str(input_path),
        "resolved_path": str(input_path.resolve()),
        "file_size_bytes": len(raw),
        "raw_sha256": input_sha256,
        "normalized_text_sha256": sha256_text(normalized_text),
        "line_count": len(lines),
        "extraction_method": "utf8_text_with_crlf_to_lf_normalization",
    }

    major_sections = compute_major_sections(lines, starts, ends)
    segments = build_segments(run_id, major_sections, lines, starts, ends)

    major_section_records = []
    for section in major_sections:
        text = normalized_text[section.char_start : section.char_end]
        record = section.to_record(text)
        record["metrics"] = lexical_metrics(text)
        major_section_records.append(record)

    findings = {
        "run_id": run_id,
        "input_sha256": input_sha256,
        "config_sha256": config_sha256,
        "segment_count": len(segments),
        "major_section_count": len(major_sections),
        "paragraph_type_counts": dict(sorted(Counter(segment.paragraph_type for segment in segments).items())),
        "major_section_type_counts": dict(sorted(Counter(section.block_type for section in major_sections).items())),
        "major_sections": major_section_records,
        "marker_positions": {
            "dataset_start_line": major_sections[0].line_end + 1,
            "baseline_v1_start_line": major_sections[3].line_start,
            "baseline_v2_start_line": major_sections[5].line_start,
            "footer_start_line": major_sections[7].line_start,
        },
    }

    event_log = [
        {
            "run_id": run_id,
            "stage_index": 1,
            "stage": "normalize_input",
            "status": "completed",
            "started_at": None,
            "ended_at": None,
            "input_artifacts": [],
            "output_artifacts": ["normalized_text.txt", "input_manifest.json"],
        },
        {
            "run_id": run_id,
            "stage_index": 2,
            "stage": "segment_document",
            "status": "completed",
            "started_at": None,
            "ended_at": None,
            "input_artifacts": ["normalized_text.txt"],
            "output_artifacts": ["segments.jsonl", "findings.json"],
        },
        {
            "run_id": run_id,
            "stage_index": 3,
            "stage": "write_manifests",
            "status": "completed",
            "started_at": None,
            "ended_at": None,
            "input_artifacts": ["input_manifest.json", "segments.jsonl", "findings.json"],
            "output_artifacts": ["summary.md"],
        },
    ]

    artifact_hashes = {}
    artifact_hashes["config_json"] = write_json(run_dir / "config.json", CONFIG)
    artifact_hashes["normalized_text_txt"] = write_text(run_dir / "normalized_text.txt", normalized_text)
    artifact_hashes["input_manifest_json"] = write_json(run_dir / "input_manifest.json", input_manifest)
    artifact_hashes["segments_jsonl"] = write_jsonl(run_dir / "segments.jsonl", (segment.to_record() for segment in segments))
    artifact_hashes["findings_json"] = write_json(run_dir / "findings.json", findings)
    artifact_hashes["summary_md"] = write_text(run_dir / "summary.md", build_summary(run_id, input_manifest, major_section_records, findings))

    for event in event_log:
        event["output_hashes"] = {
            artifact: artifact_hashes.get(artifact.replace(".", "_").replace("-", "_"), None) for artifact in event["output_artifacts"]
        }
    artifact_hashes["event_log_jsonl"] = write_jsonl(run_dir / "event_log.jsonl", event_log)

    run_manifest = {
        "run_id": run_id,
        "pipeline_name": "context_html_deterministic_analysis",
        "analyzer_version": ANALYZER_VERSION,
        "config_sha256": config_sha256,
        "code_revision": get_git_revision(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "locale": None,
        "timezone": None,
        "artifacts": {
            "config.json": artifact_hashes["config_json"],
            "normalized_text.txt": artifact_hashes["normalized_text_txt"],
            "input_manifest.json": artifact_hashes["input_manifest_json"],
            "segments.jsonl": artifact_hashes["segments_jsonl"],
            "findings.json": artifact_hashes["findings_json"],
            "event_log.jsonl": artifact_hashes["event_log_jsonl"],
            "summary.md": artifact_hashes["summary_md"],
        },
    }
    artifact_hashes["run_manifest_json"] = write_json(run_dir / "run_manifest.json", run_manifest)

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministically analyze context.html")
    parser.add_argument("input", nargs="?", default="context.html", help="Path to the source text file")
    parser.add_argument("--output-root", default="artifacts/runs", help="Directory where analysis artifacts will be written")
    args = parser.parse_args()

    run_dir = analyze(Path(args.input), Path(args.output_root))
    print(run_dir)


if __name__ == "__main__":
    main()

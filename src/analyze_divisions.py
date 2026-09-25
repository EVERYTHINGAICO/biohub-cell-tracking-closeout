from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import geff
import networkx as nx
import numpy as np
import pandas as pd


SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)


def node_xyz_um(attrs: dict) -> np.ndarray:
    return np.array([attrs["z"], attrs["y"], attrs["x"]], dtype=np.float64) * SCALE


def lineage_span_after(graph: nx.DiGraph, node_id: int) -> int:
    node_t = int(graph.nodes[node_id]["t"])
    descendants = nx.descendants(graph, node_id)
    if not descendants:
        return 1
    max_t = max(int(graph.nodes[node]["t"]) for node in descendants)
    return max_t - node_t + 1


def lineage_age_before(graph: nx.DiGraph, node_id: int) -> int:
    node_t = int(graph.nodes[node_id]["t"])
    ancestors = nx.ancestors(graph, node_id)
    if not ancestors:
        return 1
    min_t = min(int(graph.nodes[node]["t"]) for node in ancestors)
    return node_t - min_t + 1


def analyze_geff(geff_path: Path) -> list[dict]:
    graph, _ = geff.read(geff_path, structure_validation=False)
    if not isinstance(graph, nx.DiGraph):
        graph = nx.DiGraph(graph)

    rows = []
    for parent_id in graph.nodes:
        children = list(graph.successors(parent_id))
        if len(children) < 2:
            continue

        parent_attrs = graph.nodes[parent_id]
        parent_xyz = node_xyz_um(parent_attrs)
        child_distances = []
        child_positions = []
        child_spans = []
        for child_id in children:
            child_attrs = graph.nodes[child_id]
            child_xyz = node_xyz_um(child_attrs)
            child_positions.append(child_xyz)
            child_distances.append(float(np.linalg.norm(child_xyz - parent_xyz)))
            child_spans.append(lineage_span_after(graph, child_id))

        child_child_distances = [
            float(np.linalg.norm(a - b))
            for a, b in combinations(child_positions, 2)
        ]

        parent_step_um = np.nan
        predecessors = list(graph.predecessors(parent_id))
        if predecessors:
            pred_attrs = graph.nodes[predecessors[0]]
            parent_step_um = float(np.linalg.norm(parent_xyz - node_xyz_um(pred_attrs)))

        rows.append(
            {
                "volume": geff_path.stem,
                "prefix": geff_path.stem.split("_", 1)[0],
                "parent_id": int(parent_id),
                "parent_t": int(parent_attrs["t"]),
                "parent_z": float(parent_attrs["z"]),
                "parent_y": float(parent_attrs["y"]),
                "parent_x": float(parent_attrs["x"]),
                "n_children": int(len(children)),
                "children_ids": ";".join(str(int(child_id)) for child_id in children),
                "child_distance_um_mean": float(np.mean(child_distances)),
                "child_distance_um_min": float(np.min(child_distances)),
                "child_distance_um_max": float(np.max(child_distances)),
                "child_child_distance_um_mean": float(np.mean(child_child_distances)) if child_child_distances else np.nan,
                "parent_step_um": parent_step_um,
                "parent_lineage_age_frames": int(lineage_age_before(graph, parent_id)),
                "child_lineage_span_mean_frames": float(np.mean(child_spans)),
                "child_lineage_span_min_frames": int(np.min(child_spans)),
                "child_lineage_span_max_frames": int(np.max(child_spans)),
            }
        )

    return rows


def describe(series: pd.Series) -> dict:
    clean = series.dropna()
    if clean.empty:
        return {"count": 0}
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "p10": float(clean.quantile(0.10)),
        "p90": float(clean.quantile(0.90)),
    }


def write_markdown_report(divisions: pd.DataFrame, output: Path) -> None:
    lines = []
    lines.append("# GT Division Analysis")
    lines.append("")
    lines.append(f"Total divisions: {len(divisions)}")
    lines.append(f"Volumes with divisions: {divisions['volume'].nunique() if len(divisions) else 0}")
    lines.append("")

    if divisions.empty:
        output.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("## Timepoints")
    lines.append("")
    lines.append(f"Min t: {int(divisions['parent_t'].min())}")
    lines.append(f"Median t: {float(divisions['parent_t'].median()):.2f}")
    lines.append(f"Max t: {int(divisions['parent_t'].max())}")
    lines.append("")
    lines.append("Top parent_t counts:")
    for t, count in divisions["parent_t"].value_counts().sort_index().items():
        lines.append(f"- t={int(t)}: {int(count)}")
    lines.append("")

    lines.append("## Child Geometry")
    for column in [
        "child_distance_um_mean",
        "child_distance_um_max",
        "child_child_distance_um_mean",
        "parent_step_um",
        "parent_lineage_age_frames",
        "child_lineage_span_mean_frames",
    ]:
        stats = describe(divisions[column])
        lines.append(f"- {column}: {stats}")
    lines.append("")

    lines.append("## By Prefix")
    for prefix, count in divisions["prefix"].value_counts().sort_index().items():
        lines.append(f"- {prefix}: {int(count)}")
    lines.append("")

    lines.append("## Top Volumes")
    for volume, count in divisions["volume"].value_counts().head(20).items():
        lines.append(f"- {volume}: {int(count)}")
    lines.append("")

    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-dir",
        default="/mnt/d/descargas al dico vergas/biohub-cell-tracking-during-development/train",
    )
    parser.add_argument("--output", default="gt_divisions.csv")
    parser.add_argument("--summary-output", default="docs/gt_divisions_summary.md")
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    all_rows = []
    geff_paths = sorted(train_dir.glob("*.geff"))
    print(f"Scanning {len(geff_paths)} GEFF files...")
    for geff_path in geff_paths:
        rows = analyze_geff(geff_path)
        all_rows.extend(rows)

    divisions = pd.DataFrame(all_rows)
    divisions.to_csv(args.output, index=False)
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_report(divisions, summary_output)

    print(f"Total divisions: {len(divisions)}")
    if not divisions.empty:
        print(f"Volumes with divisions: {divisions['volume'].nunique()}")
        print("\nDivisions by prefix:")
        print(divisions["prefix"].value_counts().sort_index().to_string())
        print("\nTimepoint summary:")
        print(divisions["parent_t"].describe().to_string())
        print("\nTop timepoints:")
        print(divisions["parent_t"].value_counts().sort_index().to_string())
        print("\nGeometry summary:")
        print(
            divisions[
                [
                    "child_distance_um_mean",
                    "child_distance_um_max",
                    "child_child_distance_um_mean",
                    "parent_step_um",
                    "parent_lineage_age_frames",
                    "child_lineage_span_mean_frames",
                ]
            ].describe().to_string()
        )
        print("\nTop volumes:")
        print(divisions["volume"].value_counts().head(20).to_string())

    print(f"\nSaved divisions CSV: {args.output}")
    print(f"Saved summary: {summary_output}")


if __name__ == "__main__":
    main()

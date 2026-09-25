from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import geff
import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
MATCH_UM = 7.0


@dataclass(frozen=True)
class VolumeMetrics:
    volume: str
    pred_nodes: int
    gt_nodes: int
    est_nodes: int
    matched_nodes: int
    edge_tp: int
    edge_fp: int
    edge_fn: int
    div_tp: int
    div_fp: int
    div_fn: int
    edge_jaccard: float
    adjusted_edge_jaccard: float
    division_jaccard: float
    combined_score: float
    penalty: float


def _gt_to_frames(graph: nx.DiGraph) -> pd.DataFrame:
    rows = []
    for node_id, attrs in graph.nodes(data=True):
        rows.append(
            {
                "node_id": int(node_id),
                "t": int(attrs["t"]),
                "z": float(attrs["z"]),
                "y": float(attrs["y"]),
                "x": float(attrs["x"]),
            }
        )
    return pd.DataFrame(rows, columns=["node_id", "t", "z", "y", "x"])


def _load_gt_volume(geff_path: Path) -> tuple[pd.DataFrame, set[tuple[int, int]], int]:
    graph, metadata = geff.read(geff_path, structure_validation=False)
    if not isinstance(graph, nx.DiGraph):
        graph = nx.DiGraph(graph)

    nodes = _gt_to_frames(graph)
    edges = {(int(source), int(target)) for source, target in graph.edges()}

    estimated_nodes = len(nodes)
    extra = getattr(metadata, "extra", None)
    if isinstance(extra, dict) and "estimated_number_of_nodes" in extra:
        estimated_nodes = int(extra["estimated_number_of_nodes"])

    return nodes, edges, estimated_nodes


def load_gt(train_dir: str | Path, volumes: list[str] | None = None) -> dict[str, dict]:
    train_path = Path(train_dir)
    if volumes is None:
        geff_paths = sorted(train_path.glob("*.geff"))
    else:
        geff_paths = [train_path / f"{volume}.geff" for volume in volumes]

    gt_data = {}
    for geff_path in geff_paths:
        if not geff_path.exists():
            raise FileNotFoundError(f"Missing GT GEFF: {geff_path}")
        nodes, edges, estimated_nodes = _load_gt_volume(geff_path)
        gt_data[geff_path.stem] = {
            "nodes": nodes,
            "edges": edges,
            "estimated_nodes": estimated_nodes,
        }
    return gt_data


def _prepare_submission(submission: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"}
    missing = sorted(required - set(submission.columns))
    if missing:
        raise ValueError(f"Submission missing columns: {missing}")

    data = submission.copy()
    data["dataset"] = data["dataset"].astype(str)
    data["row_type"] = data["row_type"].astype(str)
    return data


def _match_nodes(pred_nodes: pd.DataFrame, gt_nodes: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    pred_to_gt: dict[int, int] = {}
    gt_to_pred: dict[int, int] = {}

    if pred_nodes.empty or gt_nodes.empty:
        return pred_to_gt, gt_to_pred

    pred_by_t = {int(t): frame.reset_index(drop=True) for t, frame in pred_nodes.groupby("t")}
    gt_by_t = {int(t): frame.reset_index(drop=True) for t, frame in gt_nodes.groupby("t")}

    for t in sorted(set(pred_by_t) & set(gt_by_t)):
        pred_t = pred_by_t[t]
        gt_t = gt_by_t[t]

        pred_coords = pred_t[["z", "y", "x"]].to_numpy(dtype=np.float64) * SCALE
        gt_coords = gt_t[["z", "y", "x"]].to_numpy(dtype=np.float64) * SCALE

        tree = cKDTree(pred_coords)
        candidate_pairs: list[tuple[int, int, float]] = []
        for gt_idx, coord in enumerate(gt_coords):
            pred_indices = tree.query_ball_point(coord, MATCH_UM)
            for pred_idx in pred_indices:
                distance = float(np.linalg.norm(coord - pred_coords[pred_idx]))
                candidate_pairs.append((gt_idx, int(pred_idx), distance))

        if not candidate_pairs:
            continue

        invalid_cost = MATCH_UM + 1.0
        cost = np.full((len(gt_t), len(pred_t)), invalid_cost, dtype=np.float64)
        for gt_idx, pred_idx, distance in candidate_pairs:
            if distance < cost[gt_idx, pred_idx]:
                cost[gt_idx, pred_idx] = distance

        row_idx, col_idx = linear_sum_assignment(cost)
        gt_ids = gt_t["node_id"].to_numpy(dtype=np.int64)
        pred_ids = pred_t["node_id"].to_numpy(dtype=np.int64)
        for row, col in zip(row_idx, col_idx):
            if cost[row, col] <= MATCH_UM:
                gt_id = int(gt_ids[row])
                pred_id = int(pred_ids[col])
                gt_to_pred[gt_id] = pred_id
                pred_to_gt[pred_id] = gt_id

    return pred_to_gt, gt_to_pred


def _edge_set(edges: pd.DataFrame) -> set[tuple[int, int]]:
    if edges.empty:
        return set()
    return {
        (int(row.source_id), int(row.target_id))
        for row in edges[["source_id", "target_id"]].drop_duplicates().itertuples(index=False)
    }


def _jaccard(tp: int, fp: int, fn: int) -> float:
    denominator = tp + fp + fn
    if denominator == 0:
        return 0.0
    return tp / denominator


def _score_edges(
    pred_edges: set[tuple[int, int]],
    gt_edges: set[tuple[int, int]],
    pred_to_gt: dict[int, int],
) -> tuple[int, int, int]:
    mapped_pred_edges = set()
    for pred_source, pred_target in pred_edges:
        gt_source = pred_to_gt.get(pred_source)
        gt_target = pred_to_gt.get(pred_target)
        if gt_source is None or gt_target is None:
            # Edge touches a cell that is not in the (sparse) GT annotation.
            # The official/competitor metric IGNORES these, it does not count
            # them as false positives.
            continue
        mapped_pred_edges.add((gt_source, gt_target))

    tp_edges = mapped_pred_edges & gt_edges
    tp = len(tp_edges)
    fn = len(gt_edges) - tp
    fp = len(mapped_pred_edges - gt_edges)
    return tp, fp, fn


def _score_divisions(
    pred_edges: set[tuple[int, int]],
    gt_edges: set[tuple[int, int]],
    pred_to_gt: dict[int, int],
    gt_to_pred: dict[int, int],
) -> tuple[int, int, int]:
    gt_out: dict[int, set[int]] = {}
    pred_out: dict[int, set[int]] = {}

    for source, target in gt_edges:
        gt_out.setdefault(source, set()).add(target)
    for source, target in pred_edges:
        pred_out.setdefault(source, set()).add(target)

    gt_divisions = {node for node, children in gt_out.items() if len(children) >= 2}
    pred_divisions = {node for node, children in pred_out.items() if len(children) >= 2}

    matched_pred_divisions = {
        pred_node
        for pred_node in pred_divisions
        if pred_to_gt.get(pred_node) in gt_divisions
    }
    matched_gt_divisions = {
        gt_node
        for gt_node in gt_divisions
        if gt_to_pred.get(gt_node) in pred_divisions
    }

    tp = len(matched_gt_divisions)
    fp = len(pred_divisions - matched_pred_divisions)
    fn = len(gt_divisions - matched_gt_divisions)
    return tp, fp, fn


def evaluate_volume(submission: pd.DataFrame, volume: str, gt: dict) -> VolumeMetrics:
    pred_vol = submission[submission["dataset"] == volume]
    pred_nodes = pred_vol[pred_vol["row_type"] == "node"].copy()
    pred_edges_df = pred_vol[pred_vol["row_type"] == "edge"].copy()

    pred_nodes = pred_nodes[["node_id", "t", "z", "y", "x"]].copy()
    for column in ["node_id", "t"]:
        pred_nodes[column] = pred_nodes[column].astype(np.int64)
    for column in ["z", "y", "x"]:
        pred_nodes[column] = pred_nodes[column].astype(np.float64)

    gt_nodes = gt["nodes"]
    gt_edges = gt["edges"]
    estimated_nodes = int(gt["estimated_nodes"])
    pred_edges = _edge_set(pred_edges_df)

    pred_to_gt, gt_to_pred = _match_nodes(pred_nodes, gt_nodes)

    edge_tp, edge_fp, edge_fn = _score_edges(pred_edges, gt_edges, pred_to_gt)
    div_tp, div_fp, div_fn = _score_divisions(pred_edges, gt_edges, pred_to_gt, gt_to_pred)

    edge_jaccard = _jaccard(edge_tp, edge_fp, edge_fn)
    division_jaccard = _jaccard(div_tp, div_fp, div_fn)
    penalty = 1.0
    if len(pred_nodes) > estimated_nodes and len(pred_nodes) > 0:
        penalty = estimated_nodes / len(pred_nodes)
    adjusted_edge = edge_jaccard * penalty
    combined = 0.5 * adjusted_edge + 0.5 * division_jaccard

    return VolumeMetrics(
        volume=volume,
        pred_nodes=len(pred_nodes),
        gt_nodes=len(gt_nodes),
        est_nodes=estimated_nodes,
        matched_nodes=len(pred_to_gt),
        edge_tp=edge_tp,
        edge_fp=edge_fp,
        edge_fn=edge_fn,
        div_tp=div_tp,
        div_fp=div_fp,
        div_fn=div_fn,
        edge_jaccard=edge_jaccard,
        adjusted_edge_jaccard=adjusted_edge,
        division_jaccard=division_jaccard,
        combined_score=combined,
        penalty=penalty,
    )


def compute_metrics(submission_df: pd.DataFrame, train_dir: str | Path) -> dict[str, float | int]:
    submission = _prepare_submission(submission_df)
    volumes = sorted(submission["dataset"].dropna().unique().tolist())
    gt_data = load_gt(train_dir, volumes)
    volume_metrics = [evaluate_volume(submission, volume, gt_data[volume]) for volume in volumes]

    edge_tp = sum(row.edge_tp for row in volume_metrics)
    edge_fp = sum(row.edge_fp for row in volume_metrics)
    edge_fn = sum(row.edge_fn for row in volume_metrics)
    div_tp = sum(row.div_tp for row in volume_metrics)
    div_fp = sum(row.div_fp for row in volume_metrics)
    div_fn = sum(row.div_fn for row in volume_metrics)
    pred_nodes = sum(row.pred_nodes for row in volume_metrics)
    gt_nodes = sum(row.gt_nodes for row in volume_metrics)
    est_nodes = sum(row.est_nodes for row in volume_metrics)
    matched_nodes = sum(row.matched_nodes for row in volume_metrics)

    edge_jaccard = _jaccard(edge_tp, edge_fp, edge_fn)
    division_jaccard = _jaccard(div_tp, div_fp, div_fn)
    penalty = 1.0
    if pred_nodes > est_nodes and pred_nodes > 0:
        penalty = est_nodes / pred_nodes
    adjusted_edge = edge_jaccard * penalty
    combined = 0.5 * adjusted_edge + 0.5 * division_jaccard

    return {
        "edge_jaccard": edge_jaccard,
        "adjusted_edge_jaccard": adjusted_edge,
        "division_jaccard": division_jaccard,
        "combined_score": combined,
        "pred_nodes": pred_nodes,
        "gt_nodes": gt_nodes,
        "est_nodes": est_nodes,
        "matched_nodes": matched_nodes,
        "penalty": penalty,
        "edge_tp": edge_tp,
        "edge_fp": edge_fp,
        "edge_fn": edge_fn,
        "div_tp": div_tp,
        "div_fp": div_fp,
        "div_fn": div_fn,
        "volumes": len(volume_metrics),
        "per_volume": volume_metrics,
    }


def print_metrics(metrics: dict[str, float | int]) -> None:
    print("Per-volume metrics")
    print(
        "volume,pred_nodes,gt_nodes,est_nodes,matched_nodes,"
        "edge_tp,edge_fp,edge_fn,edge_jaccard,adjusted_edge_jaccard,"
        "div_tp,div_fp,div_fn,division_jaccard,combined_score,penalty"
    )
    for row in metrics["per_volume"]:
        print(
            f"{row.volume},{row.pred_nodes},{row.gt_nodes},{row.est_nodes},"
            f"{row.matched_nodes},{row.edge_tp},{row.edge_fp},{row.edge_fn},"
            f"{row.edge_jaccard:.6f},{row.adjusted_edge_jaccard:.6f},"
            f"{row.div_tp},{row.div_fp},{row.div_fn},{row.division_jaccard:.6f},"
            f"{row.combined_score:.6f},{row.penalty:.6f}"
        )

    print()
    print(f"Edge Jaccard: {metrics['edge_jaccard']:.4f}")
    print(f"Adjusted Edge Jaccard: {metrics['adjusted_edge_jaccard']:.4f}")
    print(f"Division Jaccard: {metrics['division_jaccard']:.4f}")
    print(f"Combined Score: {metrics['combined_score']:.4f}")
    print(f"Penalty: {metrics['penalty']:.4f}")
    print(f"Pred nodes: {metrics['pred_nodes']}")
    print(f"GT nodes: {metrics['gt_nodes']}")
    print(f"Est nodes: {metrics['est_nodes']}")
    print(f"Matched nodes: {metrics['matched_nodes']}")
    print(f"Edge counts: tp={metrics['edge_tp']} fp={metrics['edge_fp']} fn={metrics['edge_fn']}")
    print(f"Division counts: tp={metrics['div_tp']} fp={metrics['div_fp']} fn={metrics['div_fn']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--train-dir", required=True)
    args = parser.parse_args()

    submission = pd.read_csv(args.submission)
    metrics = compute_metrics(submission, args.train_dir)
    print_metrics(metrics)


if __name__ == "__main__":
    main()

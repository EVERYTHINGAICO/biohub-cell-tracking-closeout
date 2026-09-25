from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import geff
import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


MATCH_DISTANCE_UM = 7.0
EDGE_ADJUSTMENT_ALPHA = 0.1
DIVISION_WEIGHT = 0.1
PHYSICAL_SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)


@dataclass(frozen=True)
class TrackingGraph:
    dataset: str
    nodes: pd.DataFrame
    edges: pd.DataFrame
    estimated_number_of_nodes: int | None = None


@dataclass(frozen=True)
class NodeMatches:
    pred_to_gt: dict[int, int]
    gt_to_pred: dict[int, int]


@dataclass(frozen=True)
class CountMetrics:
    tp: int
    fp: int
    fn: int

    @property
    def denominator(self) -> int:
        return self.tp + self.fp + self.fn

    @property
    def jaccard(self) -> float:
        if self.denominator == 0:
            return 0.0
        return self.tp / self.denominator


@dataclass(frozen=True)
class DatasetMetrics:
    dataset: str
    pred_nodes: int
    gt_nodes: int
    estimated_nodes: int
    matched_nodes: int
    edge_counts: CountMetrics
    division_counts: CountMetrics
    edge_jaccard: float
    adjusted_edge_jaccard: float
    requested_adjusted_edge_jaccard: float
    division_jaccard: float
    official_like_score: float
    requested_combined_score: float


def read_gt_geff(path: str | Path) -> TrackingGraph:
    geff_path = Path(path)
    graph, metadata = geff.read(geff_path, structure_validation=False)

    node_rows = []
    for node_id, attrs in graph.nodes(data=True):
        node_rows.append(
            {
                "node_id": int(node_id),
                "t": int(attrs["t"]),
                "z": float(attrs["z"]),
                "y": float(attrs["y"]),
                "x": float(attrs["x"]),
            }
        )

    edge_rows = [
        {"source_id": int(source), "target_id": int(target)}
        for source, target in graph.edges()
    ]

    estimated_nodes = None
    extra = getattr(metadata, "extra", None)
    if isinstance(extra, dict) and "estimated_number_of_nodes" in extra:
        estimated_nodes = int(extra["estimated_number_of_nodes"])

    return TrackingGraph(
        dataset=geff_path.stem,
        nodes=pd.DataFrame(node_rows, columns=["node_id", "t", "z", "y", "x"]),
        edges=pd.DataFrame(edge_rows, columns=["source_id", "target_id"]),
        estimated_number_of_nodes=estimated_nodes,
    )


def read_submission_graph(submission: pd.DataFrame, dataset: str) -> TrackingGraph:
    required = {"dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"}
    missing = sorted(required - set(submission.columns))
    if missing:
        raise ValueError(f"Submission missing columns: {missing}")

    data = submission[submission["dataset"].astype(str) == str(dataset)].copy()
    node_rows = data[data["row_type"] == "node"].copy()
    edge_rows = data[data["row_type"] == "edge"].copy()

    nodes = node_rows[["node_id", "t", "z", "y", "x"]].copy()
    for column in ["node_id", "t"]:
        nodes[column] = nodes[column].astype(np.int64)
    for column in ["z", "y", "x"]:
        nodes[column] = nodes[column].astype(np.float64)

    edges = edge_rows[["source_id", "target_id"]].copy()
    for column in ["source_id", "target_id"]:
        edges[column] = edges[column].astype(np.int64)

    valid_nodes = set(nodes["node_id"].tolist())
    edges = edges[
        edges["source_id"].isin(valid_nodes) & edges["target_id"].isin(valid_nodes)
    ].drop_duplicates()

    return TrackingGraph(
        dataset=dataset,
        nodes=nodes.reset_index(drop=True),
        edges=edges.reset_index(drop=True),
        estimated_number_of_nodes=None,
    )


def match_nodes(
    pred_nodes: pd.DataFrame,
    gt_nodes: pd.DataFrame,
    max_distance_um: float = MATCH_DISTANCE_UM,
) -> NodeMatches:
    pred_to_gt: dict[int, int] = {}
    gt_to_pred: dict[int, int] = {}

    if pred_nodes.empty or gt_nodes.empty:
        return NodeMatches(pred_to_gt=pred_to_gt, gt_to_pred=gt_to_pred)

    pred_by_t = {int(t): g for t, g in pred_nodes.groupby("t", sort=False)}
    gt_by_t = {int(t): g for t, g in gt_nodes.groupby("t", sort=False)}

    for t in sorted(set(pred_by_t) & set(gt_by_t)):
        pred_t = pred_by_t[t].reset_index(drop=True)
        gt_t = gt_by_t[t].reset_index(drop=True)
        pred_coords = pred_t[["z", "y", "x"]].to_numpy(dtype=np.float64) * PHYSICAL_SCALE
        gt_coords = gt_t[["z", "y", "x"]].to_numpy(dtype=np.float64) * PHYSICAL_SCALE

        tree = cKDTree(gt_coords)
        candidates: list[tuple[int, int, float]] = []
        for pred_idx, coord in enumerate(pred_coords):
            gt_indices = tree.query_ball_point(coord, max_distance_um)
            for gt_idx in gt_indices:
                distance = float(np.linalg.norm(coord - gt_coords[gt_idx]))
                candidates.append((pred_idx, gt_idx, distance))

        if not candidates:
            continue

        invalid_cost = max_distance_um + 1.0
        costs = np.full((len(pred_t), len(gt_t)), invalid_cost, dtype=np.float64)
        for pred_idx, gt_idx, distance in candidates:
            if distance < costs[pred_idx, gt_idx]:
                costs[pred_idx, gt_idx] = distance

        row_ind, col_ind = linear_sum_assignment(costs)
        pred_ids = pred_t["node_id"].to_numpy(dtype=np.int64)
        gt_ids = gt_t["node_id"].to_numpy(dtype=np.int64)
        for row, col in zip(row_ind, col_ind):
            if costs[row, col] <= max_distance_um:
                pred_id = int(pred_ids[row])
                gt_id = int(gt_ids[col])
                pred_to_gt[pred_id] = gt_id
                gt_to_pred[gt_id] = pred_id

    return NodeMatches(pred_to_gt=pred_to_gt, gt_to_pred=gt_to_pred)


def _edge_set(edges: pd.DataFrame) -> set[tuple[int, int]]:
    if edges.empty:
        return set()
    return {
        (int(row.source_id), int(row.target_id))
        for row in edges[["source_id", "target_id"]].itertuples(index=False)
    }


def calculate_edge_jaccard(
    pred: TrackingGraph,
    gt: TrackingGraph,
    matches: NodeMatches,
) -> CountMetrics:
    gt_edges = _edge_set(gt.edges)
    pred_edges = _edge_set(pred.edges)

    gt_predecessors: dict[int, set[int]] = defaultdict(set)
    gt_successors: dict[int, set[int]] = defaultdict(set)
    for source, target in gt_edges:
        gt_successors[source].add(target)
        gt_predecessors[target].add(source)

    matched_gt_edges: set[tuple[int, int]] = set()
    fp = 0

    for pred_source, pred_target in pred_edges:
        gt_source = matches.pred_to_gt.get(pred_source)
        gt_target = matches.pred_to_gt.get(pred_target)

        if gt_source is not None and gt_target is not None and (gt_source, gt_target) in gt_edges:
            matched_gt_edges.add((gt_source, gt_target))
            continue

        is_fp = False
        if gt_target is not None and gt_predecessors.get(gt_target):
            if gt_source not in gt_predecessors[gt_target]:
                is_fp = True
        if gt_source is not None and gt_successors.get(gt_source):
            if gt_target not in gt_successors[gt_source]:
                is_fp = True
        if is_fp:
            fp += 1

    tp = len(matched_gt_edges)
    fn = len(gt_edges) - tp
    return CountMetrics(tp=tp, fp=fp, fn=fn)


def _to_digraph(graph: TrackingGraph) -> nx.DiGraph:
    digraph = nx.DiGraph()
    digraph.add_nodes_from(int(node_id) for node_id in graph.nodes["node_id"].tolist())
    digraph.add_edges_from(_edge_set(graph.edges))
    return digraph


def calculate_division_jaccard(
    pred: TrackingGraph,
    gt: TrackingGraph,
    matches: NodeMatches,
) -> CountMetrics:
    pred_graph = _to_digraph(pred)
    gt_graph = _to_digraph(gt)

    gt_divisions = [node for node in gt_graph.nodes if gt_graph.out_degree(node) == 2]
    pred_divisions = [node for node in pred_graph.nodes if pred_graph.out_degree(node) == 2]

    if not gt_divisions and not pred_divisions:
        return CountMetrics(tp=0, fp=0, fn=0)

    component_by_node: dict[int, int] = {}
    component_has_fork: dict[int, bool] = {}
    for component_id, component in enumerate(nx.connected_components(pred_graph.to_undirected())):
        component_set = set(component)
        for node in component_set:
            component_by_node[int(node)] = component_id
        component_has_fork[component_id] = any(
            pred_graph.out_degree(node) == 2 for node in component_set
        )

    paired_pred_divisions: set[int] = set()
    tp = 0

    for gt_division in gt_divisions:
        pre_split_gt = {gt_division, *gt_graph.predecessors(gt_division)}
        children = list(gt_graph.successors(gt_division))
        daughter_lineages = [
            {child, *nx.descendants(gt_graph, child)}
            for child in children
        ]

        pre_split_pred = [
            matches.gt_to_pred[node]
            for node in pre_split_gt
            if node in matches.gt_to_pred
        ]
        daughter_pred_hits = [
            [
                matches.gt_to_pred[node]
                for node in lineage
                if node in matches.gt_to_pred
            ]
            for lineage in daughter_lineages
        ]

        if not pre_split_pred or any(not hits for hits in daughter_pred_hits):
            continue

        matched_pred_nodes = set(pre_split_pred)
        for hits in daughter_pred_hits:
            matched_pred_nodes.update(hits)

        components = {
            component_by_node[node]
            for node in matched_pred_nodes
            if node in component_by_node
        }
        if len(components) != 1:
            continue

        component_id = next(iter(components))
        if not component_has_fork.get(component_id, False):
            continue

        tp += 1
        for pred_division in pred_divisions:
            if component_by_node.get(pred_division) == component_id:
                paired_pred_divisions.add(int(pred_division))
                break

    fn = len(gt_divisions) - tp
    fp = 0
    for pred_division in pred_divisions:
        matched_gt = matches.pred_to_gt.get(pred_division)
        if matched_gt is None:
            continue
        if gt_graph.out_degree(matched_gt) > 0 and pred_division not in paired_pred_divisions:
            fp += 1

    return CountMetrics(tp=tp, fp=fp, fn=fn)


def adjust_edge_jaccard(
    raw_jaccard: float,
    n_pred_nodes: int,
    estimated_true_nodes: int,
    alpha: float = EDGE_ADJUSTMENT_ALPHA,
) -> float:
    if estimated_true_nodes <= 0:
        return raw_jaccard
    factor = 1.0 - alpha * ((n_pred_nodes - estimated_true_nodes) / estimated_true_nodes)
    return max(0.0, raw_jaccard * factor)


def requested_overprediction_adjustment(
    raw_jaccard: float,
    n_pred_nodes: int,
    estimated_true_nodes: int,
) -> float:
    if n_pred_nodes <= 0 or estimated_true_nodes <= 0:
        return raw_jaccard
    if n_pred_nodes <= estimated_true_nodes:
        return raw_jaccard
    return raw_jaccard * (estimated_true_nodes / n_pred_nodes)


def evaluate_dataset(pred: TrackingGraph, gt: TrackingGraph) -> DatasetMetrics:
    estimated_nodes = gt.estimated_number_of_nodes or len(gt.nodes)
    matches = match_nodes(pred.nodes, gt.nodes)
    edge_counts = calculate_edge_jaccard(pred, gt, matches)
    division_counts = calculate_division_jaccard(pred, gt, matches)

    edge_jaccard = edge_counts.jaccard
    adjusted_edge = adjust_edge_jaccard(
        edge_jaccard,
        n_pred_nodes=len(pred.nodes),
        estimated_true_nodes=estimated_nodes,
    )
    requested_adjusted = requested_overprediction_adjustment(
        edge_jaccard,
        n_pred_nodes=len(pred.nodes),
        estimated_true_nodes=estimated_nodes,
    )
    division_jaccard = division_counts.jaccard

    return DatasetMetrics(
        dataset=gt.dataset,
        pred_nodes=len(pred.nodes),
        gt_nodes=len(gt.nodes),
        estimated_nodes=estimated_nodes,
        matched_nodes=len(matches.pred_to_gt),
        edge_counts=edge_counts,
        division_counts=division_counts,
        edge_jaccard=edge_jaccard,
        adjusted_edge_jaccard=adjusted_edge,
        requested_adjusted_edge_jaccard=requested_adjusted,
        division_jaccard=division_jaccard,
        official_like_score=adjusted_edge + DIVISION_WEIGHT * division_jaccard,
        requested_combined_score=0.5 * requested_adjusted + 0.5 * division_jaccard,
    )


def aggregate_metrics(metrics: Iterable[DatasetMetrics]) -> dict[str, float | int]:
    rows = list(metrics)
    edge_weight_sum = sum(row.edge_counts.denominator for row in rows)
    adjusted_edge = 0.0
    requested_adjusted_edge = 0.0
    if edge_weight_sum > 0:
        adjusted_edge = sum(
            row.adjusted_edge_jaccard * row.edge_counts.denominator
            for row in rows
        ) / edge_weight_sum
        requested_adjusted_edge = sum(
            row.requested_adjusted_edge_jaccard * row.edge_counts.denominator
            for row in rows
        ) / edge_weight_sum

    edge_tp = sum(row.edge_counts.tp for row in rows)
    edge_fp = sum(row.edge_counts.fp for row in rows)
    edge_fn = sum(row.edge_counts.fn for row in rows)
    edge_den = edge_tp + edge_fp + edge_fn
    raw_edge = edge_tp / edge_den if edge_den else 0.0

    div_tp = sum(row.division_counts.tp for row in rows)
    div_fp = sum(row.division_counts.fp for row in rows)
    div_fn = sum(row.division_counts.fn for row in rows)
    div_den = div_tp + div_fp + div_fn
    division_jaccard = div_tp / div_den if div_den else 0.0

    return {
        "datasets": len(rows),
        "pred_nodes": sum(row.pred_nodes for row in rows),
        "gt_nodes": sum(row.gt_nodes for row in rows),
        "estimated_nodes": sum(row.estimated_nodes for row in rows),
        "matched_nodes": sum(row.matched_nodes for row in rows),
        "edge_tp": edge_tp,
        "edge_fp": edge_fp,
        "edge_fn": edge_fn,
        "edge_jaccard": raw_edge,
        "adjusted_edge_jaccard": adjusted_edge,
        "requested_adjusted_edge_jaccard": requested_adjusted_edge,
        "division_tp": div_tp,
        "division_fp": div_fp,
        "division_fn": div_fn,
        "division_jaccard": division_jaccard,
        "official_like_score": adjusted_edge + DIVISION_WEIGHT * division_jaccard,
        "requested_combined_score": 0.5 * requested_adjusted_edge + 0.5 * division_jaccard,
    }


def evaluate_submission(
    submission_csv: str | Path,
    train_dir: str | Path,
    datasets: Iterable[str] | None = None,
) -> tuple[list[DatasetMetrics], dict[str, float | int]]:
    submission = pd.read_csv(submission_csv)
    train_path = Path(train_dir)

    if datasets is None:
        dataset_names = sorted(str(name) for name in submission["dataset"].dropna().unique())
    else:
        dataset_names = sorted(str(name) for name in datasets)

    results: list[DatasetMetrics] = []
    for dataset in dataset_names:
        geff_path = train_path / f"{dataset}.geff"
        if not geff_path.exists():
            raise FileNotFoundError(f"GT GEFF not found for dataset {dataset}: {geff_path}")
        gt = read_gt_geff(geff_path)
        pred = read_submission_graph(submission, dataset)
        results.append(evaluate_dataset(pred, gt))

    return results, aggregate_metrics(results)

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class WalkForwardValidator:
    def __init__(self, total_frames=100, train_frames=80, val_frames=20, stride=10, match_radius_um=7.0, physical_scale=(1.625, 0.40625, 0.40625)):
        self.total_frames = int(total_frames)
        self.train_frames = int(train_frames)
        self.val_frames = int(val_frames)
        self.stride = int(stride)
        self.match_radius_um = float(match_radius_um)
        self.physical_scale = np.array(physical_scale)
        self.splits = self._build_splits()

    def _build_splits(self):
        splits = []
        start = 0
        while start + self.train_frames + self.val_frames <= self.total_frames:
            train_idx = list(range(start, start + self.train_frames))
            val_idx = list(range(start + self.train_frames, start + self.train_frames + self.val_frames))
            splits.append({"train": train_idx, "val": val_idx, "fold": len(splits)})
            start += self.stride
        if not splits:
            train_idx = list(range(0, self.train_frames))
            val_idx = list(range(self.train_frames, self.train_frames + self.val_frames))
            splits.append({"train": train_idx, "val": val_idx, "fold": 0})
        return splits

    def get_splits(self):
        return self.splits

    def _match_nodes(self, pred_nodes, gt_nodes):
        if len(pred_nodes) == 0 or len(gt_nodes) == 0:
            return 0, len(pred_nodes), len(gt_nodes)
        pred_pos = pred_nodes[["z", "y", "x"]].values * self.physical_scale
        gt_pos = gt_nodes[["z", "y", "x"]].values * self.physical_scale
        cost = cdist(pred_pos, gt_pos)
        cost_masked = np.where(cost <= self.match_radius_um, cost, 1e9)
        row_ind, col_ind = linear_sum_assignment(cost_masked)
        tp = sum(1 for r, c in zip(row_ind, col_ind) if cost_masked[r, c] < 1e9)
        fp = len(pred_nodes) - tp
        fn = len(gt_nodes) - tp
        return tp, fp, fn

    def _compute_f1(self, tp, fp, fn):
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _match_edges(self, pred_edges, gt_edges, node_mapping):
        if len(pred_edges) == 0 or len(gt_edges) == 0:
            return 0, len(pred_edges), len(gt_edges)
        pred_set = set()
        for _, row in pred_edges.iterrows():
            src = node_mapping.get(int(row["source_id"]))
            tgt = node_mapping.get(int(row["target_id"]))
            if src is not None and tgt is not None:
                pred_set.add((src, tgt))
        gt_set = set((int(row["source_id"]), int(row["target_id"])) for _, row in gt_edges.iterrows())
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        return tp, fp, fn

    def score_fold(self, fold_idx, pred_df, gt_df):
        split = self.splits[fold_idx]
        val_frames = split["val"]
        pred_val = pred_df[pred_df["t"].isin(val_frames)]
        gt_val = gt_df[gt_df["t"].isin(val_frames)]
        pred_nodes = pred_val[pred_val["row_type"] == "node"]
        gt_nodes = gt_val[gt_val["row_type"] == "node"]
        pred_edges = pred_val[pred_val["row_type"] == "edge"]
        gt_edges = gt_val[gt_val["row_type"] == "edge"]
        node_tp, node_fp, node_fn = self._match_nodes(pred_nodes, gt_nodes)
        node_f1 = self._compute_f1(node_tp, node_fp, node_fn)
        edge_tp, edge_fp, edge_fn = self._match_edges(pred_edges, gt_edges, {})
        edge_f1 = self._compute_f1(edge_tp, edge_fp, edge_fn)
        return {
            "fold": fold_idx,
            "val_frames": val_frames,
            "node_tp": node_tp,
            "node_fp": node_fp,
            "node_fn": node_fn,
            "node_f1": node_f1,
            "edge_tp": edge_tp,
            "edge_fp": edge_fp,
            "edge_fn": edge_fn,
            "edge_f1": edge_f1,
        }

    def aggregate_scores(self, fold_scores):
        node_f1s = [s["node_f1"] for s in fold_scores]
        edge_f1s = [s["edge_f1"] for s in fold_scores]
        return {
            "num_folds": len(fold_scores),
            "node_f1_mean": np.mean(node_f1s),
            "node_f1_std": np.std(node_f1s),
            "node_f1_ci95": (np.percentile(node_f1s, 2.5), np.percentile(node_f1s, 97.5)),
            "edge_f1_mean": np.mean(edge_f1s),
            "edge_f1_std": np.std(edge_f1s),
            "edge_f1_ci95": (np.percentile(edge_f1s, 2.5), np.percentile(edge_f1s, 97.5)),
        }

if __name__ == "__main__":
    validator = WalkForwardValidator(total_frames=100, train_frames=80, val_frames=20, stride=10)
    splits = validator.get_splits()
    print(f"Number of folds: {len(splits)}")
    for s in splits:
        print(f"Fold {s['fold']}: train={s['train'][0]}-{s['train'][-1]}, val={s['val'][0]}-{s['val'][-1]}")

    np.random.seed(42)
    gt_nodes = []
    for t in range(100):
        for i in range(5):
            gt_nodes.append({"dataset": "test", "row_type": "node", "node_id": t * 10 + i, "t": t, "z": 10.0 + i, "y": 20.0 + i, "x": 30.0 + i, "source_id": -1, "target_id": -1})
    gt_df = pd.DataFrame(gt_nodes)

    pred_nodes = []
    for t in range(100):
        for i in range(5):
            noise = np.random.randn(3) * 0.5
            pred_nodes.append({"dataset": "test", "row_type": "node", "node_id": t * 10 + i, "t": t, "z": 10.0 + i + noise[0], "y": 20.0 + i + noise[1], "x": 30.0 + i + noise[2], "source_id": -1, "target_id": -1})
    pred_df = pd.DataFrame(pred_nodes)

    fold_scores = []
    for fold_idx in range(len(splits)):
        score = validator.score_fold(fold_idx, pred_df, gt_df)
        fold_scores.append(score)
        print(f"Fold {fold_idx}: node_f1={score['node_f1']:.3f}, edge_f1={score['edge_f1']:.3f}")

    agg = validator.aggregate_scores(fold_scores)
    print(f"\nAggregated:")
    print(f"  node_f1: {agg['node_f1_mean']:.3f} ± {agg['node_f1_std']:.3f} (CI95: {agg['node_f1_ci95'][0]:.3f}-{agg['node_f1_ci95'][1]:.3f})")
    print(f"  edge_f1: {agg['edge_f1_mean']:.3f} ± {agg['edge_f1_std']:.3f} (CI95: {agg['edge_f1_ci95'][0]:.3f}-{agg['edge_f1_ci95'][1]:.3f})")

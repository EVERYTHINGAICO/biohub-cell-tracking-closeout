from __future__ import annotations
import argparse
import pandas as pd


SUBMISSION_COLUMNS = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]


def _edge_row(volume, source_id, target_id):
    return {
        "dataset": volume,
        "row_type": "edge",
        "node_id": -1,
        "t": -1,
        "z": -1,
        "y": -1,
        "x": -1,
        "source_id": source_id,
        "target_id": target_id,
    }


def format_submission(tracks_df, output_path, division_edges_df=None):
    rows = []

    for volume, volume_df in tracks_df.groupby("volume", sort=True):
        node_id_counter = 1
        node_id_by_track_time = {}
        emitted_edges = set()

        for track_id, track_data in volume_df.groupby("track_id", sort=True):
            track_data = track_data.sort_values("t")
            node_ids = []

            for _, row in track_data.iterrows():
                node_id = node_id_counter
                node_id_counter += 1
                node_ids.append(node_id)
                node_id_by_track_time[(int(track_id), int(row["t"]))] = node_id
                rows.append(
                    {
                        "dataset": volume,
                        "row_type": "node",
                        "node_id": node_id,
                        "t": int(row["t"]),
                        "z": int(round(float(row["z"]))),
                        "y": int(round(float(row["y"]))),
                        "x": int(round(float(row["x"]))),
                        "source_id": -1,
                        "target_id": -1,
                    }
                )

            for source_id, target_id in zip(node_ids[:-1], node_ids[1:]):
                edge_key = (source_id, target_id)
                if edge_key in emitted_edges:
                    continue
                emitted_edges.add(edge_key)
                rows.append(_edge_row(volume, source_id, target_id))

        if division_edges_df is not None and len(division_edges_df) > 0:
            volume_edges = division_edges_df[division_edges_df["volume"] == volume]
            skipped_edges = 0
            for _, edge in volume_edges.iterrows():
                source_key = (int(edge["source_track_id"]), int(edge["source_t"]))
                target_key = (int(edge["target_track_id"]), int(edge["target_t"]))
                source_id = node_id_by_track_time.get(source_key)
                target_id = node_id_by_track_time.get(target_key)
                if source_id is None or target_id is None:
                    skipped_edges += 1
                    continue
                edge_key = (source_id, target_id)
                if edge_key in emitted_edges:
                    continue
                emitted_edges.add(edge_key)
                rows.append(_edge_row(volume, source_id, target_id))

            if skipped_edges:
                print(f"Skipped {skipped_edges} division edges for {volume}: missing node mapping")

    submission_df = pd.DataFrame(rows)
    submission_df.insert(0, "id", range(len(submission_df)))
    submission_df = submission_df[SUBMISSION_COLUMNS]
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Total rows: {len(submission_df)}")
    print(f"Nodes: {len(submission_df[submission_df['row_type'] == 'node'])}")
    print(f"Edges: {len(submission_df[submission_df['row_type'] == 'edge'])}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", type=str, required=True)
    parser.add_argument("--division-edges", type=str, default=None)
    parser.add_argument("--output", type=str, default="submission.csv")
    args = parser.parse_args()

    tracks_df = pd.read_csv(args.tracks)
    division_edges_df = None
    if args.division_edges:
        division_edges_df = pd.read_csv(args.division_edges)
    format_submission(tracks_df, args.output, division_edges_df=division_edges_df)


if __name__ == "__main__":
    main()

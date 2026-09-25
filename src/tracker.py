from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


class KalmanFilter:
    def __init__(self, initial_pos, process_noise=1.0, measurement_noise=1.0):
        self.pos = np.array(initial_pos, dtype=float)
        self.vel = np.zeros(3, dtype=float)
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.covariance = np.eye(3) * process_noise

    def predict(self):
        self.pos = self.pos + self.vel
        self.covariance = self.covariance + np.eye(3) * self.process_noise
        return self.pos

    def update(self, measurement):
        measurement = np.array(measurement, dtype=float)
        kalman_gain = self.covariance @ np.linalg.inv(
            self.covariance + np.eye(3) * self.measurement_noise
        )
        innovation = measurement - self.pos
        self.pos = self.pos + kalman_gain @ innovation
        self.vel = self.vel + kalman_gain @ innovation
        self.covariance = (np.eye(3) - kalman_gain) @ self.covariance
        return self.pos


def track_cells(predictions_df, max_distance=50.0):
    tracks = []
    track_id_counter = 0

    for volume, volume_df in predictions_df.groupby("volume", sort=True):
        active_tracks = {}

        for t in sorted(volume_df["t"].unique()):
            frame_detections = volume_df[volume_df["t"] == t]
            current_positions = frame_detections[["z", "y", "x"]].to_numpy(dtype=float)
            current_scores = frame_detections["score"].to_numpy(dtype=float)

            if len(active_tracks) == 0:
                for pos, score in zip(current_positions, current_scores):
                    track_id_counter += 1
                    active_tracks[track_id_counter] = {
                        "kalman": KalmanFilter(pos),
                        "last_pos": pos,
                    }
                    tracks.append(
                        {
                            "volume": volume,
                            "t": int(t),
                            "z": pos[0],
                            "y": pos[1],
                            "x": pos[2],
                            "score": score,
                            "track_id": track_id_counter,
                        }
                    )
                continue

            predicted_positions = []
            track_ids = []
            for track_id, track_data in active_tracks.items():
                predicted_positions.append(track_data["kalman"].predict())
                track_ids.append(track_id)

            cost_matrix = cdist(np.array(predicted_positions), current_positions)
            cost_matrix[cost_matrix > max_distance] = 1e6
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            matched_tracks = set()
            matched_detections = set()

            for row_idx, col_idx in zip(row_indices, col_indices):
                if cost_matrix[row_idx, col_idx] < max_distance:
                    track_id = track_ids[row_idx]
                    detection_pos = current_positions[col_idx]
                    score = current_scores[col_idx]

                    active_tracks[track_id]["kalman"].update(detection_pos)
                    active_tracks[track_id]["last_pos"] = detection_pos
                    tracks.append(
                        {
                            "volume": volume,
                            "t": int(t),
                            "z": detection_pos[0],
                            "y": detection_pos[1],
                            "x": detection_pos[2],
                            "score": score,
                            "track_id": track_id,
                        }
                    )
                    matched_tracks.add(track_id)
                    matched_detections.add(col_idx)

            for col_idx, pos in enumerate(current_positions):
                if col_idx not in matched_detections:
                    track_id_counter += 1
                    active_tracks[track_id_counter] = {
                        "kalman": KalmanFilter(pos),
                        "last_pos": pos,
                    }
                    tracks.append(
                        {
                            "volume": volume,
                            "t": int(t),
                            "z": pos[0],
                            "y": pos[1],
                            "x": pos[2],
                            "score": current_scores[col_idx],
                            "track_id": track_id_counter,
                        }
                    )
                    matched_tracks.add(track_id_counter)

            active_tracks = {
                track_id: track_data
                for track_id, track_data in active_tracks.items()
                if track_id in matched_tracks
            }

    return pd.DataFrame(tracks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output", type=str, default="tracks.csv")
    parser.add_argument("--max-distance", type=float, default=50.0)
    args = parser.parse_args()

    predictions_df = pd.read_csv(args.predictions)
    print(f"Loaded {len(predictions_df)} detections")

    if len(predictions_df) == 0:
        print("ERROR: No detections found. Cannot track.")
        return

    tracks_df = track_cells(predictions_df, max_distance=args.max_distance)
    tracks_df.to_csv(args.output, index=False)
    print(f"Tracks saved to {args.output}")
    print(f"Total track points: {len(tracks_df)}")
    print(f"Unique tracks: {tracks_df['track_id'].nunique()}")


if __name__ == "__main__":
    main()

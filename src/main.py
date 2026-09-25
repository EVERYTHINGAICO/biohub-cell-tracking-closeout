from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from data_loader import create_train_loader, ZarrCellDataset
from unet3d import UNet3D

def load_ground_truth_for_volume(zarr_path):
    geff_dir = Path(str(zarr_path).replace('.zarr', '.geff'))
    if not geff_dir.exists():
        return None
    try:
        import geff
        result = geff.read(geff_dir)
        graph = result[0] if isinstance(result, tuple) else result
        node_list = []
        for node_id, node_data in graph.nodes(data=True):
            node_list.append({
                'node_id': node_id,
                't': node_data.get('t', 0),
                'z': node_data.get('z', 0.0),
                'y': node_data.get('y', 0.0),
                'x': node_data.get('x', 0.0),
                'row_type': 'node'
            })
        return pd.DataFrame(node_list)
    except Exception as e:
        print(f"Warning: Could not load GT from {geff_dir}: {e}")
        return None

def create_target_from_gt(patch_coords, patch_size, gt_df, t_index, sigma=2.0):
    target = torch.zeros((1, *patch_size), dtype=torch.float32)
    if gt_df is None or len(gt_df) == 0:
        return target
    z0, y0, x0 = patch_coords
    pz, py, px = patch_size
    nodes = gt_df[gt_df['row_type'] == 'node']
    nodes_at_t = nodes[nodes['t'] == t_index]
    for _, row in nodes_at_t.iterrows():
        gz, gy, gx = float(row['z']), float(row['y']), float(row['x'])
        if (z0 <= gz < z0 + pz and y0 <= gy < y0 + py and x0 <= gx < x0 + px):
            local_z = int(gz - z0)
            local_y = int(gy - y0)
            local_x = int(gx - x0)
            radius = int(sigma * 3)
            for dz in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        cz, cy, cx = local_z + dz, local_y + dy, local_x + dx
                        if 0 <= cz < pz and 0 <= cy < py and 0 <= cx < px:
                            dist_sq = dz*dz + dy*dy + dx*dx
                            target[0, cz, cy, cx] = max(
                                target[0, cz, cy, cx],
                                np.exp(-dist_sq / (2 * sigma * sigma))
                            )
    return target

def train_model(train_loader, train_dataset, device, epochs=5, lr=1e-3):
    model = UNet3D(gradient_checkpointing=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    print(f"Training on {device}")
    print(f"Total training samples: {len(train_dataset)}")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        batches_seen = 0
        max_batches_per_epoch = 5000
        for batch_idx, (patch_tensor, coords, t_index) in enumerate(train_loader):
            if batch_idx >= max_batches_per_epoch:
                break
            optimizer.zero_grad()
            patch_tensor = patch_tensor.to(device)
            if patch_tensor.ndim == 4:
                patch_tensor = patch_tensor.unsqueeze(0)
            volume_idx = batch_idx // max(1, len(train_dataset) // len(train_dataset.zarr_paths))
            volume_idx = min(volume_idx, len(train_dataset.zarr_paths) - 1)
            zarr_path = train_dataset.zarr_paths[volume_idx]
            gt_df = load_ground_truth_for_volume(zarr_path)
            target = create_target_from_gt(coords, tuple(patch_tensor.shape[2:]), gt_df, t_index)
            if target.ndim == 4:
                target = target.unsqueeze(0)
            target = target.to(device)
            if device.type == 'cuda' and scaler is not None:
                with torch.amp.autocast(device_type='cuda'):
                    logits, probabilities = model(patch_tensor)
                    loss = criterion(logits, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, probabilities = model(patch_tensor)
                loss = criterion(logits, target)
                loss.backward()
                optimizer.step()
            train_loss += loss.item()
            batches_seen += 1
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        avg_loss = train_loss / max(1, batches_seen)
        print(f"Epoch {epoch} completed, Avg Loss: {avg_loss:.4f}")
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")
            print(f"Model saved to model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), "model_final.pth")
    print("Final model saved to model_final.pth")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    train_loader = create_train_loader(args.train_dir, patch_size=(64, 64, 64))
    train_dataset = train_loader.dataset
    print(f"Train batches: {len(train_loader)}")
    model = train_model(train_loader, train_dataset, device, epochs=args.epochs)
    print("Training complete.")

if __name__ == "__main__":
    main()

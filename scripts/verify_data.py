from pathlib import Path

train_dir = Path("/mnt/d/descargas al dico vergas/biohub-cell-tracking-during-development/train")
test_dir = Path("/mnt/d/descargas al dico vergas/biohub-cell-tracking-during-development/test")

zarr_files = sorted([p for p in train_dir.iterdir() if p.is_dir() and p.suffix == ".zarr"])
geff_files = sorted([p for p in train_dir.iterdir() if p.is_dir() and p.suffix == ".geff"])
test_zarr = sorted([p for p in test_dir.iterdir() if p.is_dir() and p.suffix == ".zarr"])

print(f"Train .zarr: {len(zarr_files)}")
print(f"Train .geff: {len(geff_files)}")
print(f"Test .zarr:  {len(test_zarr)}")

print("\nFirst 3 train .zarr:")
for p in zarr_files[:3]:
    print(f"  {p.name}")

print("\nFirst 3 test .zarr:")
for p in test_zarr[:3]:
    print(f"  {p.name}")

# Verificar que cada .zarr tiene su .geff correspondiente
zarr_names = {p.stem for p in zarr_files}
geff_names = {p.stem for p in geff_files}
missing_geff = zarr_names - geff_names
if missing_geff:
    print(f"\nWARNING: Missing .geff for: {missing_geff}")
else:
    print(f"\nAll {len(zarr_files)} .zarr have matching .geff ✓")

# Verificar estructura de un .zarr
import zarr
sample = zarr.open(str(zarr_files[0]), mode="r")
arr = sample["0"]
print(f"\nSample volume: {zarr_files[0].name}")
print(f"  shape: {arr.shape}")
print(f"  dtype: {arr.dtype}")
print(f"  chunks: {arr.chunks}")

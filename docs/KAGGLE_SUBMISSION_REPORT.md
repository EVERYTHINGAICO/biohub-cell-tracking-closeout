# Kaggle Submission Report

Date: 2026-07-01
Competition: `biohub-cell-tracking-during-development`
Final accepted submission: `54247778`
Notebook: `pedroapalaciosz/biohub-unet-dog-pipeline`, version `8`
Public score: `0.113`

## What Worked

The successful path was a Kaggle notebook submission, not a direct local CSV upload. The competition only accepts submissions produced by notebooks, so local files generated from public test data are not valid competition submissions.

The working notebook generated `/kaggle/working/submission.csv` dynamically inside Kaggle using a rule-based DoG detector and the repository tracking/formatting code. It ran without GPU and without internet.

Final kernel output:

```text
Found 4 test volumes
UNet skipped: competition notebook submissions cannot use P100 GPU or internet.
UNet total detections: 0
DoG total detections: 88712
Combined detections: 88712
Total tracks: 88712
Unique tracks: 6657
Submission saved to /kaggle/working/submission.csv
Total rows: 170767
Nodes: 88712
Edges: 82055
Submission shape: (170767, 10)
Submission columns: ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
Row types: {'node': 88712, 'edge': 82055}
```

The downloaded output had `170768` lines including the CSV header.

## Why Earlier Attempts Failed

Direct CSV submission failed because the competition only accepts notebook-generated submissions:

```text
Submission not allowed: This competition only accepts Submissions from Notebooks.
```

Notebook submissions using GPU/internet were rejected by Kaggle competition rules:

```text
Your Notebook cannot use internet access in this competition.
Please disable internet in the Notebook editor and save a new version.
Your Notebook cannot use P100 GPUs in this competition.
```

Running UNet inference in Kaggle with the available P100 failed because the installed PyTorch build did not support that GPU architecture:

```text
CUDA error: no kernel image is available for execution on the device
```

Using `zarr` in the notebook failed because `zarr` was not available in the Kaggle runtime and internet cannot be used during valid competition submissions.

## Required Kaggle Setup

Use a private Kaggle dataset containing the runnable pipeline:

```text
kaggle_artifacts/full_pipeline/
  model_final.pth
  inference.py
  tracker.py
  format_submission.py
  unet3d.py
  data_loader.py
  detection_dog.py
  wheels/
    numcodecs-*.whl
    numpy-*.whl
    typing_extensions-*.whl
```

The dataset metadata must include exactly one license. This worked:

```json
{
  "title": "Biohub Full Pipeline V2",
  "id": "pedroapalaciosz/biohub-full-pipeline-v2",
  "licenses": [
    {
      "name": "CC0-1.0"
    }
  ]
}
```

Upload or update the private dataset with:

```bash
kaggle datasets version -p kaggle_artifacts/full_pipeline -m "Added local numcodecs wheel for offline DoG" -r zip
```

If the dataset does not exist yet:

```bash
kaggle datasets create -p kaggle_artifacts/full_pipeline -r zip
```

## Required Kernel Metadata

The kernel must disable both internet and GPU:

```json
{
  "title": "Biohub UNet + DoG Pipeline",
  "slug": "biohub-unet-dog-pipeline",
  "id": "pedroapalaciosz/biohub-unet-dog-pipeline",
  "code_file": "kernel.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": false,
  "enable_internet": false,
  "dataset_sources": [
    "pedroapalaciosz/biohub-full-pipeline-v2"
  ],
  "competition_sources": [
    "biohub-cell-tracking-during-development"
  ]
}
```

## Kernel Implementation Notes

Do not assume fixed Kaggle input paths. The private dataset may mount under paths like:

```text
/kaggle/input/datasets/pedroapalaciosz/biohub-full-pipeline-v2
```

Find the pipeline root dynamically by searching for `model_final.pth`.

Find test Zarr volumes dynamically:

```python
preferred_test_dir = Path("/kaggle/input/biohub-cell-tracking-during-development/test")
if preferred_test_dir.exists():
    zarr_paths = sorted(p for p in preferred_test_dir.iterdir() if p.is_dir() and p.suffix == ".zarr")
else:
    zarr_paths = sorted(
        p for p in Path("/kaggle/input").rglob("*.zarr")
        if p.is_dir() and "test" in p.parts and pipeline_root not in p.parents
    )
```

Install only local wheels from the private dataset. Do not use internet:

```python
subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "--no-index",
    "--no-deps",
    str(numcodecs_wheel),
])
```

For the accepted submission, UNet was skipped in the Kaggle kernel because GPU was disallowed and the notebook had to run offline. The working method was DoG plus tracking and formatting.

## Submit Commands

Push the kernel:

```bash
kaggle kernels push -p kaggle_artifacts/notebook_kernel
```

Check status:

```bash
kaggle kernels status pedroapalaciosz/biohub-unet-dog-pipeline
```

Download output and logs:

```bash
kaggle kernels output pedroapalaciosz/biohub-unet-dog-pipeline -p /tmp/kaggle_output_v8
```

Submit the notebook version:

```bash
kaggle competitions submit \
  -c biohub-cell-tracking-during-development \
  -f submission.csv \
  -k pedroapalaciosz/biohub-unet-dog-pipeline \
  -v 8 \
  -m "UNet + DoG ensemble"
```

Check submissions:

```bash
kaggle competitions submissions -c biohub-cell-tracking-during-development | head -8
```

The successful submission showed:

```text
ref: 54247778
status: SubmissionStatus.COMPLETE
public_score: '0.113'
error: ''
```

## Interpreting Kaggle Status

`SubmissionStatus.PENDING` means Kaggle accepted the submission and the scorer has not finished yet.

Empty fields mean no value has been produced yet:

```text
public_score: ''
private_score: ''
error_description: ''
```

If scoring fails, the submission usually becomes `COMPLETE` with an empty score and a non-empty `error_description`.

If scoring succeeds, the submission becomes `COMPLETE` with a numeric `public_score`.

## Do Not Repeat These Mistakes

Do not submit local CSV files directly for this competition.

Do not enable internet in the notebook.

Do not enable GPU unless Kaggle rules/runtime change.

Do not depend on `zarr` being installed in Kaggle.

Do not assume `/kaggle/input/<dataset-name>` paths are stable.

Do not copy a public-test CSV into the notebook output and expect it to score on hidden data.

Do not include API tokens or credentials in repo files, datasets, notebooks, logs, or reports.

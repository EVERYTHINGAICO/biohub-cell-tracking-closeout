# Method Record

This document describes the scientific and engineering method captured in this archive. It is written for future reviewers who need to understand what the workflow did, why it was credible, and where its limits are.

## Problem Formulation

The competition asks participants to infer cell lineage graphs from 3D microscopy time series. A valid output is a table of:

- **nodes:** detected cell centers with time and 3D coordinates;
- **edges:** directed parent-child links between nodes;
- **division events:** encoded as one parent with two outgoing child edges.

The core modeling problem is a constrained directed graph reconstruction task:

1. detect cells in each frame;
2. link detections across time;
3. preserve biological/topological constraints;
4. format the graph as Kaggle's required CSV.

## Data Geometry

The local metric and analysis scripts use the physical voxel scale:

```text
z = 1.625 um
y = 0.40625 um
x = 0.40625 um
```

Distances are therefore evaluated in microns, not raw voxel units. This matters because the volumes are anisotropic; a raw Euclidean distance in voxel space would overweight or underweight motion depending on axis.

## Metric Reconstruction

The local metric proxy in `src/metric_local_v2.py` implements the key lessons from the public/competition behavior:

- match predicted and ground-truth nodes per frame using a physical distance threshold;
- map predicted edges through matched nodes;
- ignore predicted edges touching unannotated cells in sparse ground truth;
- compute edge Jaccard and division Jaccard;
- apply an over-prediction adjustment using `estimated_number_of_nodes` when available in GEFF metadata.

The important scientific caveat is that this is a **proxy**. It is useful for directional experiments, but the Kaggle hidden test remains the authoritative score.

## Experiment Families

### 1. Rule-Based Detection and Linking

Early work used DoG-style detections with Hungarian/motion linking and submission formatting. This established the baseline and proved the notebook-submission mechanics.

Observed public-score path:

```text
0.826 -> 0.842
```

### 2. Learned Graph Public Notebook Adaptation

Public temporal-UNet / graph-linking notebooks improved the score materially. We adapted public work, tuned threshold and short-track post-processing, and validated that notebook submissions were the correct path for this competition.

Observed public-score path:

```text
0.856 -> 0.867 -> 0.873
```

### 3. Advanced Post-Processing

Boristown/Yusuke-style public notebooks added topology-preserving post-processing:

- T+2-aware division ranking;
- line-fit smoothing;
- safe division recovery;
- short-track filtering;
- gap and relink controls.

Observed public-score path:

```text
0.889 -> 0.893 -> 0.896
```

### 4. September DeepCenter / ILP Public Adaptation

Near closeout, newer public notebooks appeared with stronger DeepCenter-style detection, public temporal-UNet support packs, DivNet-style division support, and faster ILP/post-processing. We adapted the strongest visible candidate into a private Kaggle kernel.

Final candidate:

```text
Kernel: pedroapalaciosz/biohub-0-951-deepcenter-fast-ilp
Version: 1
Submission ref: 56537358
Rows: 245195
Nodes: 124772
Edges: 120423
Status at last documentation: PENDING
```

## Negative Results

The following paths were investigated and did not become the final route:

- a from-scratch 3D UNet detector underperformed the established detection stack;
- appearance-only pair classifiers did not generalize enough to beat geometric/graph constraints;
- pseudo-label distillation mostly learned to imitate its teacher rather than discover independent errors;
- self-supervised patch features did not produce reliable cell-association signal;
- local post-processing sweeps were useful, but small proxy gains did not always transfer to Kaggle.

Preserving these failures is part of the scientific value of the archive: they prevent repeated work and document the decision boundary.

## Validity Threats

- **Sparse labels:** train labels are not dense full-cell annotations. Some local false positives are invisible to the metric.
- **Public leaderboard feedback:** public notebook adaptation can overfit public leaderboard dynamics.
- **External public assets:** final candidate quality depends on public Kaggle notebooks/datasets. Attribution and license review are required before public GitHub release.
- **Notebook-only submission rule:** direct local CSVs are invalid for this competition; all serious submissions must be notebook generated.

## Reproducibility Standard

For each serious submission, preserve:

- Kaggle kernel slug and version;
- competition submission ref;
- public score when complete;
- notebook metadata;
- input dataset sources;
- exact command used to submit;
- output summary: rows, nodes, edges, datasets;
- constraints: GPU, internet, dependency packaging.

This standard is implemented in `docs/COMPETITION_CLOSEOUT_PLAN.md`.

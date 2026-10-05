# Biohub Cell Tracking During Development

**Everything AI Co research archive for 3D cell tracking, bioimage analysis, and reproducible Kaggle competition workflows.**

This private repository documents our closeout work for Kaggle's `biohub-cell-tracking-during-development` competition: a notebook-only cell lineage tracking challenge using 3D microscopy time series, sparse lineage labels, graph reconstruction, and strict offline inference constraints.

The goal of this archive is not just to store code. It is to preserve a scientifically auditable trail: what we tried, what failed, what improved public score, what came from public Kaggle work, and what should or should not be published after the competition.

## Snapshot

| Field | Value |
| --- | --- |
| Organization | Everything AI Co |
| Domain | Bioimage analysis, 3D cell tracking, scientific ML, Kaggle reproducibility |
| Competition | `biohub-cell-tracking-during-development` |
| Deadline | `2026-09-29 23:59:00` |
| Best completed July submission | `0.896` public score |
| Final official submission | ref `56537358`, `COMPLETE` |
| Final official kernel | `pedroapalaciosz/biohub-0-951-deepcenter-fast-ilp`, version `1` |
| Final official method family | DeepCenter detection + temporal UNet/linking + ILP + graph post-processing |
| Final official scores | public `0.94431`, private `0.91439` |
| Public leaderboard row | rank `1645 / 3950` teams, team `PEDRO A. PALACIOS Z.` |
| Repo visibility | Private until final source-license/publication review |

## Why This Matters

Cell tracking during development is a graph problem hidden inside image data. A useful solution must:

- detect cell centers in anisotropic 3D volumes;
- associate detections across time into directed lineage graphs;
- support cell division events;
- avoid invalid graph topology such as multiple parents;
- run inside Kaggle's notebook execution constraints;
- produce an auditable `submission.csv` generated inside Kaggle.

This repo captures the engineering and scientific workflow behind that process: metric reconstruction, sparse-label validation, model adaptation, post-processing experiments, and final competition submission hygiene.

## Method Summary

The final candidate submission uses a public high-scoring Kaggle notebook family rather than a direct local CSV upload. The adapted kernel is configured as a no-internet Kaggle notebook with T4 GPU execution.

At a high level:

1. **Detection:** DeepCenter-style 3D center-prior detector candidates identify likely cell centers.
2. **Temporal modeling:** a temporal UNet / learned graph association model scores candidate links across adjacent frames.
3. **Optimization:** ILP-style graph selection enforces a coherent cell lineage topology.
4. **Post-processing:** gap handling, division safeguards, short-track filtering, motion relinking, and graph integrity checks improve score while preserving valid structure.
5. **Submission formatting:** the notebook writes Kaggle's required node/edge table with columns:
   `id`, `dataset`, `row_type`, `node_id`, `t`, `z`, `y`, `x`, `source_id`, `target_id`.

See [docs/METHODS.md](docs/METHODS.md) for the scientific method record.

## Scientific Rigor

This archive is intentionally conservative about claims.

- Scores are reported as Kaggle public scores only when the Kaggle CLI reported them.
- Public notebook title scores are treated as external claims until our own submission is scored.
- The local metric work is documented as a proxy, not as a substitute for hidden-test scoring.
- Sparse ground truth behavior is called out explicitly: many unannotated cells are ignored by the metric rather than counted as false positives.
- Failed paths are preserved because they explain why the final solution focused on public model adaptation and graph post-processing.

Key negative results:

- naive 3D UNet detection underperformed the established detection stack;
- appearance-only linkers did not learn enough association signal from sparse labels;
- self-supervised appearance features did not provide robust edge recovery;
- public-post-processing tuning reached a plateau until stronger public DeepCenter-style notebooks appeared.

## Competition Timeline

| Stage | Result |
| --- | --- |
| Early DoG baseline | `0.826` public score |
| Rule-based V3/two-pass linker | `0.842` |
| Public learned-graph adaptation | `0.856` |
| Threshold/short-track tuning | `0.867` to `0.873` |
| Boristown-style post-processing | `0.889` |
| Yusuke LB897 family | `0.896` |
| September DeepCenter/ILP public adaptation | ref `56537358`, public `0.94431`, private `0.91439` |

Operational details live in [docs/COMPETITION_CLOSEOUT_PLAN.md](docs/COMPETITION_CLOSEOUT_PLAN.md).

## Reproducibility

This repository does not include Kaggle data, model weights, output CSVs, private artifacts, or raw chat logs. Reproduction must use Kaggle-attached competition data and public Kaggle datasets/notebooks referenced in the docs.

Check submissions:

```bash
kaggle competitions submissions -c biohub-cell-tracking-during-development | head -8
```

Check the final kernel:

```bash
kaggle kernels status pedroapalaciosz/biohub-0-951-deepcenter-fast-ilp
```

Submit the final notebook version if needed:

```bash
kaggle competitions submit \
  -c biohub-cell-tracking-during-development \
  -f submission.csv \
  -k pedroapalaciosz/biohub-0-951-deepcenter-fast-ilp \
  -v 1 \
  -m "Public DeepCenter fast ILP 0.951 adaptation"
```

## Repository Map

```text
docs/
  METHODS.md                    scientific method record and validation notes
  ATTRIBUTION.md                source notebook/dataset attribution
  COMPETITION_CLOSEOUT_PLAN.md  operational closeout state
  GITHUB_PUBLISHING_AUDIT.md    what can and cannot be published
src/
  detection, tracking, formatting, metric, and experiment scripts
scripts/
  small utility scripts
kaggle_final/
  metadata for the final adapted Kaggle kernel
```

## SEO Keywords

Biohub cell tracking, Kaggle bioimage analysis, 3D cell tracking, cell lineage reconstruction, microscopy time series, DeepCenter, temporal UNet, ILP tracking, graph tracking, scientific machine learning, reproducible ML, AI governance for research workflows, Everything AI Co.

## Publication Policy

This repository is private during active competition closeout. Before public release:

- do not include Kaggle credentials;
- do not redistribute competition data;
- do not include generated submissions, outputs, model weights, or private datasets;
- attribute all public Kaggle notebooks and datasets;
- make clear which results are our submissions and which are public-notebook claims.

See [docs/GITHUB_PUBLISHING_AUDIT.md](docs/GITHUB_PUBLISHING_AUDIT.md).

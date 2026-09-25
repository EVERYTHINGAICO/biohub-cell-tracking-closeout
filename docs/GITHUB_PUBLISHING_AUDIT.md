# GitHub Publishing Audit

Date checked: 2026-09-25
Competition: `biohub-cell-tracking-during-development`

## Bottom Line

Publishing a cleaned repository is feasible, but do not publish this workspace as-is.

Recommended timing:

- Before the competition deadline: keep GitHub private unless the same competition code is also shared publicly on Kaggle for all competitors.
- After the deadline: publish a cleaned repository with code, documentation, and links to Kaggle resources, but without competition data, generated submissions, large outputs, model weights, credentials, or raw chats.

## Rule Summary

From the Biohub competition rules page and Kaggle terms:

- Competition data should not be redistributed outside Kaggle.
- During the competition period, if competition code is shared publicly, it should be shared through Kaggle competition notebooks/discussions for all competitors.
- Public user submissions on Kaggle/GitHub should respect the source licenses and attributions of notebooks, datasets, and models reused.

Practical interpretation for this repo:

- Code we wrote can be published after cleanup.
- Public Kaggle notebook adaptations can be referenced and attributed; redistributing full copied notebooks should be checked against notebook license/terms.
- Data, outputs, generated CSVs, private Kaggle datasets, model weights, and raw transcripts should stay out of GitHub.

## Safe To Publish

- `README.md`
- `.gitignore`
- `docs/*.md`, except raw/private chat contents
- `src/*.py` reusable local code and experiments
- `scripts/*.py` if they do not embed data or secrets
- Small metadata files that describe reproducibility, for example selected `kernel-metadata.json`
- Original small analysis scripts:
  - `analyze_edges.py`
  - `visualize_detections.py`

## Publish With Caution

- `kaggle_haideptry951/biohub-0-951-sota-deepcenter-fast-ilp-19m.ipynb`
  - It is copied/adapted from a public Kaggle notebook. Prefer linking to the original Kaggle notebook and documenting our kernel ref unless license/attribution is clear.
- Other copied public notebooks:
  - `kaggle_lb893/`
  - `kaggle_lb897/`
  - `kaggle_min6/`
  - `kaggle_scorepush/`
  - `kaggle_swap/`
  - `boristown_notebook/`
  - `pilkwang_*`
  - `variant_*`
  - Prefer links + notes instead of committing full notebooks.
- `docs/agent_chat_history/memory/*.md`
  - Useful project memory, but contains account/environment details. Sanitize first.
- `AGENTS.md`
  - Useful internal instructions, but competition-specific operational notes may mention account/kernel refs. OK if reviewed.

## Do Not Publish

- Credentials and local env:
  - `.env`
  - `~/.kaggle/*`
  - any `kaggle.json`, `access_token`, API key, token, password, or credential
- Competition data:
  - `data/`
  - `test/`
  - any `.zarr` or `.geff` data copied from Kaggle
- Generated predictions/submissions/tracks:
  - `submission*.csv`
  - `tracks*.csv`
  - `detections*.csv`
  - `divisions*.csv`
  - `predictions*.csv`
  - `tracks_per_volume/`
- Model weights and training outputs:
  - `*.pth`
  - `*.pt`
  - `models/`
  - `kaggle_*/output*/`
- Raw/private transcripts:
  - `docs/agent_chat_history/raw/`
  - `context.html`
  - `prompt*.md`
  - `problemas.html`
  - `artifacts/`
- Prompt injection / irrelevant local file:
  - `agent.md`

## Current Local Risk Findings

Large files found locally:

- `detections_train.csv` ~413 MB
- `detections_train_filtered.csv` ~389 MB
- `submission_train.csv` ~441 MB
- `tracks_train.csv` ~404 MB
- `tracks_full.csv` ~48 MB

Secret scan notes:

- No obvious literal Kaggle key was found in the scanned non-notebook files after excluding raw chat/data/output folders.
- `.env` exists and must remain ignored.
- `docs/agent_chat_history/memory/biohub-kaggle-status.md` mentions that a Kaggle token exists in `.env`; do not publish unsanitized memory if avoiding operational details.

## Recommended Public Repo Shape

```text
.
├── README.md
├── .gitignore
├── docs/
│   ├── COMPETITION_CLOSEOUT_PLAN.md
│   ├── GITHUB_PUBLISHING_AUDIT.md
│   ├── KAGGLE_SUBMISSION_REPORT.md
│   └── ...
├── src/
│   └── reusable experiment code
├── scripts/
│   └── small utility scripts
└── kaggle_final/
    ├── kernel-metadata.json
    └── README.md with links to the public/source Kaggle notebooks
```

## Pre-Publish Commands

Run from the cleaned repo root:

```bash
git status --short
find . -type f -size +20M -print
rg -n -i "KAGGLE_KEY|kaggle\\.json|api_key|access_token|password|credential|secret|BEGIN .*PRIVATE|ghp_|sk-" .
```

The expected result before public release is:

- no large files except intentionally documented small notebooks if kept;
- no credentials;
- no raw competition data;
- no generated submissions or model weights.

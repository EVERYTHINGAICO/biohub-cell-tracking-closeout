# Biohub Cell Tracking Kaggle Closeout

This repository records our work for the Kaggle competition
`biohub-cell-tracking-during-development`.

## Status

- Competition deadline checked through Kaggle CLI: `2026-09-29 23:59:00`.
- Account used: `pedroapalaciosz`.
- Best July submission in this workspace: `0.896` public score.
- Current closeout submission:
  - Kernel: `pedroapalaciosz/biohub-0-951-deepcenter-fast-ilp`
  - Version: `1`
  - Submission ref: `56537358`
  - Status when recorded: `PENDING`
  - Method family: public DeepCenter + temporal UNet + ILP/post-processing notebook adaptation.

See [docs/COMPETITION_CLOSEOUT_PLAN.md](docs/COMPETITION_CLOSEOUT_PLAN.md) for the current operational state.

## Important Notes

- This is a Kaggle code competition. Valid submissions must be made from a Kaggle notebook version, not by directly uploading a local CSV.
- Final notebooks should run with internet disabled and T4 GPU selected when required.
- Do not commit Kaggle credentials, raw competition data, downloaded private outputs, model weights, or large generated CSV files.
- Raw agent chat histories are private project artifacts and are excluded from the public repo.

## Key Project History

Early local/rule-based work progressed from DoG tracking baselines around `0.826` to public learned-graph/post-processing adaptations around `0.896`. Near closeout, stronger public notebooks appeared, including public DeepCenter/ILP variants in the `0.947` to `0.951` range. We adapted the strongest visible public candidate into our Kaggle account and submitted it as ref `56537358`.

## Reproduction Pointers

Check competition submissions:

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

## Repository Hygiene

Before publishing to GitHub:

```bash
rg -n -i "KAGGLE_KEY|kaggle.json|api_key|token|password|credential|secret" .
find . -type f -size +50M -print
```

Review any findings manually. Most large generated artifacts are intentionally ignored by `.gitignore`.

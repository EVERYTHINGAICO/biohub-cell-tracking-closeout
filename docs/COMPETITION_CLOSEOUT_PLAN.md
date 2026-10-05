# Biohub Kaggle Closeout Plan

Date checked: 2026-10-05
Competition: `biohub-cell-tracking-during-development`

## Current Competition State

- Kaggle CLI access works for account `pedroapalaciosz`.
- Account has entered the competition.
- Deadline reported by Kaggle CLI: `2026-09-29 23:59:00`.
- Competition is closed.
- Final official submission ref: `56537358`.
- Final official public score: `0.94431`.
- Final official private score: `0.91439`.
- Downloaded public leaderboard rank: `1645 / 3950`.

## Our Submission History

Best completed submissions currently visible through Kaggle CLI:

| Ref | Date | Description | Public score | Private score |
| --- | --- | --- | --- | --- |
| `56537358` | 2026-09-25 | Public DeepCenter fast ILP 0.951 adaptation | `0.94431` | `0.91439` |
| `54447806` | 2026-07-08 | swap-ontop: yusuke 0.897 pipeline + 6 target swaps | `0.896` |
| `54443420` | 2026-07-07 | min7 = yusuke LB897 reference | `0.896` |
| `54443419` | 2026-07-07 | min6 short-track A/B | `0.896` |
| `54408599` | 2026-07-06 | score_push preset A/B | `0.893` |
| `54408588` | 2026-07-06 | safe_div_precision | `0.893` |
| `54360325` | 2026-07-05 | boristown T+2 division + linefit + safediv | `0.889` |
| `54310562` | 2026-07-03 | pilkwang learned-graph + gap recovery | `0.856` |
| `54276855` | 2026-07-02 | rule-based V3 with two-pass linker | `0.842` |
| `54252425` | 2026-07-02 | rule-based DoG public baseline | `0.826` |
| `54247778` | 2026-07-02 | early UNet + DoG ensemble | `0.113` |

## Public Notebook Recon

Public notebook listing now shows stronger runnable notebooks than the July LB897 family:

| Public kernel | Claimed/title score | Notes |
| --- | --- | --- |
| `haideptry/biohub-0-951-sota-deepcenter-fast-ilp-19m` | `0.951` title | Public, GPU T4, no internet, uses public datasets including DeepCenter and DivNet. Copied to our account as `pedroapalaciosz/biohub-0-951-deepcenter-fast-ilp`. |
| `zhincez/biohub-0-947-lb-runnable-with-public-datasets` | `0.947` title | Public, GPU T4, no internet, public datasets. Good fallback if 0.951 fails. |
| `beraterolelk/0-947-lb-biohub-deepcenter-ilp-tracker` | `0.947` title | Public, GPU T4, no internet, public datasets. Similar DeepCenter/ILP family. |
| `hengck23/end2end-cell-linker-raw-edge-ja-0-9-no-ilp` | raw edge JA 0.9 | Public but has internet enabled in metadata; useful for ideas, less ideal for final valid submission unless made offline-safe. |

## Active Work

- Local adapted notebook folder: `kaggle_haideptry951/`
- Kaggle kernel pushed: `pedroapalaciosz/biohub-0-951-deepcenter-fast-ilp`
- Version pushed: `1`
- Kernel status: `COMPLETE`
- Downloaded output: `kaggle_haideptry951/output_v1/`
- Submission CSV shape: `245195` rows x `10` columns
- Submission row types: `124772` nodes, `120423` edges
- Competition submission ref: `56537358`
- Final submission status: `COMPLETE`
- Final public/private scores: `0.94431` / `0.91439`
- Submission time: `2026-09-25 02:05:09.477000`

Submitted command used:

```bash
kaggle competitions submit \
  -c biohub-cell-tracking-during-development \
  -f submission.csv \
  -k pedroapalaciosz/biohub-0-951-deepcenter-fast-ilp \
  -v 1 \
  -m "Public DeepCenter fast ILP 0.951 adaptation"
```

Check final score:

```bash
kaggle competitions submissions -c biohub-cell-tracking-during-development | head -8
```

Late exploratory kernels from `2026-09-27` completed and produced structurally valid `submission.csv` files, but they were not submitted before the competition deadline and therefore are not official results.

## Final Competition Checklist

- Official high-scoring notebook submission completed: `56537358`.
- Submission came from a Kaggle notebook version, not a direct local CSV.
- Logs, output summary, metadata, and kernel references were preserved locally.
- Kaggle credentials, raw competition data, private dataset contents, and large generated CSV/model outputs remain out of GitHub.

## GitHub Repo Cleanup Plan

Recommended public repo contents:

- `README.md`: narrative, final score, method summary, setup notes.
- `docs/`: reports, closeout plan, metric notes, traceability.
- `src/`: local experiments and reusable utilities.
- `kaggle_haideptry951/`: final adapted notebook metadata and notebook, if license allows.
- `public_notebooks/`: optional, only if licenses permit redistribution; otherwise replace with links and notes.
- `.gitignore`: exclude `.env`, Kaggle tokens, `data/`, large CSVs, model weights, notebook outputs, raw chat transcripts.

Before publishing:

```bash
rg -n -i "KAGGLE_KEY|kaggle.json|api_key|token|password|credential|secret" .
find . -type f -size +50M -print
```

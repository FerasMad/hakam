# Preprocessing - review folder

Everything in this folder is **aggregate statistics only** and safe to share. It is
the reviewable evidence that data preprocessing is complete.

Nothing here contains video, frames, match identifiers or local paths. Those live in
`artifacts/preprocessing/private/`, which is gitignored and must not be shared - the
SoccerNet NDA permits research and educational use but forbids redistribution.

## Start here

| File | What it shows |
|---|---|
| `REPORT.md` | The full written report - source, validation, label quality, limitations |
| `class_distribution.png` | Cascade targets per split, and the annotator uncertainty held out of training |
| `split_summary.csv` | Observed counts against official counts |
| `cascade_supervision_counts.csv` | Usable training samples at each cascade stage |
| `label_distribution.csv` | Raw label distributions per split |
| `dataset_summary.json` | Machine-readable run record: settings, git commit, timing |

## Headline

| Split | Actions | Official | Clips | Missing | Conflicts |
|---|---|---|---|---|---|
| train | 2,916 | 2,916 | 6,621 | 0 | 0 |
| valid | 411 | 411 | 970 | 0 | 0 |
| test | 301 | 301 | 706 | 0 | 0 |

All 8,297 clips were decoded, not merely checked for existence. Zero duplicates, and no
exact video crosses from training into validation or test.

## Two decisions worth asking about

**Ambiguity is handled per stage, not per row.** 675 actions carry labels the annotator
marked uncertain. Dropping those whole rows is the obvious move and it silently discards
usable supervision - an `Offence + severity 4.0` still tells you a foul occurred and a
card was due, and only the colour is unresolved. Supervision is withheld one stage at a
time, and the uncertain actions are kept as a named evaluation slice.

**Stage 3 was cut on evidence.** Training holds 686 yellow cards against 27 red. That
cannot support a defensible classifier, so the vision model stops at card / no-card and
the language model discusses severity from the retrieved Law text instead of asserting a
trained colour prediction.

## Reproducing it

```bash
python -m src.data.preprocess --mode full --splits train valid test
python -m pytest tests -q
```

The pipeline records its settings, the git commit and the elapsed time into
`dataset_summary.json`, and exits non-zero if any quality gate fails. 66 automated tests
cover label normalisation, conflict handling, manifest construction and frame sampling.

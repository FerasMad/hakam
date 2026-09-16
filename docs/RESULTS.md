# Results

All scores are **balanced accuracy** (the average of per-class recall, so 0.50 is
chance for a yes/no decision). 95% intervals resample whole matches, because
incidents from the same match are correlated.

Splits are the official SoccerNet-MVFoul ones, separated by match:
2,916 train / 411 validation / 301 test actions.

## Final model

VideoMAE-base (Kinetics-400), last 3 of 12 blocks fine-tuned, one head per task,
every camera view used as a training sample and views averaged per incident at
test time. Retrained on train + validation with the recipe and stopping epoch fixed
in advance; decision thresholds chosen on validation.

| Task | Test | What it means |
|---|---|---|
| Card (no card / card) | **0.64** | The headline decision |
| Offence (no offence / offence) | **0.64** | Only ~25 no-offence test incidents — treat with caution |
| Body part (upper / lower) | **0.68** | Used in the explanation |
| Action family (4 classes) | **0.54** | Chance is 0.25; used in the explanation only when confident |

For context: the VARS paper reports 0.34 balanced accuracy on its 4-class severity
task, and in its human study qualified referees agreed on severity only weakly
(Cohen's κ 0.21). The labels are hard even for people.

## How we got there (card, validation)

| Step | Change | Card BA |
|---|---|---|
| Baseline | Frozen VideoMAE-small, linear head | 0.506 |
| Exp 1–2 | Fine-tune small; centre crop vs full frame | 0.550 / 0.552 (memorised training set) |
| Exp 3 | Fixed augmentation, regularisation | 0.583 |
| Exp 4 | Crop centred on the players in contact | 0.547 — no gain |
| Exp 5–6 | Longer training, denser frame window | ≤ 0.55 — overfits |
| Exp 7 | VideoMAE-base | 0.578 (last-3-epoch mean) |
| Round 4 | Every camera view + four heads + weight averaging, 3 seeds | 0.654 |
| Overnight | 60- and 80-epoch runs, flip test-time augmentation, per-task ensemble | 0.672 |

Findings:
- **Capacity and data moved the number; epochs did not.** Every run peaked by epoch
  3–5. A 60-epoch run matched, but did not beat, the 5-epoch model (0.652 vs 0.654).
- **Using all camera views** was the largest single gain (+5 points).
- **Longer training helped the attribute heads** (action +9, body part +3 on validation).
- **Adding the validation set to training** added 3–4 points on card and offence on test.

## Test results across rounds

| Task | Round 4 ensemble | Overnight ensemble | Final model (train+valid) |
|---|---|---|---|
| Card | 0.606 | 0.593 | 0.642 |
| Offence | 0.610 | 0.617 | 0.640 |
| Body part | 0.678 | 0.683 | 0.682 |
| Action | 0.344 (8 classes) | 0.475 (4 families) | 0.540 (4 families) |

The test set was scored three times — once per round. Every model and threshold was
chosen on validation, never on test, but these are not single-look estimates.

## Action classes → Law 12 families

The annotators' 8 action classes became 4 families after round 4:

| Family | Includes | Why |
|---|---|---|
| tackle | standing tackling, tackling, challenge | challenge recall was 0.02 — visually the same duel |
| hands | holding, pushing | pushing recall was 0.00 with 190 clips |
| elbowing | elbowing | |
| high leg | high leg | |

Dive (73 clips, recall 0.00) has no family; those incidents still count for card,
offence and body part. The action is left out of the explanation when its confidence
is below 60%.

## Language layer

See [`llm/REPORT.md`](llm/REPORT.md): 47 curated bilingual Law 12 / Law 5 chunks,
hybrid retrieval, prompt v2 with claim checking. On 24 test contracts v2 reached
1.00 label faithfulness with 0% unsupported claims (v1: 0.71 and 67%).

## Limitations

- The model decides from a 2.6-second window at 224×224; small contact details are
  often below the resolution the backbone sees.
- Referee strictness varies by match and league (card rate per match ranges 0–100%),
  and test matches are unseen referees — part of the label is not in the video.
- Faithfulness is checked against label vocabulary, not every paraphrase.
- A faithful explanation of a wrong vision decision is still wrong; the two are
  reported separately, and low-confidence cases are referred to a human.

# Preprocessing report - Hakam

Generated 2026-09-06T13:13:03Z from commit `0142d08`, pipeline v1.0, in 409.4s.

Aggregate statistics only. Row-level manifests stay under `artifacts/preprocessing/private/` and are never committed or shared - they carry match identifiers and local paths.

## 1. Source and access

SoccerNet-MVFoul, obtained through the official SoccerNet downloader under a signed NDA with KAUST. Research and educational use only; the dataset is not redistributed. The download password is held in a secret store, never in the repository.

## 2. Splits - observed against official

| split   |   actions |   actions_official |   clips |   clips_valid |   multiview_actions |   ambiguous |   conflicts |   excluded |
|:--------|----------:|-------------------:|--------:|--------------:|--------------------:|------------:|------------:|-----------:|
| train   |      2916 |               2916 |    6621 |          6621 |                2916 |         541 |           0 |          1 |
| valid   |       411 |                411 |     970 |           970 |                 411 |          85 |           0 |          0 |
| test    |       301 |                301 |     706 |           706 |                 301 |          49 |           0 |          0 |

Observed counts match the official counts exactly for all three splits. The shipped splits are used as-is; nothing is re-partitioned.

## 3. Media validation

- Clips referenced: **8,297**
- Failing validation: **0**
- Exact duplicate clip rows: **0**
- Cross-split duplicate (leakage): **none**

Every clip was decoded, not merely stat-ed. File metadata can claim a frame count the container will not actually yield, and a clip that fails only at decode time would otherwise surface as a crash during training.

## 4. Label quality

| status         |   actions |
|:---------------|----------:|
| valid          |      2913 |
| ambiguous      |       675 |
| usable_partial |        39 |
| excluded       |         1 |

- Unknown primary label values: **0**
- Cross-field conflicts (for example no-offence carrying a card): **0**
- Annotator-uncertain actions: **675**

## 5. Ambiguity policy - stage-wise drop

`Between`, severity `2.0` and severity `4.0` are deliberate uncertainty markers written by a professional referee, not corruption. Discarding a whole incident because one later stage is unresolved throws away usable supervision, so supervision is withheld per stage instead:

| Raw annotation | Stage 1 offence | Stage 2 card | Stage 3 colour |
|---|---|---|---|
| offence + severity 2.0 | trains | withheld | withheld |
| offence + severity 4.0 | trains | trains | withheld |
| between + any severity | withheld | withheld | withheld |

These actions are retained as a named evaluation slice for the error analysis, where they test whether model confidence falls on the cases the annotator also found hard.

## 6. Cascade supervision available

| split   |   actions |   stage 1 offence |   stage 2 card |   stage 3 colour |
|:--------|----------:|------------------:|---------------:|-----------------:|
| train   |      2916 |              2819 |           2060 |              713 |
| valid   |       411 |               396 |            299 |              107 |
| test    |       301 |               286 |            230 |               73 |

**Finding that changed the design.** Training carries 686 yellow cards against **27 red**. That cannot support a defensible classifier, so stage 3 is cut from the vision model. Severity is instead discussed by the language model from the retrieved Law text and the auxiliary attributes, with no asserted colour prediction.

Stage 1 is roughly 9:1 toward offence, so a model answering offence every time scores 50% balanced accuracy while learning nothing. Stage 2, at about 2:1, is the only reasonably balanced decision and carries the product claim.

## 7. Configuration

- Sampling window: frames **43-107** (2.56s at 25fps), **16** frames
- Ambiguity policy: `stagewise-drop`
- Seed: `42`, mode: `full`

The window was measured, not inherited. The published baseline's 63-87 spans 0.96s, so 16 frames drawn from it are near-duplicates and the backbone sees an almost still image. 43-107 reproduces the 16-frames-at-stride-4 sampling the backbone was pretrained with, and won on every diagnostic measurement.

Augmentation is applied at training time only and never materialised to disk. Validation and test preprocessing is deterministic: the same clip and config produce the same tensor on every run.

## 8. Sample visualisation

`private/sample_grid.png` shows the centre of the sampling window across camera types. It contains frames from restricted video and is deliberately excluded from this shareable folder.

## 9. Exclusions

Actions excluded: **1**. Clips excluded: **0**. Every exclusion is written to `private/exclusions.csv` with a machine-readable reason code, a timestamp and the pipeline version. The raw download is never modified; corrections are derived values with provenance, not edits.

## 10. Known limitations

- **Red cards are too rare to model.** 27 in training. Stage 3 is cut rather than reported as a weak result.
- **Stage 1 is severely imbalanced** (about 9:1). Balanced accuracy near 50% there reflects the prior, not the model.
- **Replay speed varies between views of one incident**, so views are not interchangeable and any reasoning about impact velocity must account for it.
- **Rare camera types form a long tail** (spider cam, goal-line technology, inside the goal), each with a handful of clips.
- **Ambiguous labels are held out, not solved.** They bound how well any model can do here, and are the honest explanation for published accuracy sitting near 50%.

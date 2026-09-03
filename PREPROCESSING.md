# Hakam Data Preprocessing Specification

## 1. Purpose

This document defines the required preprocessing pipeline for Hakam's
SoccerNet-MVFoul data. It is an implementation specification: the pipeline is
not considered complete until it can inspect the real dataset, produce the
documented artifacts, and pass the acceptance checks in this document.

The preprocessing pipeline must:

1. preserve the official SoccerNet splits;
2. inventory and validate every action and video clip;
3. normalize labels without overwriting the source annotations;
4. detect missing, ambiguous, contradictory, and unknown labels;
5. derive internally consistent cascade targets;
6. detect corrupt media, exact duplicates, and cross-split leakage;
7. sample model-ready frames reproducibly;
8. report class balance and representative samples; and
9. maintain an auditable record of every exclusion and transformation.

Preprocessing ends with validated manifests and a tested frame sampler. Frozen
backbone inference and feature caching are the first steps of model development,
although they may be run immediately after this pipeline.

## 2. Definition of done

Preprocessing is complete only when all of the following are true:

- The data source, access procedure, licence restrictions, and official split
  sizes are documented.
- `Train`, `Valid`, and `Test` are loaded from their official directories without
  re-splitting.
- Every annotation record has a unique composite key: `(split, action_id)`.
- Every referenced clip has been checked for existence and basic decodability.
- Every action and clip receives an explicit quality status.
- Raw label values are retained, and canonical labels are stored in separate
  columns.
- Cross-field conflicts such as `No offence + Red card` are detected and never
  silently converted into valid training labels.
- Ambiguous labels are handled per cascade stage, not by blindly deleting the
  entire row.
- Missing clips, corrupt clips, exact duplicates, label conflicts, unknown
  values, and exclusions are written to reports.
- Exact duplicate videos do not cross from the training set into validation or
  test data.
- Class distributions and the amount of usable supervision at each cascade
  stage are reported for every split.
- A deterministic frame sampler returns 16 RGB frames from frames 63-87 at
  `224 x 224` for every valid clip.
- Training augmentation is separated from deterministic validation/test
  preprocessing.
- Automated tests cover label normalization, conflict handling, manifest
  generation, and frame sampling.
- One documented command can reproduce all preprocessing outputs.

## 3. Data governance and storage rules

SoccerNet-MVFoul is access-controlled and must not be redistributed. Treat the
downloaded dataset and all row-level derivatives as private data.

### 3.1 Never commit

Do not commit any of the following:

- videos, archives, extracted frames, thumbnails, or sample grids derived from
  restricted videos;
- original or copied `annotations.json` files;
- row-level manifests containing annotations, match names, or local paths;
- hashes that could unnecessarily fingerprint the restricted files;
- model checkpoints when they exceed the repository's intended artifact policy.

Keep these under ignored directories such as:

```text
data/
features_cache/
artifacts/preprocessing/private/
```

Only aggregate statistics and documentation should be considered for the public
repository, and only after checking the data agreement.

### 3.2 Raw data is immutable

Never edit annotations or video files in place. The pipeline must create a
normalized manifest and an exclusion manifest while leaving the raw download
unchanged. A correction is a new derived value plus a reason code, not a change
to the source file.

### 3.3 Process one split at a time

Use the standard-definition download first. To control disk use:

```text
extract one split -> validate -> create manifests -> extract features later
```

Do not export every sampled frame as an image. Decode the required frames on
demand or pass them directly to the frozen backbone.

## 4. Expected source layout

```text
data/
├── Train/
│   ├── annotations.json
│   ├── action_0/
│   │   ├── clip_0.mp4
│   │   └── clip_1.mp4
│   └── ...
├── Valid/
├── Test/
└── Chall/
```

Expected official action counts:

| Split | Directory | Actions | Labels expected |
|---|---|---:|---|
| Train | `Train` | 2,916 | Yes |
| Validation | `Valid` | 411 | Yes |
| Test | `Test` | 301 | Yes |
| Challenge | `Chall` | 273 | No |
| Total | - | 3,901 | 3,628 labelled actions |

The challenge split is optional during preprocessing because it has no public
targets. It must never be treated as labelled test data.

## 5. Output data contracts

Create two private, row-level manifests. CSV is sufficient and avoids requiring
an additional Parquet engine.

### 5.1 Action manifest

One row per incident:

| Column | Type | Description |
|---|---|---|
| `split` | string | `train`, `valid`, `test`, or `challenge` |
| `action_id` | string | Raw action identifier; unique only within a split |
| `action_key` | string | Stable composite key, for example `train:104` |
| `source_match` | string | Raw `UrlLocal`, retained privately for provenance |
| `num_clips_raw` | integer | Number of clips listed in annotations |
| `num_clips_valid` | integer | Number remaining after media validation |
| `offence_raw` | string/null | Original annotation value |
| `severity_raw` | string/null | Original annotation value |
| `action_class_raw` | string/null | Original annotation value |
| `offence` | string/null | Canonical offence label |
| `severity` | string/null | Canonical severity label |
| `target_offence` | string/null | `offence` or `no_offence` |
| `target_card` | string/null | `card` or `no_card` |
| `target_card_colour` | string/null | `yellow` or `red` |
| `supervise_offence` | boolean | Whether the row may train stage 1 |
| `supervise_card` | boolean | Whether the row may train stage 2 |
| `supervise_card_colour` | boolean | Whether the row may train stage 3 |
| `ambiguous` | boolean | Raw annotation expresses genuine uncertainty |
| `quality_status` | string | Status defined in section 7 |
| `quality_codes` | string | Pipe-separated reason codes |
| `include_multiview` | boolean | At least two valid, non-duplicate views |

Keep all remaining raw auxiliary fields in explicitly named `*_raw` columns.

### 5.2 Clip manifest

One row per video view:

| Column | Type | Description |
|---|---|---|
| `action_key` | string | Foreign key to the action manifest |
| `clip_index` | integer | Position in the raw `Clips` list |
| `annotated_url` | string/null | Original `Clips[].Url` |
| `resolved_path` | string | Validated local path |
| `camera_type_raw` | string/null | Raw camera description |
| `timestamp_raw` | number/null | Raw match timestamp |
| `replay_speed_raw` | number/null | Raw replay speed |
| `file_exists` | boolean | File is present |
| `file_size_bytes` | integer/null | File size |
| `decodable` | boolean | PyAV can open and decode required frames |
| `frame_count` | integer/null | Decoded or trusted stream frame count |
| `fps` | number/null | Average frame rate |
| `width` / `height` | integer/null | Source dimensions |
| `duration_seconds` | number/null | Derived duration |
| `sha256` | string/null | Computed only where duplicate checking requires it |
| `quality_status` | string | Clip quality status |
| `quality_codes` | string | Pipe-separated reason codes |
| `include` | boolean | Clip may be used by downstream processing |

## 6. Normalization rules

Always store both the raw and canonical value. Normalization may remove
irrelevant formatting differences; it must not invent a missing label or resolve
a semantic disagreement.

### 6.1 Offence

| Raw value | Canonical value |
|---|---|
| `Offence` | `offence` |
| `No offence`, `No Offence` | `no_offence` |
| `Between` | `between` |
| empty/null | null |
| any other value | null plus `UNKNOWN_OFFENCE` |

Trim surrounding whitespace before matching. Do not use unrestricted fuzzy
matching. Any newly observed spelling must first appear in the unknown-value
report and then be added to a reviewed mapping table.

### 6.2 Severity

Normalize numeric or string representations to the following canonical values:

| Raw value | Meaning |
|---|---|
| `1`, `1.0`, `"1.0"` | `no_card` |
| `2`, `2.0`, `"2.0"` | `borderline_no_yellow` |
| `3`, `3.0`, `"3.0"` | `yellow` |
| `4`, `4.0`, `"4.0"` | `borderline_yellow_red` |
| `5`, `5.0`, `"5.0"` | `red` |
| empty/null | null |
| any other value | null plus `UNKNOWN_SEVERITY` |

Do not round arbitrary numbers into these classes.

### 6.3 Auxiliary labels

Normalize only values listed in the verified vocabulary for:

- action class;
- body part;
- upper-body part;
- contact;
- multiple fouls;
- try to play;
- touch ball;
- handball; and
- handball offence.

An invalid auxiliary label must normally disable only that auxiliary target. It
must not discard an otherwise valid primary cascade target.

## 7. Record status and reason codes

Every action and clip must receive exactly one primary status:

| Status | Meaning | Default action |
|---|---|---|
| `valid` | All required values are consistent | Include |
| `usable_partial` | Some cascade or auxiliary targets are unavailable | Include only valid targets |
| `ambiguous` | Annotation explicitly expresses uncertainty | Exclude uncertain stages; retain for analysis |
| `conflict` | Two fields make incompatible claims | Quarantine from supervised training |
| `invalid_media` | Required media is missing or undecodable | Exclude affected clip/action |
| `excluded` | Fails a hard rule or approved manual exclusion | Exclude |

Reason codes must be machine-readable constants. At minimum support:

```text
MISSING_OFFENCE
MISSING_SEVERITY
UNKNOWN_OFFENCE
UNKNOWN_SEVERITY
AMBIGUOUS_OFFENCE
CONDITIONAL_SANCTION_WITH_AMBIGUOUS_OFFENCE
AMBIGUOUS_CARD_DECISION
AMBIGUOUS_CARD_COLOUR
OFFENCE_SEVERITY_CONFLICT
MISSING_CLIP
CORRUPT_CLIP
INSUFFICIENT_FRAMES
INVALID_FPS
INVALID_RESOLUTION
INVALID_REPLAY_SPEED
FEWER_THAN_TWO_VALID_VIEWS
DUPLICATE_WITHIN_ACTION
DUPLICATE_ACROSS_ACTIONS
CROSS_SPLIT_DUPLICATE
AUXILIARY_LABEL_CONFLICT
```

## 8. Offence/severity conflict matrix

Apply this matrix before deriving cascade targets. `null` means that the target
must not supervise that stage.

| Offence | Severity | Status | Stage 1 | Stage 2 | Stage 3 | Required handling |
|---|---|---|---|---|---|---|
| No offence | empty or 1.0 | valid | no offence | null | null | Valid terminal branch |
| No offence | 2.0 | conflict | null | null | null | Quarantine; sanction contradicts terminal branch |
| No offence | 3.0 | conflict | null | null | null | Quarantine: `No offence + Yellow card` |
| No offence | 4.0 | conflict | null | null | null | Quarantine: `No offence + borderline Yellow/Red` |
| No offence | 5.0 | conflict | null | null | null | Quarantine: `No offence + Red card` |
| Offence | 1.0 | valid | offence | no card | null | Valid |
| Offence | 2.0 | ambiguous | offence | null | null | Stage 1 is usable; card/no-card is uncertain |
| Offence | 3.0 | valid | offence | card | yellow | Valid |
| Offence | 4.0 | ambiguous | offence | card | null | Stages 1 and 2 are usable; colour is uncertain |
| Offence | 5.0 | valid | offence | card | red | Valid |
| Offence | empty | usable_partial | offence | null | null | Flag missing severity; use stage 1 only |
| Between | any recognized severity | ambiguous | null | null | null | Preserve for analysis; no cascade stage is safe to supervise |
| empty | any value | excluded | null | null | null | Exclude because the cascade root is missing |
| unknown | any value | excluded | null | null | null | Exclude and report schema drift |

### 8.1 Important default: do not silently salvage conflicts

For a true contradiction such as `No offence + Red card`, the default is to set
all three supervision flags to false and write the record to the conflict report.
The pipeline does not know whether the offence label or the severity label is
correct.

If a qualified reviewer later adjudicates the record, store the reviewed value,
reviewer, date, and rationale in a separate overrides file. Never edit the raw
annotation. If the red card represents misconduct outside the challenge being
classified, that is a different domain category and requires an explicit schema
extension; it must not be forced into the current foul cascade.

### 8.2 Ambiguity is not the same as corruption

`Between`, severity `2.0`, and severity `4.0` are intentional uncertainty labels.
Keep them for limitations and error analysis. Under the default stage-wise policy:

- `Offence + 2.0` can train offence detection, but not card or colour;
- `Offence + 4.0` can train offence detection and card detection, but not colour;
- `Between` cannot train any downstream cascade stage because stage 1 is unknown.

When `Between` is accompanied by a card-like severity, add
`CONDITIONAL_SANCTION_WITH_AMBIGUOUS_OFFENCE` to the review report. Do not call
it a proven contradiction without domain adjudication: it can mean "if this is
an offence, this would be the sanction." It is still unusable by the current
cascade because its root decision is unresolved.

This corrects a common data-loss error: discarding an entire incident when one
later stage is ambiguous.

## 9. Additional cross-field validation

Implement conservative rules. A plausible combination must not be labelled a
conflict merely because it is uncommon.

### 9.1 Safe rules

- `Bodypart = Upper body` with an empty `Upper body part`: mark the auxiliary
  detail missing; do not discard the primary targets.
- `Bodypart = Under body` with a populated `Upper body part`: flag an auxiliary
  conflict and disable the upper-body-part target.
- An empty `Try to play` or `Touch ball` value disables that auxiliary target;
  do not impute `No`.
- `Without contact` is not automatically invalid. Diving and attempted dangerous
  challenges can legitimately have no contact.
- Empty handball-specific fields may be structurally valid for non-handball
  actions. Do not fill them with `No` without verifying the source semantics.
- A replay speed that is empty may be reported as missing. A non-numeric or
  non-positive value is invalid.

### 9.2 Rules to avoid

Do not infer labels from filenames, camera type, match identity, or another
auxiliary label. Do not use model predictions to "repair" ground truth.

## 10. Schema and annotation validation

For each labelled split:

1. Confirm `annotations.json` exists and is valid JSON.
2. Confirm the root contains `Actions` and an optional declared action count.
3. Compare the declared count, parsed count, and expected official count.
4. Require a dictionary-like action collection.
5. Use `(split, action_id)` as the unique identifier; action IDs may repeat in
   different splits.
6. List missing required keys and all previously unseen values.
7. Compare annotated clip URLs with resolved filesystem paths.
8. Do not assume clip indices are contiguous. Prefer the annotated `Clips[].Url`
   after safely normalizing it to a local `.mp4` path; fall back to a rebuilt
   path only when the annotation URL is absent.
9. Record both the annotated and resolved path so mismatches remain auditable.
10. Fail the run on malformed top-level schema; quarantine individual malformed
    records when the rest of the file can be processed safely.

## 11. Video quality validation

Use PyAV, which is already in the project requirements.

### 11.1 Fast pass

For every clip, record:

- existence and file size;
- whether a video stream is present;
- container/codec readability;
- reported frames, FPS, duration, width, and height; and
- whether the clip appears long enough to reach frame 87.

### 11.2 Decode pass

Metadata can be wrong. Attempt to decode the exact frames required by the model.
A valid baseline clip must provide all selected frame indices. Do not silently
pad a short or corrupt clip by repeating its last frame.

Use zero-based indices generated deterministically from the inclusive window:

```python
indices = np.linspace(63, 87, num=16).round().astype(int)
```

Document and test the inclusive/end-index convention against the upstream
baseline before training.

### 11.3 Multi-view rule

An action qualifies for multi-view training only when at least two distinct,
valid clips remain after media and duplicate checks. An action with one valid
view may be retained for a separately labelled single-view diagnostic, but must
not enter the main multi-view experiment unnoticed.

## 12. Duplicate and leakage detection

Exact hashing of every large file is expensive. Use a two-stage process:

1. group clips by file size;
2. compute a streaming SHA-256 only for files in repeated-size groups.

Handle duplicates as follows:

| Duplicate type | Handling |
|---|---|
| Same bytes within one action | Keep one view; flag `DUPLICATE_WITHIN_ACTION` |
| Same bytes across actions in one split | Quarantine and investigate annotation duplication |
| Train duplicate in validation or test | Hard failure: `CROSS_SPLIT_DUPLICATE` |
| Similar-looking but non-identical replay | Do not auto-delete |

Do not use perceptual similarity to remove legitimate multi-view replays
automatically. If near-duplicate detection is explored, use it only to create a
manual-review list.

Also report repeated `UrlLocal` match identifiers across splits. Because the
dataset provides official splits, do not reorganize them without a documented,
instructor-approved experimental reason.

## 13. Exclusion and override policy

The raw dataset remains untouched. Produce two private files:

```text
artifacts/preprocessing/private/exclusions.csv
artifacts/preprocessing/private/label_overrides.csv
```

An exclusion record must contain:

```text
action_key, clip_index, reason_code, detail, detected_at, pipeline_version
```

A manual label override must additionally contain:

```text
field, raw_value, reviewed_value, reviewer, reviewed_at, rationale
```

No manual override is valid without provenance. Automated preprocessing should
not create expert labels.

## 14. Split policy

- Use the official `Train`, `Valid`, and `Test` splits exactly as shipped.
- Never fit mappings, thresholds, class weights, or sampling decisions using the
  test split.
- Compute class weights from the training split only.
- Use validation for model selection and confidence-threshold calibration.
- Open the test labels only for final evaluation runs.
- Do not merge validation into training merely to increase sample count unless a
  final retraining protocol is declared after all model choices are frozen.
- Never train against the unlabelled challenge split.

## 15. Frame preprocessing

The deterministic baseline transform must:

1. decode the selected 16 frames from frames 63-87;
2. convert every frame to RGB;
3. resize/crop according to the selected backbone's documented processor;
4. produce `224 x 224` frames;
5. apply the backbone-specific normalization; and
6. output the tensor layout expected by that backbone.

Do not assume MViT and VideoMAE use the same tensor layout or normalization.
Keep backbone-specific transforms behind a single tested interface.

The same input clip and configuration must produce the same validation/test
tensor on repeated runs.

## 16. Data augmentation

Do not materialize augmented videos on disk. Apply augmentation during training
feature extraction, or cache a clearly versioned augmented embedding set.

Recommended first augmentation experiment:

- mild random resized crop;
- mild brightness/contrast/saturation jitter;
- horizontal flip when no direction-dependent target is used; and
- small temporal jitter that remains inside the reviewed foul window.

Apply one consistent spatial transform to all frames in a clip. Independent
random crops per frame create artificial camera motion. Validation and test data
must never receive random augmentation.

Maintain separate cache identities, for example:

```text
mvit_v2_s__baseline__train
mvit_v2_s__mild_aug_v1__train
mvit_v2_s__baseline__valid
mvit_v2_s__baseline__test
```

## 17. Required reports

Generate the following private artifacts on every full run:

```text
artifacts/preprocessing/private/
├── actions_train.csv
├── actions_valid.csv
├── actions_test.csv
├── clips_train.csv
├── clips_valid.csv
├── clips_test.csv
├── exclusions.csv
├── conflicts.csv
├── unknown_labels.csv
├── duplicate_clips.csv
├── missing_or_corrupt_clips.csv
└── sample_grid.png
```

Generate aggregate, potentially shareable artifacts separately:

```text
artifacts/preprocessing/summary/
├── dataset_summary.json
├── split_summary.csv
├── label_distribution.csv
├── cascade_supervision_counts.csv
├── class_distribution.png
└── preprocessing_report.md
```

The report must include:

- source and access method;
- official and observed split counts;
- number of actions, clips, and valid views;
- missing/corrupt media counts;
- exact duplicate and leakage results;
- missing, unknown, ambiguous, and conflicting label counts;
- class distributions by split;
- usable samples per cascade stage;
- preprocessing and augmentation configuration;
- representative local sample visualization;
- exclusions and their reasons; and
- known limitations.

Do not publish any artifact until its compatibility with the data agreement has
been checked.

## 18. Reproducible command interface

Implement a target command such as:

```bash
python -m src.data.preprocess \
  --data-root data \
  --splits train valid test \
  --output-dir artifacts/preprocessing \
  --ambiguity-policy stagewise-drop \
  --start-frame 63 \
  --end-frame 87 \
  --num-frames 16 \
  --seed 42
```

Recommended optional modes:

```bash
# Quick schema and label audit without decoding all videos
python -m src.data.preprocess --data-root data --mode metadata

# Full media validation
python -m src.data.preprocess --data-root data --mode full

# Validate only a small development sample
python -m src.data.preprocess --data-root data --mode full --limit-actions 100
```

The program must log the configuration, start/end time, Git commit, and counts.
It must return a non-zero exit code for hard failures such as malformed schema,
wrong official split counts, or cross-split exact duplicates. Individual bad
records should be reported and excluded without hiding the rest of the audit.

## 19. Automated tests

Add tests for at least the following cases:

### 19.1 Label tests

- both capitalizations of `No offence`;
- numeric and string severity values;
- leading/trailing whitespace;
- unknown offence and severity values;
- every valid row in the conflict matrix;
- `No offence + Red card` is quarantined;
- `Offence + severity 2.0` supervises only offence;
- `Offence + severity 4.0` supervises offence and card, but not colour;
- `Between` does not supervise the cascade under stage-wise drop;
- auxiliary missing values do not become invented negative labels.

### 19.2 Manifest tests

- duplicate action IDs across different splits remain distinct;
- a missing clip is reported;
- annotated non-contiguous clip names resolve correctly;
- one valid view disables multi-view inclusion;
- unknown keys are retained or reported without corrupting known fields;
- an empty split returns a clear error instead of an index exception.

### 19.3 Video tests

- correct 16-frame indices;
- short clip rejection;
- corrupt clip rejection;
- RGB conversion;
- expected shape and dtype;
- deterministic validation/test output; and
- temporally consistent training augmentation.

Run:

```bash
python -m pytest tests/data -q
```

## 20. Quality gates

The full preprocessing run passes only when:

1. observed split counts equal the official counts, or every difference is
   explicitly explained and approved;
2. there are no unknown primary label values;
3. all conflicts are quarantined or manually adjudicated with provenance;
4. all included clips exist and decode the required frames;
5. every included multi-view action has at least two distinct valid clips;
6. no exact video duplicate crosses from train into validation or test;
7. validation and test preprocessing is deterministic;
8. target values obey the cascade invariants below; and
9. all preprocessing tests pass.

Cascade invariants:

```text
target_offence == no_offence
    => target_card is null and target_card_colour is null

target_card == no_card
    => target_offence == offence and target_card_colour is null

target_card_colour in {yellow, red}
    => target_offence == offence and target_card == card
```

Assert these invariants in code before writing the final manifests.

## 21. Recommended implementation order

1. Add reviewed normalization functions and explicit reason codes.
2. Add conflict detection before cascade derivation.
3. Correct stage-wise ambiguity handling, especially severity `4.0`.
4. Update annotation loading to preserve raw values and annotated clip URLs.
5. Build action and clip manifests.
6. Add metadata-only quality validation.
7. Add full PyAV decode validation and frame sampling.
8. Add exact duplicate and cross-split leakage checks.
9. Generate aggregate statistics, plots, and local sample grids.
10. Add tests and a single reproducible CLI.
11. Run first on 100 actions, then one full split at a time.
12. Review all conflicts and exclusions before starting baseline training.

## 22. Current repository gaps

The current repository already provides useful foundations in
`src/data/annotations.py` and `src/data/labels.py`, but it does not yet satisfy
this specification.

Important changes required:

- detect cross-field conflicts before deriving targets;
- do not accept `No offence + Red card` as a normal no-offence row;
- for `Offence + severity 4.0`, retain the certain `card` supervision while
  withholding only the uncertain colour target;
- resolve paths from annotated clip URLs rather than assuming contiguous clip
  indices;
- handle empty action tables safely in summary generation;
- add media, duplicate, leakage, and multi-view validation;
- create persistent manifests and reports; and
- add automated tests.

Completion of these items provides the evidence required to state that Hakam's
data preprocessing phase is finished and ready for baseline model development.

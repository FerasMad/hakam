# حكم (Hakam) — project handoff

Paste this into a fresh session to resume. Updated 14 Sep 2026.
Presentation ~16 Sep 2026. Team: Feras, Anas, Bader.

---

## 1. What this is

Final capstone for the Tuwaiq **AI Model Development Bootcamp**.

**حكم (Hakam) — Arabic VAR Explainer.** A vision model classifies a foul incident
from multi-view match video. Its structured output — and *only* that output — is
handed to a language model, which retrieves the matching article from the Laws of
the Game and writes an Arabic explanation of the decision.

- Repo: <https://github.com/FerasMad/hakam> (**public** — never commit dataset or secrets)
- Brief: `Desktop/Tuwaiq Bootcamp/Final Project/AI_Model_Development_Bootcamp_Final_Capstone.docx.md`

### Hard requirements from the brief

| § | Requirement |
|---|---|
| 18 | CV model must be **trained or fine-tuned** by the team (transfer learning allowed) |
| 7 | Baseline + **at least 2 experiments** with comparison |
| 9 | LLM performs a **core function** — RAG / explanation |
| **10** | **Real integration.** CV output must become structured data the LLM consumes. Two demos bolted together is explicitly marked down. **This section decides the grade.** |
| 18 | Test on unseen data |
| 11 | Runnable prototype |
| 18 | High accuracy **not** required; evaluation and error analysis **are** |
| 17 | Every member must explain the part they built |

---

## 2. Current state (verified 14 Sep 2026)

| Track | State |
|---|---|
| **Preprocessing** | **Complete.** Spec closed, 78 tests passing |
| **CV model** | Feature extraction built; **heads not written** |
| **LLM half** | **Complete against `mock_contract()`.** Corpus, retrieval, OpenAI generation, guardrails, evaluation, and manual audit are implemented |
| **§10 integration** | Contract boundary is implemented and tested; swap the real CV output in for `mock_contract()` |
| **App / report / slides** | LLM report complete; app and slides not assessed here |

**Blocking:** the Colab feature cache was produced but never downloaded from
Drive. `features_cache/` holds only `pilot500__videomae_small.npz`. Nothing can
train until the real caches are local.

---

## 3. Dataset

**SoccerNet-MVFoul.** NDA signed with KAUST. Access credentials are held only in a
Colab secret named `SOCCERNET_PASSWORD`, never in a file.

| Split | Actions | Clips | Labelled |
|---|---|---|---|
| Train | 2,916 | 6,621 | yes |
| Valid | 411 | 970 | yes |
| Test | 301 | 706 | yes |
| Challenge | 273 | — | **no** — unusable |

**Clip format:** 126 frames, 25 fps, 398×224. Incident centred near frame 75.
2–4 camera views per action (2,241 have 2, 561 have 3, 114 have 4).

**Annotation fields (13):** UrlLocal, Offence, Contact, Bodypart, Upper body part,
Action class, Severity, Multiple fouls, Try to play, Touch ball, Handball,
Handball offence, Clips[]. Clip-level: Url, Camera type, Timestamp, Replay speed.

### Dataset gotchas that cost real time

1. **`downloadDataTask` takes the password as an ARGUMENT**, not the `.password`
   attribute (that is what `downloadGames` reads). Setting only the attribute
   gives HTTP 401, and the downloader prints rather than raises — so the failure
   surfaces cells later as a confusing missing-annotations error.
2. **The downloader nests one level deeper** than its `LocalDirectory`:
   `data/mvfouls/mvfouls/train.zip`. Non-recursive globs find nothing.
3. **The zips extract flat.** Every split contains `action_0`, `action_1`… so two
   splits extracted into one directory silently overwrite each other.
4. **Replay speed varies 1.0–10.0 between views of the same incident.** Views are
   not interchangeable; anything reasoning about velocity must account for it.

`scripts/normalise_colab_layout.py` handles 1–3.

---

## 4. Labels and the cascade

**No labelling work is needed.** A professional referee annotated all 3,628
actions. This is video *classification* — whole clip in, decision out. No
bounding boxes; the foul is localised in **time**, not space.

`src/data/labels.py` derives cascade targets:

| Raw `Offence` | Raw `Severity` | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|---|
| Offence | 1.0 | offence | no_card | — |
| Offence | 3.0 | offence | card | yellow |
| Offence | 5.0 | offence | card | red |
| No offence | any | no_offence | — stops | — |
| Offence | 2.0 | offence | **withheld** | **withheld** |
| Offence | 4.0 | offence | card | **withheld** |
| Between | any | **withheld** | **withheld** | **withheld** |

**Ambiguity is dropped per stage, not per row.** `Between`, severity 2.0 and 4.0
are deliberate uncertainty markers from the referee. Discarding a whole incident
because one later stage is unresolved throws away usable supervision — an
`Offence + 4.0` still proves a foul occurred and a card was due.

**Supervision available:**

| Stage | Train | Valid | Test | Class split (train) |
|---|---|---|---|---|
| 1 — offence | 2,819 | 396 | 286 | 2,495 / 324 → **88:12** |
| 2 — card | 2,060 | 299 | 230 | 1,303 / 757 → **63:37** |
| 3 — colour | 713 | 107 | 73 | 686 / **27** → **cut** |

**675 ambiguous actions** (541/85/49) held out as a named evaluation slice.

---

## 5. Preprocessing — complete

```bash
python -m src.data.preprocess --mode full --splits train valid test
```

Verified: 19/19 required artifacts, 21/21 action columns, 19/19 clip columns,
0 cascade-invariant violations, 0 conflicts, 0 duplicates, no cross-split
leakage, 78 tests passing. Full run ~7 minutes.

Outputs split by disclosure:
- `artifacts/preprocessing/summary/` — aggregate, shareable (mirrored to `docs/preprocessing/`)
- `artifacts/preprocessing/private/` — row-level manifests, **gitignored, never share**

---

## 6. CV models

| Key | Checkpoint | Dim | Role |
|---|---|---|---|
| `videomae_small` | `MCG-NJU/videomae-small-finetuned-kinetics` | 384 | Iteration, fine-tuning |
| `videomae_base` | `MCG-NJU/videomae-base-finetuned-kinetics` | 768 | Capacity comparison |
| `mvit_v2_s` | torchvision | 768 | Published-baseline reproduction |

**`mvit_v2_s` is the config default but cannot run locally** — torchvision absent.
Colab only. It has never actually been run.

### The measurement that drives everything

Frozen-backbone probe, 100 actions (`scripts/diagnose_features.py`):

| window | pooling | camera probe | card probe |
|---|---|---|---|
| **43-107** | mean | **0.939** | 0.465 |
| 63-87 | mean | 0.908 | 0.450 |

500-action probe: card **0.556** (p=0.020), offence 0.576 (p=0.020).

**Frozen Kinetics features encode camera framing (0.94), not fouls (0.47).**
Kinetics teaches "playing soccer"; it never teaches contact intensity.
**Fine-tuning is mandatory, not an optimisation** — and §18 wants exactly that.

### Two traps

1. **`transformers` 5.8.x silently zeroes VideoMAE attention biases.** Checkpoints
   store `q_bias`/`v_bias`; transformers reports them UNEXPECTED, reports its own
   `query.bias`/`value.bias` MISSING, and loads anyway. `q_bias` reaches 2.8 while
   weights peak near 0.6 — a large perturbation that surfaces as mysteriously weak
   features. `repair_videomae_biases()` fixes it and **raises if it restores nothing**.
2. **The window was measured, not inherited.** The published 63-87 spans 0.96s, so
   16 frames from it are near-duplicates and the backbone sees a still image.
   **43-107 = 2.56s** reproduces VideoMAE's 16-frames-at-stride-4 pretraining and
   won on every measurement.

### A metric that did not work — do not reinvent it

*View contrast* (same-incident cosine minus random-pair cosine) came out at
**−0.003** and is **confounded**: two views of one incident are different camera
angles, while two random clips often share a camera type. An embedding dominated
by framing fails it by construction. The camera and card probes are the honest
readings.

---

## 7. Pipelines and experiments

```
clip.mp4 (126 frames, 25fps, 398x224)
   |  frames 43-107 = 2.56s, 16 frames sampled
   v
[augmentation - train split only, one transform per clip]
   |
   v
backbone -> mean over tokens -> one vector per CLIP
   |  2-4 views per action
   v
view pooling (mean / max / attention) -> one vector per ACTION
   |
   v
cascade:  stage 1 offence (tuned for recall >=0.97)
          stage 2 card    (the real decision)
   + auxiliary heads: action class, body part, contact, try to play, touch ball
```

**A — frozen + linear heads.** The baseline §7 requires. Expect 55–58%.
**B — fine-tune `videomae_small`.** The largest available gain. GPU only.
**C — fine-tune `videomae_base`.** Capacity question; may overfit and lose to B.

| Brief | Pipeline |
|---|---|
| Baseline | A |
| Experiment 1 | B vs A — frozen against fine-tuned |
| Experiment 2 | recall-oriented objective (class weighting / focal loss) vs standard |
| Ablation | view pooling: mean vs attention |

**Caching is per clip, never per action**, so view pooling stays a training-time
choice. Cache identities carry the augmentation recipe (`train__baseline` vs
`train__mild_aug_v1`) so an augmented set cannot be reused as the clean baseline.

**Augmentation (§16):** one spatial transform per clip — never per frame, which
would invent camera motion. Train split only, enforced in code: `build_params`
returns `None` for any other split. Temporal jitter clamped so the window cannot
slide off the incident.

---

## 8. The contract — the §10 answer

`src/contract.py`. **The LLM never sees a pixel.**

```json
{ "offence":    {"label":"offence", "confidence":0.91},
  "card":       {"label":"card",    "confidence":0.68},
  "attributes": {"action_class":"tackling", "contact":"with_contact"},
  "scene":      {"players_in_contact_zone": 2},
  "num_views":  3,
  "low_confidence_fields": ["card"] }
```

Every generated sentence must trace to a field here. `citable_values()` returns
the permitted set; anything outside it was invented. **Grounding by construction,
not by prompting** — the differentiator from X-VARS.

`mock_contract()` lets the language half be built with no trained model, which is
what makes three people work in parallel.

---

## 9. LLM half — complete against mock contracts

1. `laws/corpus.json` contains 47 curated bilingual chunks from Law 12, Law 5,
   and relevant definitions. Both official PDFs were verified to have text layers.
2. Retrieval combines contract-tag overlap with `intfloat/multilingual-e5-base`;
   passage embeddings are cached locally.
3. `gpt-5-nano` generates the Arabic explanation from contract JSON and retrieved
   articles only. V2 enforces a safe draft, semantic checks, repair, and fallback.
4. Final 24-case results: v1 faithfulness 0.710; v2 1.000, zero unsupported label
   claims, 100% correct abstention, and 100% low-confidence hedging.
5. Ten v2 outputs were manually audited and passed after the final guardrails.

See `docs/llm/REPORT.md`. Run `python scripts/demo_llm.py` for the mock-contract
demo and `python scripts/eval_llm.py` to regenerate the local evaluation CSVs.

---

## 10. Optional: pretrained detector

`src/features/detect.py` (uncommitted). Pretrained YOLOv8n — **no labelling, no
training**. Counts players, finds the contact zone, writes an overlay frame for
the demo.

- Measured: 4 clips in 6s on CPU
- Contact radius **scales with median player height**, not pixels — a fixed 120px
  put 9 of 14 players "in contact" on a wide shot
- **Expect zero accuracy gain.** Player count is weakly related to card/no-card;
  its value is explanation detail and demo credibility
- Degrades to `None` if ultralytics is absent — never a core dependency

**For spatial "where":** the right answer is an **attention heatmap** from the
fine-tuned model (attention rollout / Grad-CAM). Zero labels, derived from the
model that made the decision, and it doubles as a diagnostic — if attention lands
on the crowd, the model has not learned fouls. Frame it honestly as *where the
model attended*, not *where the foul is*.

---

## 11. Metrics and estimates

**Report two numbers for every stage, always naming the stage:** oracle-parent
(flattering) and end-to-end (honest).

**Recall compounds:** end-to-end card recall = P(stage 1 says offence | card) ×
P(stage 2 says card | card). At 0.90 × 0.80 that is **0.72**, not 0.80. Hence
stage 1 is tuned for near-total recall, not balance.

| Configuration | Balanced accuracy |
|---|---|
| Stage 1, offence | 60–68% |
| Stage 2, frozen | 55–58% |
| **Stage 2, fine-tuned** | **62–70%, centre ~65%** |
| End-to-end | 59–64% |
| **At 60% coverage, answered cases** | **72–80%** |

**End-to-end explanation:** decision 0.65 × retrieval 0.90 × faithful 0.92 ≈
**54%**; at 60% coverage ≈ **63%**. Faithfulness alone: **90–95%**, high by
construction.

**Never quote a bare accuracy.** Stage 1 is 9:1, so 88% is achievable by a model
that learned nothing.

### Measured costs

| Operation | CPU | T4 |
|---|---|---|
| Decode one clip | 45 ms | — |
| Frozen forward, `videomae_small` | 1,197 ms | ~160 ms |
| Full corpus (8,297 clips) | 2.8 h | 30–60 min |
| YOLO detection | 1.5 s/clip | — |
| Feature cache size | 25 MB | — |
| Preprocessing full run | 434 s | — |

Extraction on a T4 is **decode-bound**, not GPU-bound (5 s per batch of 32 on
Colab's 2 vCPUs). More GPU will not help; more CPU workers would.

---

## 12. Related work — disclose before a judge finds it

- **VARS** — <https://arxiv.org/abs/2304.04617> — CVPR 2023W. **~50% balanced
  accuracy is state of the art on this dataset.** Uses attention view pooling.
- **X-VARS** — <https://arxiv.org/abs/2404.06332> — end-to-end multimodal LLM,
  English, trained on SoccerNet-XFoul (22k referee-written QA pairs).
  **Our differentiator: the LLM never sees video, so grounding is structural.**
- **Challenge baseline** — 39.5% foul type / 34.5% offence+severity.
- Reference implementations: <https://github.com/csjihwanh/soccernet-MLV>
  (2nd place, CVPR 2024), <https://github.com/druefena/MVFoul>

**Open questions:** does X-VARS retrieve over the Laws, and in what language? Is
SoccerNet-XFoul accessible as a reference for explanation quality?

---

## 13. Decisions and why

1. **Cascade of binary heads**, not one multi-class model — mirrors referee
   reasoning, and each stage has a 50% floor instead of 12.5%.
2. **Stage 3 cut.** 27 red cards in training cannot support a classifier.
3. **Ambiguity dropped per stage**, held out as an evaluation slice.
4. **Frozen features cached per clip** — but the probe proved they are
   insufficient, so fine-tuning is the real path.
5. **The LLM never sees video.** The §10 answer.
6. **Screening tool, not decision-maker.** The costly error is a *miss*, so recall
   on the card class outranks precision. Three-way confidence band: auto-clear /
   refer to human / flag.
7. **Coverage-conditioned accuracy is the headline**, not flat accuracy.
8. **NDA:** video never leaves the ephemeral Colab runtime; only embeddings reach
   Drive.

---

## 14. Commands

```bash
python -m src.data.preprocess --mode full --splits train valid test
python -m pytest tests -q
python scripts/diagnose_features.py --n 100
python scripts/extract_all.py --backbones videomae_small --splits train valid test
python scripts/extract_all.py --backbones videomae_small --splits train --augment mild_aug_v1
python scripts/normalise_colab_layout.py
```

Colab: <https://colab.research.google.com/github/FerasMad/hakam/blob/main/notebooks/colab_extract.ipynb>

---

## 15. What is left, in priority order

1. **Download the feature cache from Drive** — blocks everything on the CV side.
2. Cascade heads on cached features → baseline, with the stage named.
3. Fine-tune (Experiment 1) — the only change likely to move the number.
4. Evaluation harness: PR curves, threshold at target recall, review load.
5. Feed the real CV `HakamContract` into the completed LLM layer on the **test
   split — touched once, in that step only**.
6. Streamlit app, full report (§16), 10–12 slides, impact card, backup demo video.

**Biggest remaining risk:** the real CV model and end-to-end test integration.

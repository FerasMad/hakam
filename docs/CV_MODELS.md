# CV models and training pipelines

What the vision half of Hakam uses, why, and the three pipelines worth running.
Every number here was measured on this dataset, not quoted from a paper.

---

## 1. The measurement that drives the design

A frozen Kinetics backbone was probed on 100 actions before committing to a
training plan (`scripts/diagnose_features.py`, 2 Sep 2026):

| window | pooling | camera probe | card probe |
|---|---|---|---|
| 43-107 | mean | **0.939** | 0.465 |
| 43-107 | fc_norm | 0.939 | 0.465 |
| 63-87 | mean | 0.908 | 0.450 |
| 63-87 | fc_norm | 0.908 | 0.450 |

Read those two columns together. The embedding separates **camera framing** almost
perfectly (0.94 - close-up against main camera) and separates **card from no-card**
at chance (0.47). A larger 500-action probe put the card signal at 0.556 with
p=0.020: real, but far too weak to build on.

**Conclusion: frozen Kinetics features encode the sport, not the foul.**
Kinetics-400 teaches a model to recognise "playing soccer". It never teaches contact
intensity, which is the entire decision here. Fine-tuning is therefore mandatory,
not an optimisation.

That is convenient for the brief: section 18 requires a model the team trained or
fine-tuned, and this is evidence for the choice rather than an assertion.

### A metric that did not work, kept here as a warning

The diagnostic also computed *view contrast* - mean cosine between two views of one
incident, minus mean cosine between random clips - expecting it to be positive. It
came out at -0.003, and the metric is confounded: two views of one incident are
*different camera angles*, while two random clips often share a camera type. An
embedding dominated by framing fails this test by construction. Do not quote it.
The camera and card probes are the honest readings.

---

## 2. Backbones

| Key | Checkpoint | Dim | Role |
|---|---|---|---|
| `videomae_small` | `MCG-NJU/videomae-small-finetuned-kinetics` | 384 | Iteration, pilots, fine-tuning |
| `videomae_base` | `MCG-NJU/videomae-base-finetuned-kinetics` | 768 | Capacity comparison |
| `mvit_v2_s` | torchvision | 768 | Published-baseline reproduction |

**`mvit_v2_s` is the config default but cannot run locally** - torchvision is not
installed. It is Colab-only. Pass `--backbones videomae_small` explicitly on a
laptop.

### One trap worth knowing about

`transformers` 5.8.x loads these VideoMAE checkpoints with their attention biases
**silently zeroed**. The upstream files store `q_bias`/`v_bias`; transformers reports
them UNEXPECTED, reports its own `query.bias`/`value.bias` MISSING, and loads anyway.
In the small checkpoint `q_bias` reaches 2.8 while attention weights peak near 0.6,
so this is a large perturbation that surfaces as mysteriously weak features rather
than as an error.

`repair_videomae_biases()` in `src/features/extract.py` restores them and **raises if
it restores nothing**, so a future library change cannot re-introduce the bug quietly.

---

## 3. The pipeline

```
clip.mp4 (126 frames, 25fps, 398x224)
   |
   |  frames 43-107  =  2.56 s centred on the incident
   |  16 frames sampled uniformly  ->  stride 4, matching VideoMAE pretraining
   v
[augmentation]  train split only, one transform per clip
   |
   v
backbone processor  ->  resize / normalise as the checkpoint expects
   |
   v
backbone  ->  mean over spatiotemporal tokens  ->  one vector per CLIP
   |
   |  2-4 clips per action
   v
view pooling  ->  one vector per ACTION
   |
   v
cascade heads
   stage 1: offence / no-offence      (tuned for recall, not balance)
   stage 2: card / no-card            (the real decision)
```

**Why 43-107 and not the published 63-87.** The published window spans 0.96 s, so 16
frames drawn from it are near-duplicates and the backbone effectively sees a still
image. 43-107 reproduces the 16-frames-at-stride-4 sampling VideoMAE was pretrained
with, and won on every measurement above.

**Why per-clip caching.** View pooling stays a training-time choice instead of being
baked into an expensive extraction run. Changing pooling strategy costs seconds, not
hours.

**Why stage 3 is cut.** Training holds 686 yellow cards against 27 red. That cannot
support a defensible classifier. Severity is discussed by the language model from the
retrieved Law text instead of asserted by a trained head.

---

## 4. Three pipelines

### A - Frozen backbone + linear heads (the baseline)

Extract embeddings once, train light heads on top. Experiments become seconds instead
of hours.

- **Expect:** stage 2 balanced accuracy near 0.55-0.60. Weak, and that is the point.
- **Role:** the baseline section 7 requires, and the control that makes fine-tuning's
  gain measurable.
- **Cost:** extraction ~30-60 min on a T4 for all 8,297 clips, then seconds per head.

### B - Fine-tune `videomae_small` (the one that should work)

Unfreeze the backbone and train end-to-end on the cascade targets.

- **Expect:** the largest single gain available. Frozen features are the bottleneck,
  so removing that bottleneck is the highest-value experiment.
- **Watch:** 2,916 training actions is small for an 87M-parameter model. Use the
  augmentation recipe, freeze the early blocks, keep the learning rate low
  (1e-5 to 5e-5), and stop on validation.
- **Cost:** GPU-only. Budget an hour or two per run on a T4.

### C - Fine-tune `videomae_base` (capacity comparison)

Same as B with the larger backbone.

- **Role:** answers "would more capacity help?" with a number rather than a guess.
- **Risk:** more parameters against the same 2,916 actions. May overfit and lose to B.
  A negative result here is still worth reporting.

### Mapping to the brief

| Brief requirement | Pipeline |
|---|---|
| Baseline | A |
| Experiment 1 | B vs A - frozen against fine-tuned |
| Experiment 2 | recall-oriented objective (class weighting / focal loss) vs standard |
| Ablation | view pooling: mean against attention |

Change one variable per comparison. C is a bonus if time allows.

---

## 5. View pooling

Each action carries 2-4 views: 2,241 actions have 2, 561 have 3, 114 have 4.

- **Mean** - the baseline. Simple, no parameters, surprisingly hard to beat.
- **Max** - lets one decisive angle dominate. Plausible for fouls, where usually one
  view actually shows the contact.
- **Attention** - learns per-view weights. What the published baseline uses, and where
  VARS reports its gains. Costs parameters the dataset may not support.

Replay speed varies from 1.0 to 10.0 *between views of the same incident*, so views
are genuinely not interchangeable. Anything reasoning about impact velocity must
account for it.

---

## 6. Metrics

**Report two numbers for every stage**, always naming the stage:

- **Oracle-parent** - stage 2 evaluated on ground-truth offences. Flattering.
- **End-to-end** - stage 2 inherits stage 1's errors. The honest one.

**Recall compounds through the cascade.** End-to-end card recall equals
P(stage 1 says offence | card) x P(stage 2 says card | card). At 0.90 x 0.80 that is
**0.72**, not 0.80. Stage 1 silently caps everything downstream, which is why it is
tuned for near-total recall on "offence" (cheap - 88% of actions are offences) rather
than for balance.

**Headline metric: end-to-end card recall, always reported with its review load.**
Recall alone is trivially faked by predicting "card" every time.

Never quote a bare accuracy. Stage 1 is 9:1, so "88% accurate" is achievable by a
model that has learned nothing.

---

## 7. Measured costs

| Operation | Cost |
|---|---|
| Decode one clip | ~45 ms (CPU, single core) |
| Frozen forward, `videomae_small` | ~1,200 ms/clip CPU, ~160 ms/clip T4 |
| Full extraction, 8,297 clips | ~2.8 h CPU, ~30-60 min T4 |
| Feature cache on disk | ~25 MB - embeddings, not frames |

Extraction on a T4 is **decode-bound**, not GPU-bound (5 s per batch of 32 on Colab's
2 vCPUs). More GPU will not speed it up; more CPU workers would.

---

## 8. Open risks

- **Fine-tuning may overfit.** 2,916 actions is small. Augmentation, layer freezing
  and early stopping are the mitigations; a negative result is still reportable.
- **Stage 1 barely matters once tuned for recall.** Say so plainly rather than
  presenting it as a strong classifier - it is a filter, by design.
- **Published SOTA is ~50% balanced accuracy** on the multi-class task (VARS,
  CVPR 2023W). Any claim far above that invites doubt. Disclose the prior art before
  a judge finds it.
- **`mvit_v2_s` has never been run.** It is the config default and is Colab-only.
  Either run it or change the default before the demo.

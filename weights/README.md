# Model weights

The vision model is not in git (330 MB). Put these two files here:

| File | Where it comes from |
|---|---|
| `final.pt` | Drive `hakam_colab/runs/refit_v3/final.pt` — the final model (trained on train + validation) |
| `thresholds.json` | see below |

`thresholds.json` holds the decision thresholds chosen on validation. The values
were verified against Drive `hakam_colab/runs/final_v3/metrics.json` on 16 September
2026:

```json
{
  "model_version": "hakam-refit-v3",
  "thresholds": {"card": 0.50, "offence": 0.60, "body_part": 0.49}
}
```

Do not add an `action_class` threshold: that head is multiclass. The separate
0.60 product gate controls whether an offence decision is explained or referred
for human review.

Check it works:

```bash
python -m src.inference.predict data/mvfouls/Test/action_0/clip_0.mp4 data/mvfouls/Test/action_0/clip_1.mp4
```

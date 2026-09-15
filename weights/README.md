# Model weights

The vision model is not in git (330 MB). Put these two files here:

| File | Where it comes from |
|---|---|
| `final.pt` | Drive `hakam_colab/runs/refit_v3/final.pt` — the final model (trained on train + validation) |
| `thresholds.json` | see below |

`thresholds.json` holds the decision thresholds chosen on validation. Copy the
`thresholds` block from Drive `hakam_colab/runs/final_v3/metrics.json`:

```json
{
  "model_version": "hakam-refit-v3",
  "thresholds": {"card": 0.48, "offence": 0.52, "body_part": 0.53}
}
```

(Use the exact values from `metrics.json`; the numbers above are an example.)

Check it works:

```bash
python -m src.inference.predict data/mvfouls/Test/action_0/clip_0.mp4 data/mvfouls/Test/action_0/clip_1.mp4
```

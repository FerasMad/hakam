# Model weights

The local web backend requires both of these production artifacts in this
directory (the 330 MB checkpoint is normally kept out of git):

| File | Where it comes from |
|---|---|
| `final.pt` | Drive `hakam_colab/runs/refit_v3/final.pt` — the final model (trained on train + validation) |
| `thresholds.json` | validation-derived production thresholds |

The supplied `thresholds.json` is:

```json
{"offence": 0.6, "card": 0.5, "body_part": 0.49}
```

Do not retune these values or add an `action_class` threshold. The action head
is multiclass, while the separate product abstention gate remains 0.60 on the
confidence of the selected offence label.

Check it works:

```bash
python -m src.inference.predict data/mvfouls/Test/action_0/clip_0.mp4 data/mvfouls/Test/action_0/clip_1.mp4
```

The deployed checkpoint contains `offence`, `card`, `action_class`, and
`body_part` heads. It does not contain a card-colour head.

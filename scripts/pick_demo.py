"""Pick ten demo videos from a full-test predictions.csv, strictly.

    python scripts/pick_demo.py artifacts/full_test/predictions.csv

Website mode (last clip). Eight "clean" picks where the model agrees with the referee on
everything it shows - offence, card, body part, and the action family whenever the action is
displayed (confidence >= 0.60) - spread over outcomes and action families; plus one
refer-to-human case and one confident honest mistake.
"""

from __future__ import annotations

import sys

import pandas as pd

M = "last"
SLOTS = [  # (reason, filter)
    ("card · tackle", lambda d: (d.true_card == "card") & (d.fam == "tackle")),
    ("card · elbowing", lambda d: (d.true_card == "card") & (d.fam == "elbowing")),
    ("card · high leg", lambda d: (d.true_card == "card") & (d.fam == "high leg")),
    ("card · hands", lambda d: (d.true_card == "card") & (d.fam == "hands")),
    ("red-card incident", lambda d: d.raw_severity.astype(str).isin(["5.0", "4.0"]) & (d.true_card == "card")),
    ("no card · tackle", lambda d: (d.true_card == "no_card") & (d.fam == "tackle")),
    ("no card · hands", lambda d: (d.true_card == "no_card") & (d.fam == "hands")),
    ("no card · other", lambda d: d.true_card == "no_card"),
    ("no offence", lambda d: d.true_offence == "no_offence"),
    ("card · any", lambda d: d.true_card == "card"),
]


def main() -> int:
    df = pd.read_csv(sys.argv[1], dtype={"action_id": str}).fillna("")
    d = df.copy()
    d["answered"] = ~d[f"{M}_abstain"].astype(str).str.lower().eq("true")
    d["fam"] = d[f"{M}_pred_action_class"]
    d["shows_action"] = d[f"{M}_conf_action_class"].astype(float) >= 0.60
    ok = (
        d["answered"]
        & (d.true_offence == d[f"{M}_pred_offence"])
        & ((d.true_card == "") | (d.true_card == d[f"{M}_pred_card"]) | (d.true_offence == "no_offence"))
        & ((d.true_body_part == "") | (d.true_body_part == d[f"{M}_pred_body_part"]))
        & (~d.shows_action | (d.true_action_class == "") | (d.true_action_class == d.fam))
    )
    d["strength"] = d[[f"{M}_conf_offence", f"{M}_conf_card", f"{M}_conf_body_part"]].astype(float).min(axis=1)
    clean = d[ok].sort_values("strength", ascending=False)

    picks, used, clean_slots = [], set(), 0
    for reason, rule in SLOTS:
        if clean_slots == 8:
            break
        cand = clean[rule(clean) & ~clean.action_id.isin(used)]
        if len(cand):
            r = cand.iloc[0]
            picks.append((r, reason))
            used.add(r.action_id)
            clean_slots += 1

    refer = d[~d["answered"] & ~d.action_id.isin(used)].sort_values(f"{M}_conf_offence")
    if len(refer):
        picks.append((refer.iloc[0], "refer to human (low confidence)"))
        used.add(refer.iloc[0].action_id)
    wrong = d[d["answered"] & (d.true_card != "") & (d.true_card != d[f"{M}_pred_card"])
              & (d.true_offence == d[f"{M}_pred_offence"]) & ~d.action_id.isin(used)]
    wrong = wrong.sort_values(f"{M}_conf_card", ascending=False)
    if len(wrong):
        picks.append((wrong.iloc[0], "honest mistake (card wrong, confident)"))

    print("| # | Action | Shows | Clip | Referee | Model (last clip) |")
    print("|---|---|---|---|---|---|")
    for i, (r, reason) in enumerate(picks, 1):
        action = r.fam if r.shows_action else "(action hidden, <60%)"
        if r.answered:
            card = (f"{r[f'{M}_pred_card']} {float(r[f'{M}_conf_card']):.0%} · "
                    if r[f"{M}_pred_offence"] == "offence" else "")
            model = (f"{r[f'{M}_pred_offence']} {float(r[f'{M}_conf_offence']):.0%} · {card}"
                     f"{action} · {r[f'{M}_pred_body_part']} {float(r[f'{M}_conf_body_part']):.0%}")
        else:
            model = f"refer to human (offence confidence {float(r[f'{M}_conf_offence']):.0%})"
        referee = (f"{r.true_offence or '?'} · {r.true_card or '-'} · {r.raw_action} · "
                   f"severity {r.raw_severity} · {r.true_body_part}")
        print(f"| {i} | {r.action_id} | {reason} | `{r.last_clip}` | {referee} | {model} |")
    print(f"\n{len(clean)} of {len(d)} incidents are fully correct in website mode.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

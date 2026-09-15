# Overnight v5 run — observations log

Notebook: `notebooks/hakam_train_v5_overnight.ipynb` on hakamtuw@gmail.com, A100 High-RAM.
Baseline to beat (round 4 ensemble, test): card 0.606 · offence 0.610 · body part 0.678 · action (8-class) 0.344.

## 15 Sep 2026

- **Earlier attempt died overnight.** Runtime disconnected (sign-in/session lost); no
  `overnight_report.json` or `final_v3` reached Drive. Lesson: one Google account per
  Chrome profile, laptop plugged in, sleep off.
- **Sign-in loop** ("Your Google Account was signed out on a different tab") came from
  several Google accounts in one browser. Fixed by a separate Chrome profile with only
  hakamtuw signed in. Do not open Drive/Colab on another account in that window while
  the run is going.
- **Action head changed before this run** (commit e85cf17): 8 annotator classes → 4
  Law 12 families (tackle, hands, elbowing, high leg); dive ignored by the action head.
  The long runs train the 4-family head; round 4 models are converted by summing
  probabilities. The action number from this run is therefore **not comparable** to
  round 4's 0.344 — report both.
- **Run started, all setup cells passed.** At 1h 12m the queue showed `=== step wait`
  with no further output. Expected: the wait step prints nothing while both 60-epoch
  runs train in the background (`logs/long1.log`, `logs/long2.log`).
- **Colab terminal did not accept typing** while the long cell was executing. Use the
  file browser (`/content/hakam/logs/*.log`) to read progress instead. Do not add a new
  cell — it would queue behind the overnight cell.
- **Monitoring constraint:** the notebook runs in a Chrome profile the assistant's
  browser extension is not attached to, so progress checks depend on screenshots of the
  logs. Touching hakamtuw from another browser risks the sign-out loop again.
- **Test-set reuse:** round 4 already scored the test split once. `final_v3` scores it
  again after selection on validation only. State this in the report; treat the test
  numbers as confirmation, not as a tuning signal.

## 16 Sep 2026 — run finished (report written 07:43)

Every step `ok`: wait 91 min · TTA 1.7 min · long3_seed7 (80 epochs, from long1) 58 min ·
select · final · refit 5 min.

**Validation selection (per task, chosen before test was read):**

| Task | Chosen pool | Valid BA | Round 4 base (valid) |
|---|---|---|---|
| Card | all 6 models | 0.672 | 0.654 |
| Offence | long runs + flip TTA | 0.567 | 0.558 |
| Action (4 families) | long runs + flip TTA | 0.543 | 0.452 |
| Body part | long runs + mv13_attr | 0.740 | 0.708 |

**Test:**

| Task | Round 4 ensemble | final_v3 ensemble | Refit (train+valid, 5 ep) |
|---|---|---|---|
| Card | 0.606 | 0.593 | **0.642** |
| Offence | 0.610 | 0.617 | **0.640** |
| Action | 0.344 (8 classes) | 0.475 (4 families) | **0.540** (4 families) |
| Body part | 0.678 | 0.683 | 0.682 |

Observations:
- **60 epochs did not beat 5 for card.** Valid BA: base (5 ep) 0.654, long (60 ep, best
  epoch kept) 0.652. The early-peak finding from round 3 holds; epochs are not the lever.
- **Long runs helped the attribute heads:** action +9 pts, body part +3 pts on valid.
- **Card valid rose 1.8 pts but test fell 1.3 pts** — both inside the ±5-pt CI; no real
  change. Do not claim an improvement.
- **More data helped most.** The refit adds the 411 validation actions to training and
  gains 3–4 pts on card and offence on test. Its recipe and stopping epoch were fixed in
  advance (not tuned on test); thresholds come from `final_v3`.
- **Test has now been read three times** (round 4, final_v3, refit). Report this. Refit
  was pre-specified in the queue, so it is a legitimate final model, but none of these
  numbers is a pristine single-look estimate.
- Offence remains the weakest decision (valid 0.57); treat its test 0.64 cautiously —
  only ~25 no-offence test actions.
- The refit note says "thresholds from final_v2" — stale text; the code reads the
  `final_v3` thresholds.

## Checks

- [x] Both long runs finished, no OOM (wait step ok, both ran TTA)
- [x] Step outcomes recorded
- [x] final_v3 and refit test tables recorded
- [ ] Contracts for the app: `contracts_v3` comes from the final_v3 ensemble, not the refit

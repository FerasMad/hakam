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

## Checks still to record

- [ ] Latest epoch + card/offence/body/action BA for long1 and long2
- [ ] Did either run show `out of memory` or stall?
- [ ] Selected epoch per long run (best on validation)
- [ ] Step outcomes in `overnight_report.json` (tta, long3_seed7, select, final, refit)
- [ ] `final_v3` test table vs round 4 — adopt only tasks where validation improved

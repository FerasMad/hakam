# Hakam audit — 16 September 2026

Everything below was tested on `main` (947400b + the full-test scripts), not taken from
earlier reports. Vision numbers come from `scripts/full_test.py` run on Colab (A100) with the
final `refit_v3` weights on all 301 test incidents; app checks were run locally.

## 1. What was tested

| Area | Test | Result |
|---|---|---|
| Python | `pytest -q` | **153 passed**, 0 failed |
| Frontend | `npm ci`, `typecheck`, `verify`, `build` | all pass, 0 vulnerabilities |
| Backend | `/api/health`, `/api/ready`, `/api/analyze` with and without weights | correct states; real clip analysed in 1.5 s on CPU |
| Website | loads, Arabic/English switch, RTL layout, video-only picker | pass |
| Thresholds | Drive `final_v3/metrics.json` vs `weights/README.md` | **match**: offence 0.60, card 0.50, body part 0.49 |
| Vision model | full test set, two modes (below) | reproduces reported results |
| RAG + ruling | offline v3 ruling for every answered test decision | 100% on every check |
| Secrets | scan of all branches | no keys or tokens |

## 2. Vision model

| Task | All views (reported protocol) | Last clip only (website) | Reported |
|---|---|---|---|
| Card | **0.655** [0.61–0.69] | 0.609 [0.56–0.66] | 0.642 |
| Offence | **0.640** [0.51–0.79] | 0.599 [0.49–0.74] | 0.640 |
| Body part | **0.700** [0.66–0.75] | 0.716 [0.68–0.76] | 0.682 |
| Action family | 0.510 [0.43–0.59] | 0.608 [0.47–0.73] | 0.540 |
| Refer-to-human rate | 15.3% | 19.6% | — |
| Offence accuracy when it answers | **94.6%** | 93.6% | — |
| Card BA when it answers | 0.660 | 0.634 | — |
| Inference | 0.01 s per view (A100) | ~1.4 s per clip incl. decoding (laptop CPU) | — |

Findings:
- **The reported numbers hold up.** All four all-views results are within their 95% intervals of
  the reported values (card +1.3, body part +1.8, action −3.0 points).
- **The website loses accuracy by using one clip.** Card drops 4.6 points and offence 4.1 points
  versus all views; the refer-to-human rate rises from 15% to 20%. The model was trained to average
  every camera angle.
- **Offence is really a "was it a foul" detector with a strong bias to yes.** No-offence recall is
  0.41 (all views) / 0.36 (last clip) on only 22 no-offence incidents, hence the wide interval.
- **The close-up helps the action head** (0.51 → 0.61): the wide shot blurs what kind of foul it was.
- **Weak classes:** high leg recall 0.22 (all views), hands 0.35. The action is hidden from the
  ruling below 60% confidence, which is doing its job.

## 3. RAG and Arabic ruling (offline, no API)

| Check | All views (255 answered) | Last clip (242 answered) |
|---|---|---|
| Six sections present | 100% | 100% |
| No unsupported claims | 100% | 100% |
| Cites the Law | 100% | 100% |
| Hedging when a field is below 60% | 100% | 100% |
| Every Law 12 article the ruling rests on is returned | 100% | 100% |
| Hybrid vs tag-only top-3 agreement | 41.6% | 41.3% |

Findings:
- The rule-based restart and sanction plus the claim check make the offline ruling fully faithful on
  real model outputs, not only on mock contracts.
- Hybrid (embedding) retrieval changes more than half of the top-3 articles compared with tags alone.
  It never removes the ruling's own articles (those are pinned first), but the extra "related
  articles" shown in the evidence panel are embedding-driven and have not been judged by a person.
- Live GPT-5 nano was not re-tested here (no key on this laptop); Anas's recorded run stands:
  v3 faithfulness 1.00, 0% unsupported, 10/10 manual audit, ~1.7 s and ~$0.0001 per explanation.

## 4. Backend and website

Working well: strict model and threshold validation, safe errors, single-flight inference,
temporary-file cleanup, request IDs, security headers, deterministic fallback, readiness page,
bilingual RTL/LTR UI, human-review state before any retrieval or LLM call.

Issues found:
1. **The first analysis is slow.** The backend loads the vision model at start but not the
   embedding model; the first `/api/analyze` after a restart adds roughly 10 s to load it.
2. **One video only by design** (`num_views` must be 1), which costs the 4–5 points above.
3. **The static-site mount never activates.** The backend serves `web/out`, but `next.config.ts`
   has no static export, so the demo always needs two processes (FastAPI + Next.js).
4. Uploads are rejected unless the browser labels them `video/*` — fine in the website, but a
   `curl` or script upload without that type gets "Choose a browser-supported video file".

## 5. Deployment (local laptop)

The team decided on local-only; the handoff and checklists agree. For the laptop:
- Both servers must run: `python -m uvicorn backend.app:app --port 8000` and `cd web && npm run dev`
  (or `npm run build && npm start`).
- `weights/final.pt` (330 MB) and `weights/thresholds.json` must be copied to that laptop.
- The first run downloads the embedding model (~1 GB) — do it before the venue.
- `.env.example` still carries `HAKAM_WEIGHTS_REPO` / `HF_TOKEN` lines from the dropped online plan.

## 6. Repository and handoff

- Branches: `llm` and `final-integration` are merged (safe to delete). `Bader` ("UI version",
  16 Sep 04:31) predates PR #2 and would **remove** the logo, team footer, checklist, smoke test and
  Anas's LLM fixes if merged — do not merge; confirm with Bader, then delete.
- Anas's handoff names him "Project owner"; the repository, dataset work and vision model are Feras's —
  the handoff should credit each member's part.
- The presentation and the handoff are not on GitHub yet; `TEAM_CHECKLIST.md` still lists the slides
  and the PR as open.
- The handoff's "148 passed, 5 skipped" is now 153 passed.
- The old SoccerNet password is still in the **public** repository's git history.

## 7. Suggestions, in priority order

| # | Suggestion | Why | Effort |
|---|---|---|---|
| 1 | **Warm up the embedding model at backend start** (one `retrieve()` on a mock contract in the lifespan) | removes the ~10 s first-request delay on stage | 5 min |
| 2 | **Allow 1–4 videos per incident on the website** (drop the `num_views == 1` gate, multi-file upload) | +4–5 points card/offence; matches how the model was trained and evaluated | 1–2 h |
| 3 | **Present the answered-case numbers**: "when Hakam answers, it is right about the offence 94% of the time, and refers 1 in 5 cases to a human" | honest, and the strongest true claim | slides |
| 4 | **Say plainly that no-offence is the weak spot** (recall ~0.4 on 22 cases) | a grader will ask; the refer-to-human gate is the mitigation | slides |
| 5 | Demo from the **10 picked videos** (below), rehearse #9 and #10 as the honesty moments | shows grounding, abstention and error analysis | rehearsal |
| 6 | Add a one-command `run_demo.bat` that starts both servers and opens the browser | fewer steps on stage | 15 min |
| 7 | Remove the Hugging Face lines from `.env.example`; tick the done items in `TEAM_CHECKLIST.md`; commit the presentation and a corrected handoff | repo matches reality | 15 min |
| 8 | Delete merged branches; decide on `Bader`; consider rewriting history (or making the repo private) for the old password | hygiene and the dataset agreement | 10 min |
| 9 | Have a person judge the embedding-retrieved "related articles" for 10 cases, or show only the ruling's pinned articles | the only part of the evidence panel not verified | 30 min |
| 10 | Confirm the OpenAI budget alert and verify live generation once on the presentation laptop | last unchecked item for the LLM | 10 min |

## 8. Ten demo videos (website mode, last clip)

Eight are fully correct on everything the website shows; #9 and #10 are deliberate.
Paths are relative to the project folder (`...\Final Project\hakam\`). 116 of 301 test incidents are
fully correct in website mode.

| # | Incident | Shows | Clip | Referee | Model |
|---|---|---|---|---|---|
| 1 | 88 | card · tackle | `data/mvfouls/Test/action_88/clip_1.mp4` | offence · card · tackling · lower body | offence 80% · card 81% · tackle · lower body 91% |
| 2 | 240 | card · elbowing | `data/mvfouls/Test/action_240/clip_1.mp4` | offence · card · elbowing · upper body | offence 71% · card 64% · action hidden · upper body 69% |
| 3 | 204 | card · holding | `data/mvfouls/Test/action_204/clip_1.mp4` | offence · card · holding · upper body | offence 78% · card 56% (hedged) · action hidden · upper body 58% |
| 4 | 144 | severe card (severity 4) | `data/mvfouls/Test/action_144/clip_2.mp4` | offence · card · tackling · severity 4 | offence 83% · card 82% · tackle · lower body 73% |
| 5 | 62 | no card · tackle | `data/mvfouls/Test/action_62/clip_2.mp4` | offence · no card · standing tackle | offence 77% · no card 74% · tackle · lower body 70% |
| 6 | 167 | no card · holding | `data/mvfouls/Test/action_167/clip_1.mp4` | offence · no card · holding · upper body | offence 60% · no card 65% · action hidden · upper body 63% |
| 7 | 277 | no card · tackle | `data/mvfouls/Test/action_277/clip_1.mp4` | offence · no card · standing tackle | offence 79% · no card 67% · tackle · lower body 73% |
| 8 | 63 | no offence | `data/mvfouls/Test/action_63/clip_1.mp4` | no offence · challenge · upper body | no offence 61% · upper body 89% |
| 9 | 208 | refer to human | `data/mvfouls/Test/action_208/clip_2.mp4` | offence · card · standing tackle | refers to a human (offence confidence 40%) |
| 10 | 31 | honest mistake | `data/mvfouls/Test/action_31/clip_2.mp4` | offence · no card · tackling | offence 88% · **card 81%** · tackle — wrong card, confidently |

No clean high-leg or straight-red (severity 5) incident exists in website mode; incident 27 (red,
elbowing) is predicted as high leg.

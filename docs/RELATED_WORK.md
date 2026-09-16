# Related work

What has been tried on automated foul decisions and grounded explanations, and
what each result means for Hakam. Numbers are quoted from the papers.

## Vision: foul and sanction recognition

| Work | Method | Result | Lesson for Hakam |
|---|---|---|---|
| **VARS** — Held et al., CVPRW 2023; journal version 2024 | Per-view MViT encoder, attention pooling across views, multi-task heads (4-class offence+severity, 8-class foul type). 16 frames over 1 s centred on the foul, 224×224. Introduced SoccerNet-MVFoul. | Severity acc 0.46, **balanced acc 0.34**; foul type acc 0.50, BA 0.41 | Our first window (2.56 s, stride 4) was much sparser around the contact than theirs. Attention gave the live wide shot the lowest weight, which supports our last-clip (close-up) choice. |
| **VARS human study** (same journal paper) | 15 certified referees and 15 players, 77 actions, 3 views, 5 s clips at 25 fps | Referees: severity 60%, foul type 70%. Players: 58% / 75%. **Cohen's κ 0.21** (referees). Only 16% of referees agreed on the severity of the same action. | **This is the ceiling.** The card label is barely reproducible between qualified referees, so a model near 0.60 is near human level, not failing. |
| **SoccerNet 2024 challenge** | Winner M2VFCN (Kinetics-400, imbalance-aware loss). 2nd: frozen VideoChat2 UMT-L + QFormer + MLP over 4 views. | Combined BA 44.8% (baseline 37.0%) | Large pretrained encoders and explicit imbalance handling. |
| **SoccerNet 2025 challenge** | Winner TAdaFormer-L/14 (Kinetics-400/710), **16 frames at stride 2**, 720p input at 280×490, learnable live-vs-replay view embeddings, max pooling, two stages (fine-tune, then retrain heads), ensemble over frame steps 1/2/3. 3rd place: zero-shot Video-LLM prompted with IFAB rule summaries and chain-of-thought. | Combined 52.2%; offence BA 56.1% | Dense temporal sampling, higher resolution and ensembles are what moved the leaderboard. A zero-shot LLM already reaches ~49% combined. |
| **CAS-FD** — arXiv 2608.17060, 2026 | VideoMAE-Base; YOLOv8 signals find the contact frame, then 16 frames at stride 3 centred on it. Foul vs dive, 600 single-view clips. | **86% vs 74%** with uniform sampling (+12 points) | *When* the frames are taken matters more than cropping in space. This matches our Experiment 4 (spatial crop, no gain). |
| **UniSoccer / MatchVision** — Rao et al., CVPR 2025 | Soccer-specific visual encoder pretrained on 1,988 matches (SoccerReplay-1988), with task heads | Reports state of the art on SoccerNet-MVFoul | Domain pretraining helps; not available to us in the time. |

## Language: explaining the decision

| Work | Method | Result | Lesson for Hakam |
|---|---|---|---|
| **X-VARS** — Held et al., CVPRW 2024 | Fine-tuned CLIP ViT-L/14 video features plus the classifier's predictions fed into Video-ChatGPT (Vicuna). SoccerNet-XFoul: 22k question–answer pairs from 70+ referees. | Foul/severity acc 62%. Referees rated explanations 3.8/5 vs 4.0 for human referees. | The authors report **hallucinations: it describes actions that are not in the video**. Hakam's language model never sees video, only the contract, which removes that failure mode by construction. |
| **SoccerRef-Agents** — arXiv 2604.23392, 2026 | Five agents: video-to-text, Laws of the Game retrieval (2025/26 edition), 184-case precedent database, match context, and a chief agent that decides from structured outputs | Video questions from MVFoul (4-class) **40.2%** (GPT-4o 37.7%); referee rating 3.65/5 | End-to-end LLMs are weak at judging fouls from video. Grounding the explanation in retrieved Law text is the same direction as Hakam. |
| **FERA** — arXiv 2509.18527 (fencing) | Pose model → structured action tokens → FAISS retrieval of rules → LLM writes the justification; the decision itself comes from a small classifier | Decision acc 0.62, macro-F1 0.63, ECE 0.097 | Closest architecture to Hakam, in another sport. It also reports calibration, which supports our abstention threshold. |

## Deployment context

**FIFA Football Video Support (FVS)**, 2025–26: a low-cost, coach-challenge
review system using the broadcast feed, now full-time in Italy's Serie C and
Spain's Liga F and Primera Federación. It shows the demand Hakam addresses:
leagues without full VAR infrastructure still need review support.

## Where Hakam sits

- **Vision decides, language explains.** Like VARS for the decision and FERA /
  SoccerRef-Agents for grounding, with a hard boundary: the LLM receives only the
  JSON contract and retrieved Law text.
- **Abstains** below 60% confidence instead of guessing, and hedges
  low-confidence fields.
- **Honest evaluation.** Match-clustered bootstrap CIs, and a best-epoch number
  reported alongside the mean of the last three epochs.
- **Not directly comparable** to the challenge metric: Hakam's headline is binary
  card vs no card with ambiguous labels dropped, not the 4-class offence+severity
  balanced accuracy. The human κ of 0.21 is the fair reference point.

## Sources

- VARS: https://arxiv.org/abs/2304.04617
- Towards AI-powered VARS (journal, human study): https://arxiv.org/abs/2407.12483
- X-VARS: https://arxiv.org/abs/2404.06332
- SoccerNet 2024 challenge results: https://arxiv.org/abs/2409.10587
- SoccerNet 2025 challenge results: https://arxiv.org/abs/2508.19182
- CAS-FD: https://arxiv.org/abs/2608.17060
- UniSoccer / MatchVision: https://arxiv.org/abs/2412.01820
- SoccerRef-Agents: https://arxiv.org/abs/2604.23392
- FERA: https://arxiv.org/abs/2509.18527
- FIFA Football Video Support: https://inside.fifa.com/news/football-video-support-moving-forwards-and-gaining-global-momentum

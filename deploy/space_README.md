---
title: Hakam
emoji: ⚽
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# حكم (Hakam)

Arabic explanations for football foul decisions: a video model decides, a language
model explains the ruling using the IFAB Laws of the Game.

Source code: <https://github.com/FerasMad/hakam>

This Space needs these secrets (Settings → Variables and secrets):

| Secret | Purpose |
|---|---|
| `HF_TOKEN` | read access to the private weights (and cases) repos |
| `HAKAM_WEIGHTS_REPO` | e.g. `user/hakam-weights` |
| `HAKAM_CASES_REPO` | optional, enables the test-set browser |
| `OPENAI_API_KEY` | live explanations; without it the ruling is built offline from Law 12 rules |

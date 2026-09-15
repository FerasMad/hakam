# Task: the Hakam (حكم) demo app

**Owner:** Bader · **Working app:** 15 Sep evening · **Backup video:** 15 Sep night · **Presentation:** 16 Sep 2026

You are turning the existing Streamlit prototype into the app we present on stage.
This file is self-contained: hand it to your coding agent as-is. Work through the
steps in order; each one ends with a check you can see.

---

## 1. What the app shows (the story on stage)

A referee-style decision support tool, in four panels:

1. **The incident** – the video clip of a foul.
2. **The vision decision** – offence? card? with a confidence for each.
3. **The contract** – the small JSON the vision model produced. *This is the only
   thing the language model receives. It never sees the video.* This is the point
   the graders reward (brief §10), so it must be visible, not hidden.
4. **The Arabic explanation** – written by Claude from the contract plus retrieved
   Laws of the Game articles, with a faithfulness check proving it did not invent
   anything.

When the model is unsure (offence confidence < 60%), the app **refers the case to a
human** instead of explaining it. That is a feature — show it in the demo.

---

## 2. What already exists (do not rewrite)

| File | What it does |
|---|---|
| `app/main.py` | Streamlit app: sidebar input, decision, contract JSON, explanation, retrieved laws, faithfulness |
| `src/contract.py` | `HakamContract`, `should_abstain()`, `low_confidence_fields()`, `from_dict()` |
| `src/llm/generate.py` | `explain(contract)` → `Explanation(text_ar, articles, prompt_version, ...)` (Anas) |
| `src/llm/faithfulness.py` | `score(text, contract, articles)` (Anas) |
| `laws/corpus.json` | Curated Law 12 / Law 5 articles in Arabic and English |
| `artifacts/contracts/` | **Real** contracts from the trained model on the test set (you download these, step 3) |

Label vocabulary the real contracts use:

| Field | Values |
|---|---|
| `offence` | `offence`, `no_offence` |
| `card` | `card`, `no_card`, or `null` when no offence |
| `card_colour` | always `null` (colour is not predicted — never display a colour as a decision) |
| `action_class` | `tackle`, `hands`, `elbowing`, `high leg` — **absent** when the model was unsure |
| `body_part` | `upper_body`, `under_body` |

---

## 3. Setup (≈20 min)

```bash
git clone https://github.com/FerasMad/hakam.git
cd hakam
git checkout -b app
pip install -r requirements.txt
```

API key — environment variable only, **never in code, never committed**:

```bash
# Windows PowerShell
$env:ANTHROPIC_API_KEY="..."
# macOS / Linux
export ANTHROPIC_API_KEY=...
```

Real contracts (ask Feras for access to the `hakamtuw` Drive folder):

1. Download `MyDrive/hakam_colab/contracts/` (or `contracts_v3/` if Feras says the
   overnight run finished — use whichever he confirms).
2. Put its contents at `artifacts/contracts/` so you have
   `artifacts/contracts/test_index.json` and `artifacts/contracts/test/<action_id>.json`.

Clips (NDA-covered — you signed it; they stay on your laptop, never uploaded,
never committed, never in any public copy of the video):

3. Download `MyDrive/hakam_colab/data/Test.zip` and extract it to
   `data/mvfouls/Test/` so clips live at `data/mvfouls/Test/action_<id>/clip_<n>.mp4`.

**Check:** `streamlit run app/main.py` opens; the sidebar offers
**"Test set (real model)"**; choosing an action shows a decision and a contract.

If the LLM call fails, the app shows a template explanation labelled
`template fallback` — that is expected without a key, and it is the stage safety net.

---

## 4. Steps

### Step 1 — Load the clip automatically (≈30 min)
In `app/main.py`, when the source is "Test set (real model)", look for
`data/mvfouls/Test/action_<id>/` and show its clips with `st.video`. Show the
**last** clip first (it is the close-up replay), the others in an expander. Keep the
manual upload as a fallback. If the folder is missing, show a short note, not an error.

**Check:** selecting a test action plays its video with no upload.

### Step 2 — Arabic, right-to-left layout (≈45 min)
- Page title and all user-facing labels in Arabic, with English as a small caption
  underneath where useful for the graders.
- Wrap Arabic blocks in `dir="rtl"` and a readable Arabic font (e.g. *Noto Naskh
  Arabic* or *IBM Plex Sans Arabic* from Google Fonts via
  `st.markdown(..., unsafe_allow_html=True)`).
- Use the existing `LABEL_AR` map for every label; add any missing one there, not inline.

**Check:** no English label is shown alone for a decision; Arabic text is aligned right.

### Step 3 — The decision panel (≈45 min)
Replace the plain metrics with a clear verdict card:
- Three bands, using `contract.should_abstain()` and the card label:
  - **إحالة للحكم** (refer to human) — offence confidence < 60%
  - **بطاقة محتملة — تستوجب مراجعة** (flag) — offence + card
  - **لا إجراء** (clear) — no offence, or offence + no card
- Confidence as a progress bar per field, with the 60% threshold marked.
- Fields listed in `contract.low_confidence_fields()` get a visible "غير مؤكد" tag.
- If `action_class` is absent, show nothing for it (do not print "unknown").

**Check:** a low-confidence action shows the refer band and no explanation text.

### Step 4 — Make the contract panel the hero (≈30 min)
- Title it in Arabic and English: *"العقد — the only input the language model sees"*.
- Show the JSON (`st.code(..., language="json")`) next to a one-line flow:
  `video → vision model → contract → Claude + Laws → Arabic explanation`.
- A small note: "No frames, no file names, no match names are sent to the LLM."

**Check:** someone seeing the app for 5 seconds understands the LLM never sees the video.

### Step 5 — Explanation, laws and faithfulness (≈45 min)
- Show `Explanation.text_ar` in a large RTL block.
- Below it, the retrieved articles as expanders (law + section as the title, Arabic
  text inside, English text in a nested expander).
- Faithfulness: use `src.llm.faithfulness.score(text, contract, articles)` instead of
  the local helper in `main.py`, and show `faithfulness` as a percentage, `supported`
  labels as green chips, `unsupported` as red chips, `cites_article` and
  `hedged_low_conf` as ✓/✗.
- Show the engine caption (`LLM + retrieval` or `template fallback`).
- **Cache** explanations per `action_id` with `@st.cache_data` so switching back and
  forth does not call the API again (saves money and avoids stage latency).

**Check:** a normal action shows ≥ 90% faithfulness and at least one cited article.

### Step 6 — Referee-label toggle for the demo (≈20 min)
- A sidebar checkbox **"Show referee's label"** (off by default).
- When on, show the referee's label from `test_index.json` (`truth`) beside the model
  decision, with ✓ / ✗.
- It must never be passed to `explain()` — it is for the audience only.

**Check:** toggling it does not change the explanation or trigger an API call.

### Step 7 — Pick the demo cases (≈30 min)
Browse the test set and choose **four actions**; write their ids in
`app/demo_cases.json`:

| Slot | What to find |
|---|---|
| 1 | Offence + card, model **correct**, high confidence |
| 2 | No card, model **correct** |
| 3 | Low confidence → **refer to human** |
| 4 | Model **wrong** — to show honest error analysis |

Add a sidebar selector "Demo cases" that jumps to these four. Only ids go in the file —
no clips, no labels.

**Check:** the four cases load in under 3 s each once their explanations are cached.

### Step 8 — Stage safety (≈30 min)
- **Pre-warm:** a sidebar button "Prepare demo" that calls `explain()` for the four demo
  cases and caches them, so nothing waits on the network during the talk.
- **Offline mode:** if `ANTHROPIC_API_KEY` is missing or the call fails, the template
  explanation appears with its label — the app never shows a traceback.
- Run it once with Wi-Fi off to confirm.

**Check:** with Wi-Fi off, all four demo cases still render end to end.

### Step 9 — Screenshots and backup video (≈45 min)
- Screenshots of each panel for the slides: save to `docs/figures/app/`.
  **Crop or blur the video frame** in any screenshot that goes into the repo — the
  footage is NDA; the rest of the UI is fine.
- Record a 2–3 minute screen capture walking the four demo cases in order (OBS or the
  OS recorder). Keep it **local and in the team's private Drive only** — it contains
  NDA footage.

**Check:** video plays start to finish; screenshots in the repo contain no video frames.

### Step 10 — Hand over (≈15 min)
- Small commits with clear messages on branch `app`; open a PR to `main`.
- In the PR description: how to run, the four demo ids, what happens offline.

---

## 5. Rules

- **Never commit** the API key, clips, `data/`, `frame_cache/`, `artifacts/`
  (gitignored — keep it that way) or the demo recording.
- **Never pass** video, frame paths, match names or ground-truth labels to `explain()`.
  The contract JSON is the only input.
- Don't change `src/contract.py`, `src/llm/*` or `src/models/*` — ask Feras (models) or
  Anas (LLM) if you need something there.
- Never display a card colour (yellow/red) as the model's decision. The Law text may
  mention colours conditionally; that is fine inside a retrieved article.
- The app must never crash on stage: every external call is wrapped and falls back.

## 6. Done means

- [ ] Test action selected → its clip plays automatically
- [ ] Arabic RTL UI, verdict card with three bands and confidence bars
- [ ] Contract panel clearly states the LLM never sees the video
- [ ] Explanation + retrieved laws + faithfulness chips from `faithfulness.score`
- [ ] Referee-label toggle, never sent to the LLM
- [ ] Four demo cases in `app/demo_cases.json`, pre-warm button, cached explanations
- [ ] Works with Wi-Fi off (template fallback)
- [ ] Screenshots in `docs/figures/app/` (no video frames), backup video in private Drive
- [ ] PR open against `main`

## 7. Talking points you own on stage (≈1 minute)

- "The vision model decides; the language model only explains the decision it is given."
- "Everything Claude writes is checked against the contract — here is the faithfulness score."
- "When the model is unsure, it refers to a human rather than guessing."
- Case 4: "Here the model is wrong — and the confidence shows why a human stays in the loop."

Questions → Feras (model, contracts, data access) · Anas (explanations, laws).

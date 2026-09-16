# حكم (Hakam)

> Arabic explanations for football foul decisions — a video model decides, a
> language model explains it using the Laws of the Game.

## What it does

1. **Watches** a foul from several camera angles and decides: offence? card?
   It also describes the action (tackle, hands, elbowing, high leg) and body part.
2. **Writes a contract** — a small JSON with those labels and their confidence.
3. **Retrieves** the matching articles from the IFAB Laws of the Game.
4. **Explains** the ruling in Arabic: the decision, the restart (free kick / penalty),
   the disciplinary sanction, the Law it comes from, and why it is a violation — and
   checks that every claim is in the contract.

When the model is unsure (offence confidence below 60%), it refers the case to a
human instead of explaining.

```
multi-view clip
      │  VideoMAE-base, fine-tuned (all camera views)
      ▼
HakamContract (JSON: labels + confidences)   ◄── the language model sees only this
      │  tag + multilingual-E5 retrieval over 47 Law 12 / Law 5 chunks
      ▼
Law 12 rules (restart, sanction) + GPT-5 nano for the reasoning, claim check, safe fallback
      │
      ▼
Arabic ruling (decision · restart · sanction · Law · why · confidence)
```

**The language model never sees the video.** It can only state what the contract
contains, which makes grounding structural rather than a prompt instruction.

## Results (test set)

| Task | Balanced accuracy |
|---|---|
| Card | **0.64** |
| Offence | **0.64** |
| Body part | **0.68** |
| Action family (4 classes) | **0.54** |
| Explanation faithfulness (prompt v3) | **1.00**, 0% unsupported claims |

Details, experiment history and limitations: [`docs/RESULTS.md`](docs/RESULTS.md).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements-app.txt
cp .env.example .env              # Windows: copy .env.example .env
# open .env and paste your OPENAI_API_KEY
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
cd web
cp .env.example .env.local         # Windows: copy .env.example .env.local
npm ci
npm run dev
```

Put the model weights in `weights/` (see [`weights/README.md`](weights/README.md)).
Open <http://127.0.0.1:3000>; the local API runs at <http://127.0.0.1:8000>.

| Run | Command |
|---|---|
| Test the model on your own clip | `python scripts/try_video.py my_foul.mp4` |
| Contract only (JSON) | `python -m src.inference.predict clip_0.mp4 clip_1.mp4` |
| One explanation in the terminal | `python scripts/demo_llm.py` |
| LLM evaluation (v1 vs v2 vs v3) | `python scripts/eval_llm.py` |
| Tests | `python -m pytest -q` |
| Local smoke test | `python scripts/smoke_test.py http://127.0.0.1:8000 --frontend-url http://127.0.0.1:3000 --clip your_clip.mp4` |

Without an API key the ruling is still produced, built from the contract and the Law 12
rules, and labelled offline.

Real test-set contracts go in `artifacts/contracts/` (shared privately — they are
derived from the NDA dataset).

## Presentation setup

Hakam is delivered as a local presentation application. Keep the API and frontend
terminals running on the presentation laptop, store `OPENAI_API_KEY` only in the
gitignored `.env`, and keep the deterministic Arabic fallback available if the venue
has no internet connection.

## Repository

| Path | Contents |
|---|---|
| `src/inference/` | Live inference: video clips → contract |
| `src/contract.py` | The contract between the vision model and the language model |
| `src/llm/` | Retrieval, prompts, generation, faithfulness check, Arabic lexicon |
| `src/models/` | Datasets, multi-view multi-task training, metrics |
| `src/data/` | Dataset loading, label derivation, preprocessing, augmentation |
| `backend/` | FastAPI production API, runtime validation, and pipeline adapter |
| `web/` | Next.js Arabic/English review interface |
| `laws/` | 47 curated bilingual IFAB rule chunks and their prebuilt embeddings |
| `notebooks/train_colab.ipynb` | Full training run on a Colab A100 |
| `scripts/` | Frame caching, ensembling, contracts, LLM demo and evaluation |
| `docs/` | Results, language-layer report, preprocessing report, related work |
| `tests/` | Unit tests (no API calls, no dataset needed) |

## Data

The dataset is **not** in this repository and cannot be redistributed.
**SoccerNet-MVFoul**: 3,901 foul incidents from 500 matches, 2–4 camera views each,
annotated by a professional referee. Access requires signing the agreement at
<https://github.com/SoccerNet/sn-mvfoul>. See [`data/README.md`](data/README.md).

## Related work

- **VARS** (Held et al., CVPR 2023 Workshop) introduced SoccerNet-MVFoul.
- **X-VARS** feeds video directly into a multimodal LLM and explains in English.

Hakam explains in Arabic, grounds every explanation in the Laws of the Game text,
and isolates the language model from the video. More in
[`docs/RELATED_WORK.md`](docs/RELATED_WORK.md).

## Disclaimer

A research prototype built for the Tuwaiq AI bootcamp. It does not make refereeing
decisions; it explains decisions and declares low confidence rather than guessing.

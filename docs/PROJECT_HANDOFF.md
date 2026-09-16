# Hakam (حكم) Project Handoff

- **Last updated:** 16 September 2026
- **Team:** Feras Madkhali (repository, dataset, vision model, integration) · Anas Alzahrani (RAG and language layer, app integration) · Bader Aljoudi (web app)
- **Prepared by:** Anas Alzahrani · **Prepared for:** Feras Madkhali
- **Delivery target:** Local Tuwaiq Academy capstone demonstration
- **Deployment decision:** No public or online deployment is required
- **Canonical repository:** <https://github.com/FerasMad/hakam/tree/main>

## 1. Executive summary

Hakam is a local AI-assisted football incident review application. A user uploads one football incident clip, the vision model predicts whether it is an offence and whether a card is required, and the language layer explains the result in Arabic using retrieved Laws of the Game evidence.

The main objective is to help football fans answer three questions:

1. Was the incident a foul?
2. Should a card be issued?
3. Which football law supports the ruling?

The integrated application is working locally. The FastAPI backend loads the real trained checkpoint, the Next.js frontend calls the real backend, and the OpenAI-assisted Arabic explanation layer is connected to a grounded RAG pipeline. The complete tracked code integration is available from the GitHub `main` branch and was merged through PR #2.

## 2. Source of truth for Feras

The GitHub `main` branch is the canonical source of truth:

<https://github.com/FerasMad/hakam/tree/main>

Feras should start from a clean clone or an updated local `main` branch:

```bash
git clone https://github.com/FerasMad/hakam.git
cd hakam
git switch main
git pull --ff-only origin main
```

Main GitHub locations:

| Item | GitHub location |
|---|---|
| Project overview and setup | [README.md](https://github.com/FerasMad/hakam/blob/main/README.md) |
| Frontend | [web](https://github.com/FerasMad/hakam/tree/main/web) |
| Backend | [backend](https://github.com/FerasMad/hakam/tree/main/backend) |
| Vision and language code | [src](https://github.com/FerasMad/hakam/tree/main/src) |
| RAG law corpus | [laws](https://github.com/FerasMad/hakam/tree/main/laws) |
| Model results | [docs/RESULTS.md](https://github.com/FerasMad/hakam/blob/main/docs/RESULTS.md) |
| RAG and LLM report | [docs/llm/REPORT.md](https://github.com/FerasMad/hakam/blob/main/docs/llm/REPORT.md) |
| Related work | [docs/RELATED_WORK.md](https://github.com/FerasMad/hakam/blob/main/docs/RELATED_WORK.md) |
| Model artifact instructions | [weights/README.md](https://github.com/FerasMad/hakam/blob/main/weights/README.md) |
| Merged integration | [PR #2](https://github.com/FerasMad/hakam/pull/2) |

### Runtime artifact exception

The tracked project code is in `main`, but the two local runtime files below are deliberately gitignored:

```text
weights/final.pt
weights/thresholds.json
```

`final.pt` is approximately 330 MB and is not stored in Git. Download it from the shared project Drive, under `hakam_colab/runs/refit_v3/final.pt`:

<https://drive.google.com/drive/u/2/folders/1wPsUBAkEdJ3Htfm7OAaEqXeAPFe8-8dv>

The required threshold values are documented in [weights/README.md](https://github.com/FerasMad/hakam/blob/main/weights/README.md). Both files must be placed inside `weights/` before the backend can report ready.

Anas's working copy is:

```text
/Users/anas/Desktop/hakam-main
```

Do not use the older Codex copy at:

```text
/Users/anas/.codex/.chatgpt-projects/g-p-6a990fc8f1bc819186c2f561b8ab436e/hakam-production
```

That directory still exists, but it is not a source of truth. Its frontend previously did not work correctly. Finalization should start from GitHub `main`.

## 3. Decisions made during the project

- The project will run locally for the capstone presentation.
- No Hugging Face Space, public URL, cloud backend, or online deployment is required.
- The OpenAI API key must stay local and must never be committed to Git.
- The browser frontend must never receive the OpenAI key through a `NEXT_PUBLIC_` variable.
- The locally supplied `final.pt` model and verified thresholds are the production inference artifacts documented by the repository.
- Low-confidence offence decisions are sent to human review instead of forcing a ruling.
- The language model never receives the video. It receives only a structured contract and retrieved law text.
- The tracked final application is maintained in the GitHub `main` branch.

## 4. Work completed

### Anas's documented contribution and ownership

- Owned the AI and language-layer part of the capstone integration.
- Supplied and kept the OpenAI API key local for live testing.
- Directed the final product to run locally instead of being deployed online.
- Reviewed the offence, no-offence, and human-review behavior in the working UI.
- Completed the prompt v3 evaluation and Arabic-output quality work with the final RAG pipeline.
- Confirmed Bader's working application was integrated into the GitHub `main` branch.
- Directed the final website branding, logo, Tuwaiq Academy wording, and team attribution.
- Requested and reviewed the final capstone presentation narrative and visual theme.
- Coordinated the final integration into the GitHub repository.

### Vision and model integration

- Connected the real `hakam-refit-v3` VideoMAE checkpoint.
- Confirmed strict checkpoint loading with four heads:
  - offence
  - card
  - action class
  - body part
- Confirmed the validated thresholds:
  - offence: `0.60`
  - card: `0.50`
  - body part: `0.49`
- Connected live video inference to the backend.
- Preserved multi-view averaging when several views are available.
- Added confidence-based abstention instead of guessing.

### FastAPI backend

- Added a production-style local FastAPI service.
- Added `/api/health`, `/api/ready`, and `/api/analyze` endpoints.
- Added strict runtime validation for the checkpoint and thresholds.
- Added streamed uploads with a default 200 MB limit.
- Added safe temporary-file cleanup.
- Added serialized inference to avoid concurrent model conflicts.
- Added safe error responses, request IDs, CORS configuration, and security headers.
- Added deterministic Arabic fallback behavior when OpenAI is unavailable.

### Next.js frontend

- Replaced the prototype scenario flow with the real API integration.
- Added one-video upload and browser-side validation.
- Added processing, result, error, and human-review states.
- Added offence, card, action, body-part, and confidence displays.
- Added Arabic ruling and retrieved law evidence sections.
- Added the 60% human-review threshold screen.
- Added bilingual English and Arabic interface copy.
- Added the Hakam robot referee logo.
- Added the Tuwaiq Academy capstone attribution.
- Added team names with LinkedIn links:
  - [Anas Alzahrani](https://www.linkedin.com/in/anas-alzahrani-932b38319)
  - [Feras Madkhali](https://www.linkedin.com/in/feras-madkhali-0b639b398)
  - [Bader Aljoudi](https://www.linkedin.com/in/bader-aljoudi-139a62299)

### RAG and LLM layer

- Built a RAG pipeline over 47 curated bilingual Law 12, Law 5, and glossary chunks.
- Combined exact contract-tag retrieval with multilingual E5 semantic retrieval.
- Kept the technical restart and disciplinary sanction deterministic.
- Limited GPT-5 nano to writing the Arabic reasoning line.
- Added faithfulness checking, up to two repair attempts, and a deterministic safe fallback.
- Improved the prompt from v1 to v3.
- Corrected Arabic wording for simulation, severity inference, and action phrasing.
- Added regression tests for the language-layer fixes.

### Presentation

- Created an eight-slide PowerPoint deck in the same visual theme as the web app.
- Included the project motivation, problem statement, architecture, model results, RAG design, language evaluation, web experience, limitations, and next steps.
- Included an editable native chart for the vision-model progression.
- Added research sources in the PowerPoint speaker notes.
- Visually reviewed every slide after rendering.
- Final file:

```text
presentation/Hakam_Project_Presentation.pptx
```

The PowerPoint was created after PR #2 and is not yet stored on GitHub `main`. Anas should send it with this handoff or commit both files before Feras begins finalization.

## 5. End-to-end architecture

```text
Uploaded incident clip
        |
        v
VideoMAE-base vision model
        |
        | predicts offence, card, body part, and action family
        v
HakamContract JSON
        |
        | exact tags plus multilingual E5 retrieval
        v
Relevant IFAB law chunks
        |
        | deterministic restart and sanction rules
        | GPT-5 nano generates only the Arabic reasoning line
        | claim checking, repair, and safe fallback
        v
Arabic ruling with decision, restart, sanction, law, reason, and confidence
```

The LLM never sees the uploaded video. This is an architectural boundary, not only a prompt instruction.

## 6. Important offence and no-offence behavior

The vision head supports both `offence` and `no_offence` labels.

The product does not treat every score below the offence threshold as a confident no-offence ruling. The selected offence or no-offence label must itself have at least 60% confidence. If the selected class has less than 60% confidence, the backend returns:

```text
Human review required
```

For example, a result near the decision boundary can be shown as low-confidence human review rather than as a definite no-offence decision. A clear `no_offence` result is possible when the model selects `no_offence` with sufficient confidence. When `no_offence` is selected, the card stage is removed from the contract.

This conservative behavior is intentional because the no-offence test subset is small and the product should not issue an unjustified ruling.

## 7. Reported model results

Final test-set balanced accuracy:

| Task | Balanced accuracy |
|---|---:|
| Card | 0.64 |
| Offence | 0.64 |
| Body part | 0.68 |
| Action family, four classes | 0.54 |

Card validation progression:

| Experiment | Balanced accuracy |
|---|---:|
| Frozen VideoMAE-small baseline | 0.506 |
| Experiment 3 | 0.583 |
| Multi-view round 4 | 0.654 |
| Final overnight run | 0.672 |

The largest single improvement came from using all available camera views.

### Existing-paper comparison

The VARS paper reports `0.34` balanced accuracy on a harder four-class severity task and `0.41` balanced accuracy on foul type. Hakam reached `0.64` balanced accuracy on binary card and binary offence tasks.

This demonstrates improvement in the same SoccerNet-MVFoul research direction, but it is not a strict like-for-like benchmark because the task definitions differ. Present this comparison as directional and keep that caveat visible.

## 8. RAG and language-layer results

The language evaluation used 24 contracts across three prompt versions, for 72 total outputs.

| Measure | Prompt v1 | Prompt v3 |
|---|---:|---:|
| Mean label faithfulness | 78.4% | 100% |
| Unsupported outputs | 54.2% | 0% |
| Low-confidence hedging | 33.3% | 100% |

Additional results:

- Correct abstention: 100%
- Manual v3 audit: 10 of 10 passed
- Wrong claims in the manual audit: zero
- One measured live request:
  - 1,549 input tokens
  - 76 output tokens
  - 1.704 seconds
  - approximately $0.000108 at the recorded pricing

These figures are documented in `docs/llm/REPORT.md`.

## 9. Verification already completed

- Python test suite: 148 passed, 5 skipped (re-run on 16 September: 153 passed).
- Frontend invariant verification: passed.
- TypeScript checking: passed.
- Optimized frontend build: passed.
- Real-checkpoint local smoke test: passed.
- Live GPT-5 nano explanation: passed.
- Explanation faithfulness: 1.00.
- Unsupported claims in the verified v3 output: zero.
- PowerPoint integrity and layout validation: passed.
- PowerPoint slide count: eight.
- Editable chart validation: passed.

At the time this handoff was written:

- `http://127.0.0.1:8000/api/health` returned HTTP 200 and `{"status":"ok"}`.
- `http://127.0.0.1:8000/api/ready` returned HTTP 200 and reported:
  - model loaded
  - model version `hakam-refit-v3`
  - CPU device
  - all four tasks available
  - hybrid retrieval
  - LLM configured
- `http://127.0.0.1:3000/` returned HTTP 200.

## 10. Local setup and run instructions

Start from the GitHub `main` branch, then place `final.pt` and `thresholds.json` in the local `weights/` directory as described above.

### Backend

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-app.txt
cp .env.example .env
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

### Frontend

In a second terminal:

```bash
cd web
cp .env.example .env.local
npm ci
npm run dev
```

Open:

```text
http://127.0.0.1:3000
```

The local API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### OpenAI key

The OpenAI key was available to Anas's running backend when this handoff was created. No API key is stored in GitHub.

Before restarting the backend, make sure `OPENAI_API_KEY` is available to the backend process. One supported approach is:

```bash
cp .env.example .env
```

Then edit `.env` locally and add the key. The file is gitignored. Never paste the key into source code, `web/.env.local`, a screenshot, the handoff, or Git history.

The application still returns a deterministic grounded Arabic result when the OpenAI key is unavailable.

## 11. Useful commands

```bash
# Backend and language tests
python -m pytest -q

# Frontend verification
cd web
npm run typecheck
npm run verify
npm run build

# Check backend readiness
curl http://127.0.0.1:8000/api/ready

# Test one local clip directly
python scripts/try_video.py /absolute/path/to/clip.mp4

# Full local smoke test while both servers are running
python scripts/smoke_test.py \
  http://127.0.0.1:8000 \
  --frontend-url http://127.0.0.1:3000 \
  --clip /absolute/path/to/approved_clip.mp4

# Language-layer demo and evaluation
python scripts/demo_llm.py
python scripts/eval_llm.py
```

## 12. Git and GitHub status

Repository remote:

```text
https://github.com/FerasMad/hakam.git
```

Integration history:

| Commit | Description |
|---|---|
| `12e86a9` | Integrate Hakam local web app and final language layer |
| `728ff62` | Add Hakam logo and team LinkedIn links |
| `3297092` | Update Tuwaiq Academy project attribution |
| `947400b` | Merge GitHub PR #2 into `main` |

Pull request:

- [PR #2: Integrate the local Hakam web app and final language layer](https://github.com/FerasMad/hakam/pull/2)
- Status: merged
- Source branch: `final-integration`
- Target branch: `main`

Anas's local checkout at handoff time:

```text
branch: final-integration
HEAD: 3297092
origin/main: 947400b
```

Feras does not need the `final-integration` branch for code finalization. The merged code is already in `main` at merge commit `947400b`.

The presentation and this handoff file were created after PR #2. They are not yet committed or pushed, so Anas must send them directly to Feras or add them to `main`.

For Anas to add them to `main`, first review the files, then run:

```bash
cd /Users/anas/Desktop/hakam-main
git switch main
git pull --ff-only origin main
git add PROJECT_HANDOFF.md presentation/Hakam_Project_Presentation.pptx
git commit -m "Add project handoff and capstone presentation"
git push origin main
```

Do not use `git add .` unless the complete status has been reviewed first.

## 13. Known limitations and remaining actions

### Known limitations

- Hakam is a research prototype, not an official refereeing authority.
- The model sees short, low-resolution clips and may miss wider match context.
- Referee interpretation and strictness vary across examples.
- The no-offence test subset is small, with approximately 25 incidents.
- A perfectly faithful language explanation can still explain a wrong vision prediction.
- CPU inference can be slower than GPU inference.
- Dataset clips are restricted and cannot be redistributed.
- The RAG corpus focuses on Law 12, Law 5, and the current glossary rather than every football law.

### Feras finalization checklist

1. Clone or update <https://github.com/FerasMad/hakam/tree/main>.
2. Confirm the checkout contains merge commit `947400b` or a newer `main` commit.
3. Obtain `PROJECT_HANDOFF.md` and `Hakam_Project_Presentation.pptx` from Anas if they have not been committed yet.
4. Download `weights/final.pt` from the shared Drive and place it in `weights/`.
5. Create `weights/thresholds.json` using the values in [weights/README.md](https://github.com/FerasMad/hakam/blob/main/weights/README.md), or obtain Anas's verified local copy.
6. Create a local `.env` and add `OPENAI_API_KEY`. Do not commit it.
7. Install the Python and Node dependencies.
8. Run the Python tests and frontend verification commands.
9. Start both servers and confirm `/api/ready` reports `ready`.
10. Run one approved-clip smoke test.
11. Verify live OpenAI generation and the deterministic offline fallback.
12. Perform the final responsive visual check on the presentation laptop.

### Remaining owner actions

- Confirm the OpenAI project budget and usage alert in the OpenAI owner account.
- Perform one final visual and responsive check on the actual presentation laptop.
- Run one final approved-clip smoke test on the presentation laptop.
- Verify both live OpenAI generation and the offline deterministic fallback.
- Commit and push `PROJECT_HANDOFF.md` and the PowerPoint presentation.
- Keep the trained checkpoint, Python environment, Node dependencies, and approved test clip on the presentation laptop.

## 14. Presentation-day checklist

1. Connect the laptop to power.
2. Confirm `weights/final.pt` and `weights/thresholds.json` exist.
3. Activate the Python virtual environment.
4. Start FastAPI on port 8000.
5. Start Next.js on port 3000.
6. Open the web app and verify the upload screen.
7. Check `/api/ready` before the demonstration.
8. Run an approved short clip once before the audience arrives.
9. Keep a second clip ready.
10. Keep the PowerPoint open as a backup narrative.
11. If OpenAI or internet access fails, use the deterministic fallback and explain that the core law-grounded pipeline still works locally.

## 15. Main documentation

| Document | GitHub reference and purpose |
|---|---|
| [README.md](https://github.com/FerasMad/hakam/blob/main/README.md) | Product overview and local quick start |
| [TEAM_CHECKLIST.md](https://github.com/FerasMad/hakam/blob/main/TEAM_CHECKLIST.md) | Original integration checklist |
| [docs/RESULTS.md](https://github.com/FerasMad/hakam/blob/main/docs/RESULTS.md) | Model experiments, metrics, and limitations |
| [docs/llm/REPORT.md](https://github.com/FerasMad/hakam/blob/main/docs/llm/REPORT.md) | RAG and language evaluation |
| [docs/RELATED_WORK.md](https://github.com/FerasMad/hakam/blob/main/docs/RELATED_WORK.md) | VARS, X-VARS, and comparison context |
| [docs/DEMO_CHECKLIST.md](https://github.com/FerasMad/hakam/blob/main/docs/DEMO_CHECKLIST.md) | Local demonstration checks |
| [backend/README.md](https://github.com/FerasMad/hakam/blob/main/backend/README.md) | API setup and endpoint details |
| [web/README.md](https://github.com/FerasMad/hakam/blob/main/web/README.md) | Frontend setup and integration details |
| [weights/README.md](https://github.com/FerasMad/hakam/blob/main/weights/README.md) | Checkpoint requirements |

`TEAM_CHECKLIST.md` contains older unchecked items for creating presentation slides and opening the integration PR. Those status lines are stale: PR #2 is merged and the eight-slide presentation now exists. The presentation and this handoff still need their own final commit.

## 16. Final status

The tracked application is integrated into GitHub `main` and has been verified locally with the real model and grounded language layer. The website branding and team attribution are complete. The capstone PowerPoint is complete. No online deployment is required.

The main unfinished administrative step is to send or commit this handoff file and the PowerPoint presentation. The runtime checkpoint must also be transferred separately because it is intentionally not stored in GitHub.

# Hakam local API

The FastAPI layer accepts one uploaded video and adapts the existing Python
pipeline to the web frontend. It does not duplicate model decisions,
thresholding, abstention, retrieval, ruling, or faithfulness logic.

## Install and run

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt
cp .env.example .env
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

`transformers==5.8.0` is intentionally pinned because `final.pt` was trained
with that VideoMAE parameter layout. Newer releases use incompatible attention
parameter names and fail the backend's strict checkpoint load.

Windows PowerShell activation and copy commands:

```powershell
.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Required files are `weights/final.pt` and `weights/thresholds.json`. The
backend refuses readiness rather than using fake weights or default thresholds
when either artifact is invalid. `OPENAI_API_KEY` is optional: without it, the
existing deterministic v3 Arabic explanation is returned.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Process liveness |
| `GET` | `/api/ready` | Model, artifact, head, device, and threshold readiness |
| `POST` | `/api/analyze` | Analyze exactly one multipart `video` upload |

OpenAPI documentation is available locally at `http://localhost:8000/docs`.
The default CORS allowlist contains `http://localhost:3000` and
`http://127.0.0.1:3000`; override it with comma-separated
`HAKAM_CORS_ORIGINS` values if needed.

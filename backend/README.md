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
python -m pip install -r requirements-app.txt
cp .env.example .env
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Production inference constructs the VideoMAE-base architecture locally and then
strictly loads every checkpoint tensor. It therefore needs no architecture download,
and incompatible checkpoints fail readiness instead of partially loading.

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
Uploads are streamed to a temporary file and capped at 200 MB by default; override
the limit with `HAKAM_MAX_UPLOAD_BYTES`. Every response includes `X-Request-ID`,
`X-Content-Type-Options: nosniff`, and a no-referrer policy. When `web/out` exists,
the backend serves the production frontend from `/`.

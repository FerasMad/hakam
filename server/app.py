"""Hakam API.

    uvicorn server.app:app --reload          # http://localhost:8000/docs

Routes are thin wrappers around server/services.py. When the React app has been
built (web/dist), it is served from / so the demo is a single command.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import services  # noqa: E402

app = FastAPI(title="Hakam API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


def _bad_request(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@app.get("/api/health")
def health() -> dict:
    return services.health()


@app.post("/api/predict")
async def predict(files: list[UploadFile] = File(...)) -> dict:
    payload = [(f.filename or "clip.mp4", await f.read()) for f in files]
    try:
        return services.predict_clips(payload)
    except ValueError as exc:
        raise _bad_request(exc)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"vision model unavailable: {exc}")


@app.post("/api/explain")
def explain(contract: dict = Body(...)) -> dict:
    try:
        return services.explain_contract(contract)
    except (KeyError, TypeError, ValueError) as exc:
        raise _bad_request(exc)


@app.get("/api/cases")
def cases() -> list[dict]:
    return services.list_cases()


@app.get("/api/cases/{action_id}")
def case(action_id: str, truth: bool = False) -> dict:
    try:
        return services.get_case(action_id, truth)
    except ValueError as exc:
        raise _bad_request(exc)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such test case")


@app.get("/api/cases/{action_id}/clips/{n}")
def clip(action_id: str, n: int) -> FileResponse:
    try:
        return FileResponse(services.clip_path(action_id, n), media_type="video/mp4")
    except ValueError as exc:
        raise _bad_request(exc)
    except KeyError:
        raise HTTPException(status_code=404, detail="no such clip")


@app.get("/api/demo-cases")
def demo_cases() -> list[dict]:
    return services.demo_cases()


_WEB = Path(__file__).resolve().parent.parent / "web" / "dist"
if _WEB.exists():
    app.mount("/", StaticFiles(directory=_WEB, html=True), name="web")

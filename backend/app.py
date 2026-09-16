from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.errors import (
    APIError,
    ArtifactValidationError,
    InferencePipelineError,
    InvalidVideoError,
    RulingPipelineError,
)
from backend.pipeline import AnalysisPipeline
from backend.runtime import (
    LoadedPredictor,
    RuntimeState,
    load_validated_predictor,
    read_thresholds,
    weights_directory,
)
from backend.schemas import (
    AnalysisResponse,
    ArtifactStatusPayload,
    ErrorPayload,
    LivenessPayload,
    ReadinessPayload,
)


logger = logging.getLogger("hakam.backend")
PredictorLoader = Callable[[Path, str | None, dict[str, float]], LoadedPredictor]
VideoValidator = Callable[[Path], None]


def validate_video_file(path: Path) -> None:
    """Decode one frame before starting expensive model inference."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise InvalidVideoError("video container could not be opened")
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None or frame.size == 0:
        raise InvalidVideoError("video contains no decodable frames")


def _origins() -> list[str]:
    raw = os.getenv(
        "HAKAM_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]


def _readiness(runtime: RuntimeState) -> ReadinessPayload:
    return ReadinessPayload(
        status="ready" if runtime.ready else "not_ready",
        modelLoaded=runtime.ready,
        modelVersion=runtime.model_version,
        device=runtime.device,
        tasks=runtime.tasks,
        thresholds=runtime.thresholds,
        artifacts=ArtifactStatusPayload(
            finalPt=runtime.final_pt_present,
            thresholdsJson=runtime.thresholds_present,
        ),
        reasonCode=runtime.reason_code,
    )


def _public_error(status_code: int, code: str, message: str) -> JSONResponse:
    payload = ErrorPayload(error={"code": code, "message": message})
    return JSONResponse(status_code=status_code, content=payload.model_dump())


async def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
        suffix = ".video"

    temp_dir = os.getenv("HAKAM_TEMP_DIR") or None
    if temp_dir:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)

    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix="hakam-upload-",
        suffix=suffix,
        dir=temp_dir,
        delete=False,
    )
    path = Path(handle.name)
    size = 0
    try:
        with handle:
            while chunk := await upload.read(1024 * 1024):
                handle.write(chunk)
                size += len(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    if size == 0:
        path.unlink(missing_ok=True)
        raise APIError(400, "INVALID_VIDEO", "The uploaded video is empty.")
    return path


def create_app(
    predictor_loader: PredictorLoader = load_validated_predictor,
    video_validator: VideoValidator = validate_video_file,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        runtime = RuntimeState()
        directory = weights_directory()
        runtime.final_pt_present = (directory / "final.pt").is_file()
        runtime.thresholds_present = (directory / "thresholds.json").is_file()
        application.state.runtime = runtime
        application.state.pipeline = None
        application.state.analysis_lock = asyncio.Lock()

        try:
            thresholds = read_thresholds(directory / "thresholds.json")
            runtime.thresholds = thresholds
            loaded = await asyncio.to_thread(
                predictor_loader,
                directory,
                os.getenv("HAKAM_DEVICE") or None,
                thresholds,
            )
            runtime.predictor = loaded.predictor
            runtime.tasks = loaded.tasks
            runtime.model_version = loaded.model_version
            runtime.device = loaded.device
            runtime.reason_code = None
            application.state.pipeline = AnalysisPipeline(loaded.predictor)
            logger.info(
                "Hakam model ready: version=%s device=%s tasks=%s",
                loaded.model_version,
                loaded.device,
                ",".join(loaded.tasks),
            )
        except ArtifactValidationError as exc:
            runtime.reason_code = exc.code
            logger.exception("Hakam model is not ready")
        except Exception:
            runtime.reason_code = "MODEL_LOAD_FAILED"
            logger.exception("Unexpected error while loading the Hakam model")

        yield

        runtime.predictor = None
        application.state.pipeline = None

    application = FastAPI(
        title="Hakam API",
        version="1.0.0",
        description="Local one-video integration for the existing Hakam pipeline.",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Accept", "Content-Type"],
    )

    @application.exception_handler(APIError)
    async def api_error_handler(_: Request, exc: APIError) -> JSONResponse:
        return _public_error(exc.status_code, exc.code, exc.message)

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, __: RequestValidationError) -> JSONResponse:
        return _public_error(400, "INVALID_REQUEST", "Exactly one video file is required.")

    @application.exception_handler(Exception)
    async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled backend error",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        return _public_error(500, "INTERNAL_ERROR", "The analysis could not be completed.")

    @application.get("/api/health", response_model=LivenessPayload)
    async def health() -> LivenessPayload:
        return LivenessPayload()

    @application.get(
        "/api/ready",
        response_model=ReadinessPayload,
        response_model_exclude_none=True,
        responses={503: {"model": ReadinessPayload}},
    )
    async def ready(request: Request) -> Any:
        payload = _readiness(request.app.state.runtime)
        if payload.status == "not_ready":
            return JSONResponse(
                status_code=503,
                content=payload.model_dump(exclude_none=True),
            )
        return payload

    @application.post(
        "/api/analyze",
        response_model=AnalysisResponse,
        responses={
            400: {"model": ErrorPayload},
            500: {"model": ErrorPayload},
            503: {"model": ErrorPayload},
        },
    )
    async def analyze(
        request: Request,
        video: Annotated[list[UploadFile], File(description="Exactly one incident video")],
    ) -> AnalysisResponse:
        if len(video) != 1:
            raise APIError(400, "INVALID_VIDEO_COUNT", "Exactly one video file is required.")

        upload = video[0]
        if not upload.filename or not (upload.content_type or "").lower().startswith("video/"):
            await upload.close()
            raise APIError(400, "INVALID_VIDEO", "Choose a browser-supported video file.")

        runtime: RuntimeState = request.app.state.runtime
        pipeline: AnalysisPipeline | None = request.app.state.pipeline
        if not runtime.ready or pipeline is None:
            await upload.close()
            raise APIError(503, "MODEL_NOT_READY", "The analysis model is not ready.")

        temp_path: Path | None = None
        try:
            temp_path = await _save_upload(upload)
            await asyncio.to_thread(video_validator, temp_path)
            async with request.app.state.analysis_lock:
                return await asyncio.to_thread(pipeline.analyze, temp_path)
        except InvalidVideoError as exc:
            logger.info("Rejected an unreadable video upload: %s", type(exc).__name__)
            raise APIError(400, "INVALID_VIDEO", "The uploaded video could not be decoded.") from exc
        except InferencePipelineError as exc:
            logger.exception("Vision inference failed")
            raise APIError(500, "INFERENCE_FAILED", "Video inference failed.") from exc
        except RulingPipelineError as exc:
            logger.exception("Ruling and explanation generation failed")
            raise APIError(500, "RULING_FAILED", "The grounded ruling could not be produced.") from exc
        except APIError:
            raise
        except Exception as exc:
            logger.exception("Unexpected analysis request failure")
            raise APIError(500, "INTERNAL_ERROR", "The analysis could not be completed.") from exc
        finally:
            await upload.close()
            if temp_path is not None:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Temporary upload cleanup failed", exc_info=True)

    return application


app = create_app()

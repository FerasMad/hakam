from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import create_app, validate_video_file
from backend.errors import ArtifactValidationError, InvalidVideoError
from backend.runtime import LoadedPredictor
from src.contract import HakamContract, Prediction


class RecordingPredictor:
    def __init__(self, *, fail: bool = False, confidence: float = 0.59):
        self.fail = fail
        self.confidence = confidence
        self.seen_paths: list[Path] = []
        self.predict_calls = 0

    def predict(self, paths):
        self.predict_calls += 1
        self.seen_paths = list(paths)
        if self.fail:
            raise RuntimeError("simulated inference failure")
        return HakamContract(
            action_id="api-test",
            offence=Prediction("offence", self.confidence),
            card=Prediction("card", 0.75),
            attributes={"body_part": Prediction("under_body", 0.72)},
            model_version="hakam-final",
            num_views=len(paths),
        )


def _artifact_directory(tmp_path: Path) -> Path:
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "final.pt").write_bytes(b"test-checkpoint")
    (weights / "thresholds.json").write_text(
        json.dumps({"offence": 0.6, "card": 0.5, "body_part": 0.49}),
        encoding="utf-8",
    )
    return weights


def _loaded(predictor: RecordingPredictor, thresholds: dict[str, float]) -> LoadedPredictor:
    return LoadedPredictor(
        predictor=predictor,
        thresholds=thresholds,
        tasks=["action_class", "body_part", "card", "offence"],
        model_version="hakam-final",
        device="cpu",
    )


def test_real_video_validator_rejects_unreadable_file(tmp_path):
    path = tmp_path / "invalid.mp4"
    path.write_bytes(b"not a video container")

    try:
        validate_video_file(path)
    except InvalidVideoError:
        pass
    else:
        raise AssertionError("unreadable input must be rejected")


def test_one_video_request_and_temporary_file_cleanup(tmp_path, monkeypatch):
    weights = _artifact_directory(tmp_path)
    uploads = tmp_path / "uploads"
    predictor = RecordingPredictor()
    monkeypatch.setenv("HAKAM_WEIGHTS_DIR", str(weights))
    monkeypatch.setenv("HAKAM_TEMP_DIR", str(uploads))

    def loader(_directory, _device, thresholds):
        return _loaded(predictor, thresholds)

    with TestClient(create_app(loader, lambda _path: None)) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        readiness = client.get("/api/ready")
        assert readiness.status_code == 200
        assert readiness.json()["thresholds"] == {
            "offence": 0.6,
            "card": 0.5,
            "body_part": 0.49,
        }

        response = client.post(
            "/api/analyze",
            files=[("video", ("incident.mp4", b"video-bytes", "video/mp4"))],
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "abstained"
    assert payload["contract"]["num_views"] == 1
    assert payload["contract"]["card_colour"] is None
    assert len(predictor.seen_paths) == 1
    assert not predictor.seen_paths[0].exists()
    assert list(uploads.iterdir()) == []


def test_rejects_multiple_videos_before_inference(tmp_path, monkeypatch):
    weights = _artifact_directory(tmp_path)
    predictor = RecordingPredictor()
    monkeypatch.setenv("HAKAM_WEIGHTS_DIR", str(weights))

    def loader(_directory, _device, thresholds):
        return _loaded(predictor, thresholds)

    with TestClient(create_app(loader, lambda _path: None)) as client:
        response = client.post(
            "/api/analyze",
            files=[
                ("video", ("one.mp4", b"one", "video/mp4")),
                ("video", ("two.mp4", b"two", "video/mp4")),
            ],
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_VIDEO_COUNT"
    assert predictor.seen_paths == []


def test_rejects_unsupported_mime_type_before_save(tmp_path, monkeypatch):
    weights = _artifact_directory(tmp_path)
    uploads = tmp_path / "uploads"
    predictor = RecordingPredictor()
    monkeypatch.setenv("HAKAM_WEIGHTS_DIR", str(weights))
    monkeypatch.setenv("HAKAM_TEMP_DIR", str(uploads))

    def loader(_directory, _device, thresholds):
        return _loaded(predictor, thresholds)

    with TestClient(create_app(loader, lambda _path: None)) as client:
        response = client.post(
            "/api/analyze",
            files=[("video", ("notes.txt", b"not-video", "text/plain"))],
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_VIDEO"
    assert predictor.predict_calls == 0
    assert not uploads.exists()


def test_missing_upload_is_a_sanitised_invalid_request(tmp_path, monkeypatch):
    weights = _artifact_directory(tmp_path)
    predictor = RecordingPredictor()
    monkeypatch.setenv("HAKAM_WEIGHTS_DIR", str(weights))

    def loader(_directory, _device, thresholds):
        return _loaded(predictor, thresholds)

    with TestClient(create_app(loader, lambda _path: None)) as client:
        response = client.post("/api/analyze")

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "INVALID_REQUEST",
            "message": "Exactly one video file is required.",
        }
    }
    assert predictor.predict_calls == 0


def test_unreadable_video_is_sanitised_and_cleaned(tmp_path, monkeypatch):
    weights = _artifact_directory(tmp_path)
    uploads = tmp_path / "uploads"
    predictor = RecordingPredictor()
    monkeypatch.setenv("HAKAM_WEIGHTS_DIR", str(weights))
    monkeypatch.setenv("HAKAM_TEMP_DIR", str(uploads))

    def loader(_directory, _device, thresholds):
        return _loaded(predictor, thresholds)

    def invalid(_path):
        raise InvalidVideoError("private decoder detail")

    with TestClient(create_app(loader, invalid)) as client:
        response = client.post(
            "/api/analyze",
            files=[("video", ("bad.mp4", b"not-video", "video/mp4"))],
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "INVALID_VIDEO",
            "message": "The uploaded video could not be decoded.",
        }
    }
    assert list(uploads.iterdir()) == []


def test_inference_failure_is_sanitised_and_cleaned(tmp_path, monkeypatch):
    weights = _artifact_directory(tmp_path)
    uploads = tmp_path / "uploads"
    predictor = RecordingPredictor(fail=True)
    monkeypatch.setenv("HAKAM_WEIGHTS_DIR", str(weights))
    monkeypatch.setenv("HAKAM_TEMP_DIR", str(uploads))

    def loader(_directory, _device, thresholds):
        return _loaded(predictor, thresholds)

    with TestClient(create_app(loader, lambda _path: None)) as client:
        response = client.post(
            "/api/analyze",
            files=[("video", ("incident.mp4", b"video-bytes", "video/mp4"))],
        )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INFERENCE_FAILED"
    assert list(uploads.iterdir()) == []


def test_model_not_ready_is_inspectable(tmp_path, monkeypatch):
    weights = _artifact_directory(tmp_path)
    monkeypatch.setenv("HAKAM_WEIGHTS_DIR", str(weights))

    def failed_loader(_directory, _device, _thresholds):
        raise ArtifactValidationError("MODEL_LOAD_FAILED", "private load detail")

    with TestClient(create_app(failed_loader, lambda _path: None)) as client:
        readiness = client.get("/api/ready")
        response = client.post(
            "/api/analyze",
            files=[("video", ("incident.mp4", b"video-bytes", "video/mp4"))],
        )

    assert readiness.status_code == 503
    assert readiness.json()["reasonCode"] == "MODEL_LOAD_FAILED"
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "MODEL_NOT_READY"


def test_predictor_is_loaded_once_and_reused_across_requests(tmp_path, monkeypatch):
    weights = _artifact_directory(tmp_path)
    uploads = tmp_path / "uploads"
    predictor = RecordingPredictor()
    load_calls = 0
    monkeypatch.setenv("HAKAM_WEIGHTS_DIR", str(weights))
    monkeypatch.setenv("HAKAM_TEMP_DIR", str(uploads))

    def loader(_directory, _device, thresholds):
        nonlocal load_calls
        load_calls += 1
        return _loaded(predictor, thresholds)

    with TestClient(create_app(loader, lambda _path: None)) as client:
        first = client.post(
            "/api/analyze",
            files=[("video", ("one.mp4", b"video-one", "video/mp4"))],
        )
        second = client.post(
            "/api/analyze",
            files=[("video", ("two.mp4", b"video-two", "video/mp4"))],
        )

    assert first.status_code == second.status_code == 200
    assert load_calls == 1
    assert predictor.predict_calls == 2
    assert list(uploads.iterdir()) == []


def test_ruling_failure_is_sanitised_and_temporary_file_is_cleaned(
    tmp_path, monkeypatch
):
    import backend.pipeline as pipeline_module

    weights = _artifact_directory(tmp_path)
    uploads = tmp_path / "uploads"
    predictor = RecordingPredictor(confidence=0.91)
    monkeypatch.setenv("HAKAM_WEIGHTS_DIR", str(weights))
    monkeypatch.setenv("HAKAM_TEMP_DIR", str(uploads))

    def loader(_directory, _device, thresholds):
        return _loaded(predictor, thresholds)

    def failed_ruling(_contract):
        raise RuntimeError("private ruling implementation detail")

    monkeypatch.setattr(pipeline_module, "_grounded_explanation", failed_ruling)

    with TestClient(create_app(loader, lambda _path: None)) as client:
        response = client.post(
            "/api/analyze",
            files=[("video", ("incident.mp4", b"video-bytes", "video/mp4"))],
        )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "RULING_FAILED",
            "message": "The grounded ruling could not be produced.",
        }
    }
    assert "private" not in response.text
    assert list(uploads.iterdir()) == []

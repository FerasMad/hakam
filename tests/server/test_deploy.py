from fastapi.testclient import TestClient

from scripts import fetch_assets, smoke_test
from server import services
from server.app import app


def test_fetch_downloads_only_missing_files(tmp_path, monkeypatch):
    calls = []

    def fake_download(repo_id, filename, repo_type, token, local_dir):
        calls.append(filename)
        (tmp_path / "weights" / filename).write_text("x")

    def fake_snapshot(repo_id, repo_type, token, local_dir, allow_patterns):
        calls.append("snapshot")
        (tmp_path / "artifacts" / "contracts").mkdir(parents=True)
        (tmp_path / "artifacts" / "contracts" / "test_index.json").write_text("[]")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "thresholds.json").write_text("{}")

    lines = fetch_assets.fetch("u/w", "u/c", "tok", root=tmp_path)
    assert calls == ["final.pt", "snapshot"]
    assert any("present" in line for line in lines)

    calls.clear()
    fetch_assets.fetch("u/w", "u/c", "tok", root=tmp_path)
    assert calls == []                      # second start: nothing to download


def test_smoke_test_passes_against_the_app(monkeypatch, tmp_path):
    monkeypatch.setenv("HAKAM_RETRIEVAL_MODE", "tags")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(services, "CONTRACTS_DIR", tmp_path)
    monkeypatch.setattr(services, "_predictor", object())
    monkeypatch.setattr(services, "get_predictor", lambda: type("P", (), {"device": "cpu"})())

    results = smoke_test.run("http://testserver", [], client=TestClient(app))
    failed = [name for name, ok, _ in results if not ok and not name.startswith(smoke_test.OPTIONAL)]
    assert failed == [], results
    health = TestClient(app).get("/api/health").json()
    assert health["version"] == services.VERSION and health["cases_enabled"] is False


def test_space_readme_declares_docker_port():
    text = (services.ROOT / "deploy" / "space_README.md").read_text(encoding="utf-8")
    assert "sdk: docker" in text and "app_port: 7860" in text
    assert "7860" in (services.ROOT / "Dockerfile").read_text(encoding="utf-8")

"""`python -m mediaindex setup`: what it downloads and skips (downloads mocked; the real run is in docs/progress.md)."""

import mediaindex.desktop as desktop
from mediaindex import creative, detect, edit, setup


def test_setup_downloads_only_what_is_missing(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda repo, **kw: calls.append(repo))
    monkeypatch.setattr(detect, "download", lambda: calls.append("detector"))
    monkeypatch.setattr(edit, "download", lambda: calls.append("sam+lama"))
    monkeypatch.setattr(creative, "download", lambda: calls.append("sd"))
    monkeypatch.setattr(setup, "build_ui", lambda force=False: True)
    monkeypatch.setattr(desktop, "model_cached", lambda: False)
    monkeypatch.setattr(edit, "models_cached", lambda: True)
    assert setup.main(["--no-check"]) == 0
    assert calls == ["google/embeddinggemma-2", "detector"]
    calls.clear()
    monkeypatch.setattr(desktop, "model_cached", lambda: True)
    assert setup.main(["--no-check", "--editing"]) == 0
    assert "sam+lama" not in calls and "already downloaded" in capsys.readouterr().out


def test_setup_reports_a_failed_download(monkeypatch, capsys):
    def offline(repo, **kw):
        raise OSError("no network")

    monkeypatch.setattr("huggingface_hub.snapshot_download", offline)
    monkeypatch.setattr(setup, "build_ui", lambda force=False: True)
    monkeypatch.setattr(desktop, "model_cached", lambda: False)
    assert setup.main(["--no-check"]) == 1
    assert "download failed: no network" in capsys.readouterr().out

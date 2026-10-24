"""The launcher's web UI handling: source hash, download of the matching prebuilt UI, fallback."""
import http.server
import shutil
import sys
import threading
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import launch  # noqa: E402


@pytest.fixture
def ui(tmp_path, monkeypatch):
    web = tmp_path / "web"
    (web / "src").mkdir(parents=True)
    (web / "src" / "App.tsx").write_bytes(b"export const a = 1;\n")
    (web / "index.html").write_bytes(b"<div id=root></div>\n")
    static = tmp_path / "static"
    monkeypatch.setattr(launch, "WEB", web)
    monkeypatch.setattr(launch, "STATIC", static)
    monkeypatch.setattr(launch, "STATIC_INDEX", static / "index.html")
    monkeypatch.setattr(launch, "UI_STAMP", static / ".ui-hash")
    return tmp_path


def test_ui_hash_ignores_line_endings_and_tracks_changes(ui):
    h = launch.ui_hash()
    (launch.WEB / "src" / "App.tsx").write_bytes(b"export const a = 1;\r\n")  # a Windows checkout
    assert launch.ui_hash() == h
    (launch.WEB / "src" / "App.tsx").write_bytes(b"export const a = 2;\n")
    assert launch.ui_hash() != h


def test_ui_hash_is_the_same_on_every_platform(ui):
    """Computed by hand the way CI (Linux) does it: files in case-sensitive path order, LF."""
    import hashlib

    (launch.WEB / "src" / "api.ts").write_bytes(b"x\r\n")  # sorts after App.tsx on Linux, before on Windows
    files = {"index.html": b"<div id=root></div>\n", "src/App.tsx": b"export const a = 1;\n", "src/api.ts": b"x\n"}
    h = hashlib.sha256()
    for name in sorted(files):  # str order: "index.html" < "src/App.tsx" < "src/api.ts"
        h.update(name.encode() + b"\0" + files[name] + b"\0")
    assert launch.ui_hash() == h.hexdigest()[:16]


def test_downloads_the_matching_prebuilt_ui(ui, monkeypatch):
    want = launch.ui_hash()
    served = ui / "served"
    served.mkdir()
    with zipfile.ZipFile(served / f"ui-{want}.zip", "w") as z:
        z.writestr("index.html", "<html>prebuilt</html>")
        z.writestr("assets/app.js", "console.log(1)")

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(served), **k)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Quiet)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(launch, "UI_BASE", f"http://127.0.0.1:{srv.server_address[1]}")
        monkeypatch.setattr(shutil, "which", lambda name: None)  # no Node.js
        launch.ensure_ui()
        assert (launch.STATIC / "index.html").read_text() == "<html>prebuilt</html>"
        assert (launch.STATIC / "assets" / "app.js").exists()
        assert launch.built_ui_hash() == want
        (served / f"ui-{want}.zip").unlink()
        launch.ensure_ui()  # already current: nothing to download
        assert (launch.STATIC / "index.html").exists()

        # sources changed and there's no build for them nor Node.js: the old build is kept
        (launch.WEB / "src" / "App.tsx").write_bytes(b"export const a = 3;\n")
        launch.ensure_ui()
        assert (launch.STATIC / "index.html").read_text() == "<html>prebuilt</html>"
        # ...and without any build at all, it says what to do
        shutil.rmtree(launch.STATIC)
        with pytest.raises(SystemExit, match="Node.js"):
            launch.ensure_ui()
    finally:
        srv.shutdown()

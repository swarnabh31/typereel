"""UI regression tests (Streamlit AppTest). Guards the bug where step 3 forgot which input type was chosen."""
import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
from PIL import Image

import fake_ollama

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _app():
    return AppTest.from_file(os.path.join(ROOT, "ui.py"), default_timeout=180).run()


def _btn(at, text):
    return [b for b in at.button if text in b.label][0]


def test_screenshot_flow_passes_images_and_keeps_brand_edits(tmp_path, monkeypatch):
    srv, url = fake_ollama.start()
    monkeypatch.setenv("OLLAMA_HOST", url)
    monkeypatch.setenv("TYPEREEL_TTS", "fake")
    shots = []
    for n in ("a", "b"):
        p = tmp_path / f"{n}.png"; Image.new("RGB", (1200, 700), (230, 230, 245)).save(p); shots.append(str(p))
    at = _app()
    at.radio[0].set_value("🖥️ App screenshots").run()
    at.session_state["images"] = shots
    at.run()
    at.text_area[0].set_value("Recruitment OS screens JDs. 12 minutes to shortlist.").run()
    _btn(at, "Next: choose").click().run()
    # edit the brand, leave step 2, come back: edit must still be there
    [t for t in at.text_input if t.label.startswith("Brand / product name")][0].set_value("Acme Hiring").run()
    _btn(at, "Next: generate").click().run()
    assert at.session_state["kind_sel"] == "🖥️ App screenshots"          # was reset to 'An idea' before the fix
    _btn(at, "Generate storyboard").click().run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    assert at.session_state["sb"] and at.session_state["sb"]["beats"]
    _btn(at, "← Back").click().run()
    assert [t for t in at.text_input if t.label.startswith("Brand / product name")][0].value == "Acme Hiring"
    srv.shutdown()


def test_missing_images_gives_a_clear_message_not_a_cli_crash():
    at = _app()
    at.radio[0].set_value("🖥️ App screenshots").run()
    at.session_state["images"] = ["/nope/a.png", "/nope/b.png"]
    at.session_state["step"] = 3
    at.run()
    _btn(at, "Generate storyboard").click().run()
    assert any("at least 2 images" in e.value for e in at.error)

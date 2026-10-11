import json, os, shutil, subprocess, sys

import numpy as np
import pytest
from PIL import Image

import fake_ollama
from trcore import audio as A
from trcore import brand as B
from trcore import llm as L
from trcore import planner, storyboard as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAS_FFMPEG = shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------- llm helpers
def test_extract_json_variants():
    assert L.extract_json('{"a": 1}') == {"a": 1}
    assert L.extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert L.extract_json('<think>hmm</think>{"a": 3}') == {"a": 3}
    assert L.extract_json('noise before {"a": 4} noise after') == {"a": 4}
    assert L.extract_json("not json") is None


def test_resolve_model_prefix_and_override():
    inst = ["gemma3:12b-it-q4", "nomic-embed-text:latest"]
    assert L.resolve_model(["gemma3:12b"], inst) == "gemma3:12b-it-q4"
    assert L.resolve_model(["nope"], inst) is None
    assert L.resolve_model([], inst, override="nomic") == "nomic-embed-text:latest"


# ---------------------------------------------------------------- brand
def test_all_presets_validate():
    presets = B.list_presets()
    assert len(presets) >= 6
    for p in presets:
        b = B.load_brand(p["id"])
        assert b["output"]["aspect"] in B.ASPECTS
        assert 3 <= b["voice"]["max_caption_words"] <= 20


def test_validate_clamps_garbage():
    b = B.validate_brand({"visual": {"palette": {"accent": "nope"}, "glow": 9}, "audio": {"bpm": 999, "music": "x"},
                          "motion": {"camera": ["slow_zoom", "bogus"]}, "voice": {"banned": "a, b\nc"}})
    assert b["visual"]["glow"] == 1.0 and b["audio"]["bpm"] == 140 and b["audio"]["music"] == "lofi-ambient"
    assert b["motion"]["camera"] == ["slow_zoom"] and b["voice"]["banned"] == ["a", "b", "c"]
    assert b["visual"]["palette"]["accent"].startswith("#")


def test_brand_roundtrip(tmp_path):
    b = B.load_brand("luxury-noir")
    p = B.save_brand(b, str(tmp_path / "k.json"))
    assert B.load_brand(p) == b


def test_voice_block_contains_rules():
    t = B.voice_block(B.load_brand("engineering-saas"))
    assert "NEVER use" in t and "authoritative" in t and "Limits:" in t


def test_canvas_sizes():
    b = B.load_brand()
    assert B.canvas_for(b, "final")[:2] == (1080, 1920)
    b["output"]["aspect"] = "16:9"
    assert B.canvas_for(b, "final")[:2] == (1920, 1080)
    assert B.canvas_for(b, "preview")[2] == 24


# ---------------------------------------------------------------- storyboard + lint
def _sb(**kw):
    base = {"title": "T", "cta": "C", "images": ["a.png", "b.png"], "beats": [
        {"scene": "ui", "img": 0, "caption": "Screening takes 12 minutes", "focus": "0.1,0.1,0.4,0.3"}]}
    base.update(kw)
    return S.normalize(base)


def test_normalize_repairs_bad_input():
    sb = S.normalize({"images": ["a.png"], "beats": [{"scene": "ui", "img": 9, "caption": "x"},
                                                      {"scene": "kpi", "caption": "y"},
                                                      {"scene": "weird", "caption": "z", "secs": "abc"}]})
    assert [b["scene"] for b in sb["beats"]] == ["line", "line", "line"]
    assert [b["id"] for b in sb["beats"]] == [1, 2, 3]


def test_focus_parsing_and_clamp():
    sb = _sb()
    assert sb["beats"][0]["focus"] == [0.1, 0.1, 0.4, 0.3]
    sb = S.normalize({"images": ["a"], "beats": [{"scene": "ui", "img": 0, "caption": "c", "focus": [0.9, 0.9, 0.9, 0.9]}]})
    x, y, w, h = sb["beats"][0]["focus"]
    assert x + w <= 1.0001 and y + h <= 1.0001


def test_lint_catches_banned_length_and_ungrounded_numbers():
    b = B.load_brand("engineering-saas")
    sb = S.normalize({"beats": [{"scene": "line", "caption": "A revolutionary 98% gain!"},
                                {"scene": "line", "caption": "one two three four five six seven eight nine ten eleven"}]})
    issues = S.lint(sb, b, facts={"claims": ["12 minutes"]})
    text = " | ".join(issues)
    assert "revolutionary" in text and "98%" in text and "words" in text and '"!"' in text


def test_lint_passes_clean_grounded_copy():
    b = B.load_brand("engineering-saas")
    sb = S.normalize({"beats": [{"scene": "line", "caption": "Shortlist in 12 minutes"}]})
    assert S.lint(sb, b, facts={"metrics": [{"value": "12", "label": "minutes"}]}) == []


def test_durations_scale_with_text_and_pace_and_vo():
    b = B.load_brand("engineering-saas")
    short = {"scene": "line", "caption": "Hi there", "sub": ""}
    long_ = {"scene": "line", "caption": "A much longer caption that needs more reading time", "sub": "and a sub line too"}
    assert S.beat_secs(long_, b) > S.beat_secs(short, b)
    assert S.beat_secs(short, b, vo_dur=6.0) >= 6.8
    b["motion"]["pace"] = "snappy"
    assert S.beat_secs(long_, b) <= S.beat_secs(long_, B.load_brand("engineering-saas"))


def test_srt_format():
    b = B.load_brand()
    segs = S.build_segments(_sb(), b)
    srt = S.to_srt(segs)
    assert srt.startswith("1\n00:00:00,200 --> ") and "-->" in srt


# ---------------------------------------------------------------- audio
def test_music_builds_and_is_normalised():
    m = A.synth_music(20, "lofi-ambient", 90, True)
    assert m.shape == (20 * A.SR, 2) and np.abs(m).max() <= 0.81
    early = np.sqrt((m[: 4 * A.SR] ** 2).mean()); late = np.sqrt((m[-9 * A.SR:-4 * A.SR] ** 2).mean())
    assert late > early
    assert not np.any(A.synth_music(5, "none"))


def test_ducking_lowers_music_under_voice():
    m = np.ones((10 * A.SR, 2), dtype=np.float32) * 0.5
    vo = np.zeros(10 * A.SR, dtype=np.float32); vo[3 * A.SR:6 * A.SR] = 0.3 * np.sin(np.arange(3 * A.SR) * 0.07)
    d = A.duck(m, vo, -14)
    assert d[int(4.5 * A.SR), 0] < 0.15 and d[int(9 * A.SR), 0] > 0.49


def test_fake_tts_duration_tracks_word_count(tmp_path):
    d1 = A.synth_vo("one two three", str(tmp_path / "a.wav"))
    d2 = A.synth_vo("one two three four five six seven eight", str(tmp_path / "b.wav"))
    assert d2 > d1 > 0.5


# ---------------------------------------------------------------- planner with a fake Ollama
@pytest.fixture()
def fake_llm(monkeypatch, tmp_path):
    srv, url = fake_ollama.start()
    monkeypatch.setattr(L, "OLLAMA", url)
    monkeypatch.setattr("trcore.util.STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr("trcore.planner.STATE_DIR", str(tmp_path / "state"))
    fake_ollama.State.director_calls = fake_ollama.State.revise_calls = 0
    yield url
    srv.shutdown()


def _args(argv):
    import typereel
    return typereel.build_parser().parse_args(argv)


def _shot(tmp_path, name="s.png", size=(1600, 900), color=(240, 240, 250)):
    p = tmp_path / name
    Image.new("RGB", size, color).save(p)
    return str(p)


def test_appdemo_plan_lint_revise_and_regions(fake_llm, tmp_path):
    a = _args(["--mode", "appdemo", "--brief", "Recruitment OS screens JDs. 12 minutes to shortlist.",
               "--images", _shot(tmp_path, "a.png"), _shot(tmp_path, "b.png"), "--no-memory"])
    brand = B.load_brand("engineering-saas")
    models = planner.resolve_models()
    assert models.writer and models.vision
    sb = planner.plan(a, brand, models, planner.Memory(False, None))
    assert fake_ollama.State.revise_calls == 1                      # hype draft was repaired once
    assert sb["lint"] == []
    caps = [b["caption"] for b in sb["beats"]]
    assert not any("revolutionary" in c.lower() for c in caps)
    assert len([b for b in sb["beats"] if b["scene"] == "ui"]) == 2  # duplicate region dropped
    assert all(b["focus"] for b in sb["beats"] if b["scene"] == "ui")
    assert any(b["scene"] == "kpi" and b["kpi"]["value"] == "12 min" for b in sb["beats"])
    assert sb["platform_captions"]["linkedin"]
    assert "FACTS (the only claims you may make)" in "".join(fake_ollama.State.prompts)


def test_vision_cards_are_cached(fake_llm, tmp_path):
    p = _shot(tmp_path)
    models = planner.resolve_models()
    n0 = sum(1 for x in fake_ollama.State.prompts)
    planner.describe(models.vision, p, "appdemo")
    n1 = len(fake_ollama.State.prompts)
    planner.describe(models.vision, p, "appdemo")
    assert len(fake_ollama.State.prompts) == n1 and n1 == n0 + 1


def test_text_mode_plan_and_regen(fake_llm):
    a = _args(["calm mind", "--mode", "quotes", "--lines", "6", "--no-memory"])
    brand = B.load_brand("bold-social")
    models = planner.resolve_models()
    sb = planner.plan(a, brand, models, planner.Memory(False, None))
    assert len(sb["beats"]) == 8 and sb["beats"][0]["scene"] == "line"
    sb2 = planner.regen_beat(sb, 0, "tighter", brand, models)
    assert sb2["beats"][0]["caption"].startswith("Ranked shortlist")


def test_offline_fallbacks_without_ollama(monkeypatch, tmp_path):
    monkeypatch.setattr(L, "OLLAMA", "http://127.0.0.1:9")          # nothing listens here
    models = planner.resolve_models()
    assert models.writer is None
    brand = B.load_brand()
    sb = planner.plan(_args(["demo topic"]), brand, models, planner.Memory(False, None))
    assert len(sb["beats"]) >= 4
    a = _args(["--mode", "appdemo", "--images", _shot(tmp_path, "x.png"), _shot(tmp_path, "y.png")])
    sb = planner.plan(a, brand, models, planner.Memory(False, None))
    assert [b["scene"] for b in sb["beats"]] == ["ui", "ui"]


# ---------------------------------------------------------------- render smoke tests
@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
@pytest.mark.parametrize("brand_id,aspect", [("engineering-saas", "9:16"), ("playful-consumer", "16:9"), ("editorial-minimal", "1:1")])
def test_render_smoke(tmp_path, brand_id, aspect):
    from trcore import pipeline
    brand = B.load_brand(brand_id)
    brand["output"]["aspect"] = aspect
    brand["audio"]["voiceover"] = True
    img = _shot(tmp_path, "ui.png", (1600, 700))
    sb = {"title": "Smoke test", "cta": "Done", "images": [img], "beats": [
        {"scene": "ui", "img": 0, "kicker": "STEP 1", "caption": "Zoom into the table", "sub": "with a cursor", "focus": [0.1, 0.2, 0.4, 0.4]},
        {"scene": "kpi", "kpi": {"value": "98.5%", "label": "accuracy"}, "caption": "A KPI beat"},
        {"scene": "line", "caption": "A text beat"}]}
    a = _args(["--fast", "--out", str(tmp_path)])
    res = pipeline.render_storyboard(sb, brand, a)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,width,height", "-of", "json", res["video"]],
                         capture_output=True, text=True)
    st = {s["codec_name"]: s for s in json.loads(out.stdout)["streams"]}
    assert "h264" in st and "aac" in st
    w, h = st["h264"]["width"], st["h264"]["height"]
    assert (w > h) == (aspect == "16:9") and (w == h) == (aspect == "1:1")
    assert os.path.getsize(res["video"]) > 20_000
    assert os.path.exists(res["srt"]) and os.path.exists(res["caption"]) and json.load(open(res["json"]))["brand"]["id"]


def test_missing_image_falls_back_to_text_scene(tmp_path):
    from trcore import render as R
    from trcore.storyboard import build_segments
    brand = B.load_brand()
    sb = S.normalize({"title": "t", "images": ["/nope.png"], "beats": [{"scene": "ui", "img": 0, "caption": "x"}]})
    W = R.World(brand, 360, 640, 24)
    scenes = R.build_scenes(W, build_segments(sb, brand), {})
    assert type(scenes[1]).__name__ == "LineScene"
    assert scenes[1].frame(5).size == (360, 640)


def test_scene_frames_are_deterministic_and_thread_safe(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from trcore import render as R
    from trcore.storyboard import build_segments
    brand = B.load_brand()
    img = Image.open(_shot(tmp_path, "z.png", (1200, 600)))
    sb = S.normalize({"title": "t", "images": ["z"], "beats": [{"scene": "ui", "img": 0, "caption": "cap", "focus": [0.2, 0.2, 0.3, 0.3]}]})
    W = R.World(brand, 360, 640, 24)
    sc = R.build_scenes(W, build_segments(sb, brand), {0: img.convert("RGB")})[1]
    ref = [sc.frame(k).tobytes() for k in range(0, 40, 8)]
    with ThreadPoolExecutor(4) as ex:
        par = list(ex.map(lambda k: sc.frame(k).tobytes(), range(0, 40, 8)))
    assert ref == par


def test_cli_offline_demo_end_to_end(tmp_path):
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    env = dict(os.environ, OLLAMA_HOST="http://127.0.0.1:9", TYPEREEL_TTS="fake")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "typereel.py"), "offline demo", "--fast", "--no-vo",
                        "--out", str(tmp_path)], capture_output=True, text=True, env=env, cwd=ROOT, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert any(f.endswith(".mp4") for f in os.listdir(tmp_path))

"""Brand Kit: one JSON file that defines a video's voice, look, motion and audio.

The UI edits it, the CLI reads it (--brand <preset-id | path.json>), and the prompt
builder turns its `voice` section into the instruction block every LLM stage receives.
"""
import copy, glob, json, os, re

from .util import ROOT, hex_to_rgb, rgb_to_hex, clamp

BRAND_DIR = os.path.join(ROOT, "brands")
ASPECTS = {"9:16": (1080, 1920), "16:9": (1920, 1080), "1:1": (1080, 1080), "4:5": (1080, 1350)}
PLATFORM_ASPECT = {"Reels / Shorts / TikTok": "9:16", "LinkedIn / YouTube / Web": "16:9",
                   "Feed square": "1:1", "Instagram portrait": "4:5"}

DEFAULT_BRAND = {
    "id": "custom",
    "name": "Custom",
    "description": "",
    "identity": {"name": "", "tagline": "", "website": "", "cta": "Learn more", "watermark": ""},
    "voice": {
        "tone": ["clear", "confident", "calm"],
        "person": "second person, present tense",
        "banned": ["revolutionary", "game-changing", "unlock", "supercharge", "seamless", "!"],
        "style_rules": ["Lead with the concrete outcome", "One claim per line", "Prefer numbers to adjectives"],
        "examples": [],
        "max_caption_words": 8,
        "max_sub_words": 14,
        "language": "English",
    },
    "visual": {
        "palette": {"bg": ["#0A0C22", "#1E1650"], "bg2": ["#04061A", "#38186A"],
                    "accent": "#7C5CFF", "text": "#F5F3FF"},
        "background": "grid",          # grid | gradient | ai | solid
        "grid": {"size": 64, "opacity": 0.10},
        "glow": 0.30,
        "card": {"style": "glass", "radius": 28, "glow": 0.35, "chrome": "auto"},   # glass | solid | bare
        "caption": {"style": "glass", "opacity": 120, "bg": "#0B0D1E", "size": 1.0, "position": "lower"},
        "fonts": {"heading": "Inter", "body": "Inter", "numerals": "Inter"},
        "grade": {"vignette": 0.45, "tint": 0.10},
        "art_style": "abstract dark navy and indigo gradient, soft purple volumetric glow, subtle grid, minimal",
    },
    "motion": {
        "camera": ["slow_zoom", "ui_pan", "parallax"],
        "cursor": True,
        "spotlight": True,
        "transition": "dissolve",       # dissolve | slide
        "transition_sec": 0.45,
        "pace": "calm",                 # calm | standard | snappy
    },
    "audio": {
        "music": "lofi-ambient",        # lofi-ambient | minimal-pulse | cinematic-drone | none
        "bpm": 90,
        "build": True,
        "music_volume": 0.55,
        "voiceover": False,
        "voice": "neutral",
        "voice_speed": 0.95,
        "duck_db": -14,
        "loudness": -16,
    },
    "output": {"aspect": "9:16", "quality": "final"},
}

# the eight original TypeReel palettes, expressed as brand palettes
LEGACY_PALETTES = {
    "midnight": {"bg": ["#0A0C22", "#1E1040"], "bg2": ["#040614", "#381860"], "accent": "#7C5CFF", "text": "#F5F3FF"},
    "ember": {"bg": ["#28080A", "#6E1A10"], "bg2": ["#180408", "#A03614"], "accent": "#FF9240", "text": "#FFF4EC"},
    "ocean": {"bg": ["#041A2C", "#0A4268"], "bg2": ["#02101E", "#126896"], "accent": "#54D2FF", "text": "#EEF9FF"},
    "forest": {"bg": ["#081C12", "#16482C"], "bg2": ["#04120C", "#246E40"], "accent": "#7AFFA8", "text": "#EEFFF4"},
    "noir": {"bg": ["#0C0C0E", "#26262C"], "bg2": ["#060608", "#404048"], "accent": "#FFFFFF", "text": "#FAFAFA"},
    "sunset": {"bg": ["#42103C", "#A03228"], "bg2": ["#280A2C", "#DC6030"], "accent": "#FFC878", "text": "#FFF8F0"},
    "ultraviolet": {"bg": ["#1A0634", "#541078"], "bg2": ["#0E0420", "#8224B4"], "accent": "#F078FF", "text": "#FAF0FF"},
    "paper": {"bg": ["#F6F2E8", "#E2DAC8"], "bg2": ["#FAF7F0", "#D2C8B2"], "accent": "#1E1E22", "text": "#1C1A18"},
}


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _hex(v, fallback):
    return rgb_to_hex(hex_to_rgb(v)) if hex_to_rgb(str(v)) else fallback


def _list(v, fallback):
    if isinstance(v, str):
        v = [x.strip() for x in re.split(r"[,\n]", v) if x.strip()]
    return [str(x).strip() for x in v if str(x).strip()] if isinstance(v, (list, tuple)) else list(fallback)


def validate_brand(b):
    """Merge onto defaults and coerce every field into a safe range/type."""
    b = deep_merge(DEFAULT_BRAND, b if isinstance(b, dict) else {})
    D = DEFAULT_BRAND
    v, vis, mo, au, out = b["voice"], b["visual"], b["motion"], b["audio"], b["output"]
    v["tone"] = _list(v.get("tone"), D["voice"]["tone"])
    v["banned"] = _list(v.get("banned"), [])
    v["style_rules"] = _list(v.get("style_rules"), [])
    v["examples"] = _list(v.get("examples"), [])
    v["max_caption_words"] = int(clamp(int(v.get("max_caption_words") or 8), 3, 20))
    v["max_sub_words"] = int(clamp(int(v.get("max_sub_words") or 14), 4, 30))
    pal = vis["palette"]
    for key in ("bg", "bg2"):
        pair = pal.get(key)
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
            pair = D["visual"]["palette"][key]
        pal[key] = [_hex(pair[0], D["visual"]["palette"][key][0]), _hex(pair[1], D["visual"]["palette"][key][1])]
    pal["accent"] = _hex(pal.get("accent"), D["visual"]["palette"]["accent"])
    pal["text"] = _hex(pal.get("text"), D["visual"]["palette"]["text"])
    if vis.get("background") not in ("grid", "gradient", "ai", "solid"):
        vis["background"] = "grid"
    vis["glow"] = float(clamp(float(vis.get("glow", 0.3)), 0, 1))
    vis["grid"]["size"] = int(clamp(int(vis["grid"].get("size", 64)), 24, 160))
    vis["grid"]["opacity"] = float(clamp(float(vis["grid"].get("opacity", 0.1)), 0, 0.4))
    c = vis["card"]
    c["style"] = c.get("style") if c.get("style") in ("glass", "solid", "bare") else "glass"
    c["radius"] = int(clamp(int(c.get("radius", 28)), 0, 80))
    c["glow"] = float(clamp(float(c.get("glow", 0.35)), 0, 1))
    c["chrome"] = c.get("chrome") if c.get("chrome") in ("auto", "browser", "none") else "auto"
    cp = vis["caption"]
    cp["style"] = cp.get("style") if cp.get("style") in ("glass", "solid", "none") else "glass"
    cp["opacity"] = int(clamp(int(cp.get("opacity", 120)), 0, 255))
    cp["bg"] = _hex(cp.get("bg"), "#0B0D1E")
    cp["size"] = float(clamp(float(cp.get("size", 1.0)), 0.6, 1.6))
    cp["position"] = cp.get("position") if cp.get("position") in ("lower", "center") else "lower"
    vis["grade"]["vignette"] = float(clamp(float(vis["grade"].get("vignette", 0.45)), 0, 1))
    vis["grade"]["tint"] = float(clamp(float(vis["grade"].get("tint", 0.1)), 0, 0.5))
    mo["camera"] = [x for x in _list(mo.get("camera"), []) if x in ("slow_zoom", "ui_pan", "parallax")]
    mo["cursor"] = bool(mo.get("cursor"))
    mo["spotlight"] = bool(mo.get("spotlight"))
    mo["transition"] = mo.get("transition") if mo.get("transition") in ("dissolve", "slide") else "dissolve"
    mo["transition_sec"] = float(clamp(float(mo.get("transition_sec", 0.45)), 0.1, 1.2))
    mo["pace"] = mo.get("pace") if mo.get("pace") in ("calm", "standard", "snappy") else "calm"
    au["music"] = au.get("music") if au.get("music") in ("lofi-ambient", "minimal-pulse", "cinematic-drone", "none") else "lofi-ambient"
    au["bpm"] = int(clamp(int(au.get("bpm", 90)), 60, 140))
    au["build"] = bool(au.get("build"))
    au["music_volume"] = float(clamp(float(au.get("music_volume", 0.55)), 0, 1))
    au["voiceover"] = bool(au.get("voiceover"))
    au["voice_speed"] = float(clamp(float(au.get("voice_speed", 0.95)), 0.6, 1.4))
    au["duck_db"] = float(clamp(float(au.get("duck_db", -14)), -30, 0))
    au["loudness"] = float(clamp(float(au.get("loudness", -16)), -24, -9))
    if out.get("aspect") not in ASPECTS:
        out["aspect"] = "9:16"
    if out.get("quality") not in ("preview", "final"):
        out["quality"] = "final"
    return b


def list_presets():
    items = []
    for p in sorted(glob.glob(os.path.join(BRAND_DIR, "*.json"))):
        try:
            d = json.load(open(p, encoding="utf-8"))
            items.append({"id": d.get("id") or os.path.splitext(os.path.basename(p))[0],
                          "name": d.get("name", os.path.basename(p)), "description": d.get("description", ""),
                          "path": p})
        except Exception:
            continue
    return items


def load_brand(spec=None):
    """spec: None | preset id | path to a .json brand kit | dict."""
    if isinstance(spec, dict):
        return validate_brand(spec)
    if not spec:
        spec = "engineering-saas"
    if os.path.isfile(str(spec)):
        return validate_brand(json.load(open(spec, encoding="utf-8")))
    for p in list_presets():
        if p["id"] == spec or p["name"].lower() == str(spec).lower():
            return validate_brand(json.load(open(p["path"], encoding="utf-8")))
    return validate_brand({})


def save_brand(b, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(validate_brand(b), f, indent=2, ensure_ascii=False)
    return path


def apply_palette_name(brand, name):
    if name in LEGACY_PALETTES:
        brand["visual"]["palette"] = copy.deepcopy(LEGACY_PALETTES[name])
    return brand


def palette_rgb(brand):
    """Runtime palette: RGB tuples keyed a/b/a2/b2/accent/text (+ derived muted)."""
    p = brand["visual"]["palette"]
    f = lambda h: hex_to_rgb(h)
    text = f(p["text"])
    a, b = f(p["bg"][0]), f(p["bg"][1])
    light = sum(a) / 3 > 140
    muted = tuple(int(text[i] * 0.72 + (a[i] if not light else 0) * 0.28 + (60 if light else 0) * 0) for i in range(3))
    return {"a": a, "b": b, "a2": f(p["bg2"][0]), "b2": f(p["bg2"][1]),
            "accent": f(p["accent"]), "text": text, "muted": muted, "light": light}


def voice_block(brand):
    """The text block injected into every LLM prompt so tone is enforced, not hoped for."""
    v = brand["voice"]
    ident = brand["identity"]
    lines = []
    if ident.get("name"):
        lines.append(f"Brand: {ident['name']}" + (f" — {ident['tagline']}" if ident.get("tagline") else ""))
    lines.append(f"Tone: {', '.join(v['tone'])}.")
    lines.append(f"Narration: {v['person']}. Language: {v.get('language', 'English')}.")
    if v["banned"]:
        lines.append("NEVER use these words/characters: " + ", ".join(f'"{w}"' for w in v["banned"]) +
                     " — nor hype, superlatives, exclamation marks or stock-footage clichés.")
    if v["style_rules"]:
        lines.append("Style rules:\n" + "\n".join(f"  - {r}" for r in v["style_rules"]))
    if v["examples"]:
        lines.append("Reference lines (match this register, do NOT copy):\n" +
                     "\n".join(f'  - "{e}"' for e in v["examples"][:6]))
    lines.append(f"Limits: caption <= {v['max_caption_words']} words, sub-line <= {v['max_sub_words']} words.")
    return "\n".join(lines)


def canvas_for(brand, quality=None):
    w, h = ASPECTS[brand["output"]["aspect"]]
    q = quality or brand["output"]["quality"]
    if q == "preview":
        s = 2 / 3
        return int(round(w * s / 2) * 2), int(round(h * s / 2) * 2), 24
    return w, h, 30

"""Storyboard = the editable plan (JSON). Planner writes it, the UI edits it, the renderer reads it.

beat = {id, scene: ui|kpi|photo|line|clip, img, kicker, caption, sub, vo, focus:[x,y,w,h]|None,
        kpi:{value,label}|None, secs|None, clip:[start,end]|None}
"""
import json, re

from .util import clamp, word_count, mmss

SCENES = ("ui", "kpi", "photo", "line", "clip")
PACE = {"calm": 1.15, "standard": 1.0, "snappy": 0.82}
TITLE_SECS, CTA_SECS = 2.0, 2.4


def clean_line(s):
    return re.sub(r"\s+", " ", str(s if s is not None else "")).strip().strip('"').strip()


def _focus(v):
    try:
        if isinstance(v, str):
            v = [float(x) for x in re.split(r"[,\s]+", v.strip()) if x]
        x, y, w, h = [float(a) for a in v][:4]
        x, y = clamp(x, 0, 1), clamp(y, 0, 1)
        w, h = clamp(w, 0.04, 1 - x), clamp(h, 0.04, 1 - y)
        return [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]
    except Exception:
        return None


def normalize(sb):
    """Coerce a storyboard (possibly hand-edited) into a safe, complete structure."""
    sb = dict(sb or {})
    sb.setdefault("version", 2)
    sb["title"] = clean_line(sb.get("title", ""))[:90]
    sb["cta"] = clean_line(sb.get("cta", ""))[:70]
    sb["hashtags"] = [re.sub(r"^#", "", str(h)).strip() for h in (sb.get("hashtags") or []) if str(h).strip()][:10]
    sb["description"] = clean_line(sb.get("description", ""))[:320]
    sb["art"] = clean_line(sb.get("art", ""))[:240]
    sb["images"] = [str(p) for p in (sb.get("images") or [])]
    sb.setdefault("facts", None)
    sb.setdefault("platform_captions", {})
    sb.setdefault("source", None)
    sb.setdefault("mode", "quotes")
    beats = []
    for i, b in enumerate(sb.get("beats") or []):
        scene = b.get("scene") if b.get("scene") in SCENES else "line"
        img = b.get("img")
        img = int(img) if isinstance(img, (int, float)) and not isinstance(img, bool) and 0 <= int(img) < max(len(sb["images"]), 1) and sb["images"] else None
        if scene in ("ui", "photo") and img is None:
            scene = "line"
        kpi = b.get("kpi")
        if isinstance(kpi, str):
            parts = [p.strip() for p in kpi.split("|", 1)]
            kpi = {"value": parts[0], "label": parts[1] if len(parts) > 1 else ""} if parts[0] else None
        if scene == "kpi" and not (isinstance(kpi, dict) and clean_line(kpi.get("value"))):
            scene = "line"; kpi = None
        secs = b.get("secs")
        try:
            secs = float(secs) if secs not in (None, "", 0) else None
        except (TypeError, ValueError):
            secs = None
        clip = b.get("clip")
        clip = [float(clip[0]), float(clip[1])] if isinstance(clip, (list, tuple)) and len(clip) == 2 else None
        beats.append({"id": i + 1, "scene": scene, "img": img,
                      "kicker": clean_line(b.get("kicker", ""))[:40],
                      "caption": clean_line(b.get("caption", ""))[:200],
                      "sub": clean_line(b.get("sub", ""))[:200],
                      "vo": clean_line(b.get("vo", ""))[:300],
                      "focus": _focus(b.get("focus")) if scene == "ui" else None,
                      "kpi": ({"value": clean_line(kpi["value"])[:24], "label": clean_line(kpi.get("label", ""))[:60]}
                              if scene == "kpi" else None),
                      "secs": clamp(secs, 1.0, 20.0) if secs else None,
                      "clip": clip})
    sb["beats"] = beats
    return sb


def beat_secs(beat, brand, vo_dur=None):
    """Reading-time based duration, scaled by brand pace. VO length wins when longer."""
    if beat.get("clip"):
        return clamp(beat["clip"][1] - beat["clip"][0], 1.5, 12.0)
    if beat.get("secs"):
        base = float(beat["secs"])
    else:
        words = word_count(beat.get("caption", "")) + 0.6 * word_count(beat.get("sub", ""))
        base = 1.3 + words / 3.0
        base *= PACE.get(brand["motion"]["pace"], 1.0)
        floor = {"ui": 3.4, "kpi": 3.2, "photo": 2.6}.get(beat["scene"], 2.0)
        base = clamp(max(base, floor), 2.0, 7.0)
    if vo_dur:
        base = max(base, vo_dur + 0.35 + 0.5)
    return round(base, 2)


def build_segments(sb, brand, vo_durs=None):
    """Storyboard -> ordered render segments (title, beats..., cta) with durations."""
    vo_durs = vo_durs or {}
    segs = [{"kind": "title", "text": sb["title"], "kicker": brand["identity"].get("name", ""),
             "secs": TITLE_SECS * PACE.get(brand["motion"]["pace"], 1.0) + (0.4 if brand["motion"]["pace"] == "calm" else 0)}]
    for b in sb["beats"]:
        segs.append({"kind": b["scene"], "text": b["caption"], "sub": b["sub"], "kicker": b["kicker"],
                     "img": b["img"], "focus": b["focus"], "kpi": b["kpi"], "clip": b["clip"],
                     "beat": b["id"], "vo": b["vo"], "secs": beat_secs(b, brand, vo_durs.get(b["id"]))})
    segs.append({"kind": "cta", "text": sb["cta"] or brand["identity"].get("cta", ""),
                 "sub": brand["identity"].get("website", ""), "kicker": brand["identity"].get("name", ""),
                 "secs": CTA_SECS})
    return segs


def timeline(segs):
    t, out = 0.0, []
    for s in segs:
        out.append((t, t + s["secs"]))
        t += s["secs"]
    return out


def to_srt(segs, use_vo=False):
    def ts(x):
        ms = int(round(x * 1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    rows, n = [], 1
    for s, (a, b) in zip(segs, timeline(segs)):
        text = (s.get("vo") if use_vo and s.get("vo") else s.get("text")) or ""
        if not text.strip():
            continue
        rows.append(f"{n}\n{ts(a + 0.2)} --> {ts(b - 0.15)}\n{text.strip()}\n")
        n += 1
    return "\n".join(rows)


# ------------------------------------------------------------------ quality gate (deterministic)
_NUM_RE = re.compile(r"\d[\d,.]*\s?(?:%|k|m|x|×|ms|s|min|h)?", re.I)


def _norm_num(s):
    return re.sub(r"[^\d.]", "", s.replace(",", ""))


def known_numbers(facts, cards):
    pool = json.dumps(facts or {}, ensure_ascii=False)
    for c in cards or []:
        pool += " " + json.dumps(c.get("numbers", []), ensure_ascii=False) + " " + " ".join(c.get("visible_text", []))
    return {_norm_num(m.group(0)) for m in _NUM_RE.finditer(pool) if _norm_num(m.group(0))}


def lint(sb, brand, facts=None, cards=None, past_repeats=None, check_numbers=True):
    """Return a list of human-readable issues. Empty list = pass."""
    v = brand["voice"]
    issues = []
    banned = [w for w in v["banned"] if w]
    known = known_numbers(facts, cards) if (facts or cards) else None
    openings = {}
    for b in sb.get("beats", []):
        cap, sub, i = b.get("caption", ""), b.get("sub", ""), b["id"]
        if word_count(cap) > v["max_caption_words"] + 1:
            issues.append(f"beat {i}: caption has {word_count(cap)} words (max {v['max_caption_words']}) — shorten it")
        if word_count(sub) > v["max_sub_words"] + 2:
            issues.append(f"beat {i}: sub-line has {word_count(sub)} words (max {v['max_sub_words']}) — shorten it")
        if not cap.strip():
            issues.append(f"beat {i}: caption is empty")
        blob = f"{cap} {sub} {b.get('vo', '')} {sb.get('title', '')}"
        for w in banned:
            hit = (w in blob) if not w.replace(" ", "").isalnum() else re.search(rf"\b{re.escape(w)}\b", blob, re.I)
            if hit:
                issues.append(f'beat {i}: contains banned term "{w}" — rephrase without it')
        if check_numbers and known is not None:
            for text in (cap, sub, (b.get("kpi") or {}).get("value", "")):
                for m in _NUM_RE.finditer(text or ""):
                    n = _norm_num(m.group(0))
                    if n and n not in known and not re.fullmatch(r"(step\s*)?\d", n):
                        issues.append(f'beat {i}: number "{m.group(0).strip()}" is not in the provided facts — remove or replace it')
        key = " ".join(cap.lower().split()[:2])
        if key:
            openings.setdefault(key, []).append(i)
    for key, ids in openings.items():
        if len(ids) >= 3 and len(key.split()) >= 2:
            issues.append(f'beats {ids}: all start with "{key}" — vary the openings')
    for i, txt in (past_repeats or []):
        issues.append(f'beat {i}: nearly identical to an earlier video ("{txt[:40]}") — write something new')
    return issues[:14]


# ------------------------------------------------------------------ offline fallbacks (no LLM available)
DEMO_LINES = ["Most people never start.", "You don't need permission.", "Small reps, every single day.",
              "Boredom is where mastery lives.", "Nobody is coming to save you.", "That's the whole advantage."]


def demo_storyboard(topic, n=6):
    return normalize({"mode": "quotes", "title": topic.strip().title()[:60] or "Untitled", "cta": "Follow for more",
                      "hashtags": ["motivation", "discipline"], "description": "A local reel made with TypeReel.",
                      "art": "", "beats": [{"scene": "line", "kicker": f"{i + 1:02d} / {min(n, len(DEMO_LINES))}", "caption": ln}
                                           for i, ln in enumerate(DEMO_LINES[:n])]})


def blind_screens_storyboard(paths, topic, brand):
    import os
    beats = []
    for i, p in enumerate(paths):
        name = re.sub(r"[_\-]+", " ", os.path.splitext(os.path.basename(p))[0]).strip().title()
        beats.append({"scene": "ui", "img": i, "kicker": f"STEP {i + 1}", "caption": name[:48], "sub": "", "vo": name})
    return normalize({"mode": "appdemo", "title": (topic or "Product tour")[:60], "images": paths, "beats": beats,
                      "cta": brand["identity"].get("cta", "Learn more"), "hashtags": [], "description": topic or ""})

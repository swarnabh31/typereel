"""Planner: turns user input + Brand Kit into a validated Storyboard using staged, cached LLM calls.

Stages:  facts -> screen/photo cards (vision, cached) -> director (schema-constrained)
         -> lint -> one targeted revision -> packager (platform captions)
Every stage degrades to an offline fallback, so the app never dead-ends without Ollama.
"""
import base64, glob, hashlib, io, json, os, re, sqlite3
from dataclasses import dataclass
from datetime import datetime

from . import prompts as P
from .llm import (ask_json, cosine, embed, list_models, llm, ollama_ok, resolve_model, vision_capable,
                  WRITER_PREFS, EDITOR_PREFS, EMBED_PREFS, VISION_PREFS)
from .storyboard import (blind_screens_storyboard, clean_line, demo_storyboard, lint, normalize)
from .util import STATE_DIR, clamp, log, mmss, state_path


@dataclass
class Models:
    writer: str = None
    vision: str = None
    editor: str = None
    embedder: str = None


def resolve_models(writer=None, vision=None):
    installed = list_models() if ollama_ok() else []
    m = Models(resolve_model(WRITER_PREFS, installed, writer), resolve_model(VISION_PREFS, installed, vision),
               resolve_model(EDITOR_PREFS, installed), resolve_model(EMBED_PREFS, installed))
    if m.vision and not vision_capable(m.vision):
        m.vision = None
    return m


# ------------------------------------------------------------------ memory (repeat detection that acts)
class Memory:
    def __init__(self, enabled, model):
        self.model = model
        self.enabled = bool(enabled and model)
        self.rows = []
        if not self.enabled:
            return
        self.db = sqlite3.connect(state_path("memory.db"))
        self.db.execute("CREATE TABLE IF NOT EXISTS lines (text TEXT, emb TEXT, created TEXT)")
        self.db.commit()
        for text, emb in self.db.execute("SELECT text, emb FROM lines ORDER BY created DESC LIMIT 2000"):
            try:
                self.rows.append((text, json.loads(emb)))
            except Exception:
                continue
        if self.rows:
            log("memory", f"{len(self.rows)} past lines loaded — repeats get flagged and rewritten")

    def repeats(self, beats, thr=0.92):
        """[(beat_id, old_text)] for beats nearly identical to something from an earlier video."""
        hits = []
        if not self.enabled or not self.rows:
            return hits
        for b in beats:
            try:
                v = embed(self.model, b["caption"].lower())
            except Exception:
                break
            for old, ev in self.rows:
                if v and len(ev) == len(v) and cosine(v, ev) > thr:
                    hits.append((b["id"], old)); break
        return hits

    def remember(self, beats):
        if not self.enabled:
            return
        for b in beats:
            try:
                v = embed(self.model, b["caption"].lower())
                if v:
                    self.db.execute("INSERT INTO lines VALUES (?,?,?)", (b["caption"].lower(), json.dumps(v), datetime.now().isoformat()))
            except Exception:
                break
        self.db.commit()


# ------------------------------------------------------------------ vision cards (cached by image hash)
def resolve_images(args, cap=14):
    paths = []
    for a in args or []:
        if os.path.isdir(a):
            for ext in ("png", "jpg", "jpeg", "webp"):
                paths += glob.glob(os.path.join(a, f"*.{ext}")) + glob.glob(os.path.join(a, f"*.{ext.upper()}"))
            paths = sorted(set(paths))
        elif os.path.exists(a):
            paths.append(a)
    return [os.path.abspath(p) for p in paths[:cap]]


def load_b64(path, max_px=1280):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_px, max_px))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def _sha(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:20]


def _clean_regions(regs):
    out = []
    for r in regs or []:
        try:
            x, y, w, h = [float(r[k]) for k in ("x", "y", "w", "h")]
        except Exception:
            continue
        if max(x, y, w, h) > 1.5:             # model answered in percent
            x, y, w, h = x / 100, y / 100, w / 100, h / 100
        x, y = clamp(x, 0, 0.95), clamp(y, 0, 0.95)
        w, h = clamp(w, 0.05, 1 - x), clamp(h, 0.05, 1 - y)
        if w * h < 0.0025 or w * h > 0.85:
            continue
        out.append({"label": clean_line(r.get("label", "region"))[:40], "x": round(x, 3), "y": round(y, 3),
                    "w": round(w, 3), "h": round(h, 3)})
    return out[:3]


def describe(vision, path, kind):
    """One vision call per image, cached on disk by content hash."""
    key = f"{_sha(path)}-{kind}-v2"
    cp = state_path("cache", "cards", key + ".json")
    if os.path.exists(cp):
        try:
            return json.load(open(cp, encoding="utf-8"))
        except Exception:
            pass
    schema = P.SCREEN_SCHEMA if kind == "appdemo" else P.PHOTO_SCHEMA
    prompt = P.SCREEN_PROMPT if kind == "appdemo" else P.PHOTO_PROMPT
    data = ask_json(vision, prompt, schema, 0.15, images=[load_b64(path)], ctx=4096, retries=1)
    if not isinstance(data, dict):
        return None
    if kind == "appdemo":
        data["regions"] = _clean_regions(data.get("regions"))
        data["visible_text"] = [clean_line(t)[:60] for t in data.get("visible_text", [])][:12]
        data["numbers"] = [n for n in data.get("numbers", []) if isinstance(n, dict) and n.get("value")][:6]
    json.dump(data, open(cp, "w", encoding="utf-8"))
    return data


def blind_card(path):
    name = re.sub(r"[_\-]+", " ", os.path.splitext(os.path.basename(path))[0]).strip().title()
    return {"screen": name, "purpose": "", "stage": "unknown", "visible_text": [], "numbers": [], "regions": [],
            "subjects": [], "setting": "", "action": "", "mood": "", "one_liner": name}


def character_lock(cards, embedder):
    if not embedder or len(cards) < 3:
        return list(range(len(cards))), False
    vecs = []
    for d in cards:
        try:
            vecs.append(embed(embedder, f"{d.get('subjects', '')} {d.get('setting', '')} {d.get('one_liner', '')}"))
        except Exception:
            vecs.append(None)
    valid = [i for i, v in enumerate(vecs) if v]
    if len(valid) < 3:
        return list(range(len(cards))), False
    seed = max(valid, key=lambda i: sum(cosine(vecs[i], vecs[j]) for j in valid if j != i))
    keep = [i for i in valid if i == seed or cosine(vecs[seed], vecs[i]) > 0.45]
    if len(keep) >= 3:
        log("cast", f"character lock: kept {len(keep)}/{len(cards)} photos of the same subject")
        return keep, True
    return list(range(len(cards))), False


# ------------------------------------------------------------------ helpers
def _facts(models, brief):
    if not (models.writer and brief and len(brief.strip()) > 3):
        return None
    log("facts", "building fact sheet from the brief ...")
    d = ask_json(models.writer, P.fact_sheet_prompt(brief), P.FACT_SCHEMA, 0.2, ctx=8192)
    if isinstance(d, dict) and d.get("claims") is not None:
        d["claims"] = [clean_line(c) for c in d.get("claims", [])][:8]
        return d
    return None


def _hashtags(data):
    return [re.sub(r"^#", "", str(h)).strip() for h in (data.get("hashtags") or []) if str(h).strip()][:8]


def _meta(data, brand, topic=""):
    return {"title": clean_line(data.get("title", topic))[:90] or topic,
            "cta": clean_line(data.get("cta", ""))[:60] or brand["identity"].get("cta", ""),
            "hashtags": _hashtags(data), "description": clean_line(data.get("description", ""))[:320],
            "art": clean_line(data.get("art", ""))[:240]}


def package(models, brand, sb, facts):
    """Platform captions: one cheap call, with a deterministic template fallback."""
    if models.writer:
        d = ask_json(models.writer, P.pack_prompt(brand, sb, facts), P.PACK_SCHEMA, 0.5, ctx=4096)
        if isinstance(d, dict) and d.get("linkedin"):
            sb["platform_captions"] = {k: clean_line(d.get(k, "")) for k in ("linkedin", "x", "youtube_shorts", "instagram")}
            if d.get("hashtags"):
                sb["hashtags"] = _hashtags(d) or sb["hashtags"]
            return sb
    desc = sb.get("description") or sb.get("title", "")
    tags = " ".join("#" + h for h in sb.get("hashtags", []))
    sb["platform_captions"] = {"linkedin": desc, "x": (desc[:200] + " " + tags).strip(),
                               "youtube_shorts": sb.get("title", "")[:100], "instagram": (desc + "\n" + tags).strip()}
    return sb


def _refine(models, brand, sb, raw, facts, cards, rebuild, memory=None, check_numbers=True):
    """lint -> (one) targeted revision -> re-lint. `raw` is the model's JSON; `rebuild(raw)` -> storyboard."""
    for attempt in range(2):
        past = memory.repeats(sb["beats"]) if memory else []
        issues = lint(sb, brand, facts, cards, past, check_numbers)
        if not issues:
            log("lint", "quality gate passed")
            sb["lint"] = []
            return sb
        log("lint", f"{len(issues)} issue(s): " + "; ".join(issues[:3]) + (" ..." if len(issues) > 3 else ""))
        if attempt == 1 or not models.writer:
            break
        log("lint", "asking the writer to fix only those issues ...")
        fixed = ask_json(models.writer, P.revise_prompt(brand, raw, issues, facts), None, 0.3, ctx=12288)
        if isinstance(fixed, dict) and fixed.get("beats"):
            raw = fixed
            sb = rebuild(raw)
        else:
            break
    sb["lint"] = lint(sb, brand, facts, cards, [], check_numbers)
    return sb


# ------------------------------------------------------------------ planners
def plan_appdemo(args, brand, models, memory):
    paths = resolve_images(args.images)
    if len(paths) < 2:
        raise SystemExit("appdemo needs at least 2 images")
    brief = (getattr(args, "brief", None) or args.topic or "").strip()
    facts = _facts(models, brief)
    if models.vision:
        log("vision", f"reading {len(paths)} screenshots with {models.vision} (cached by hash) ...")
        cards = [describe(models.vision, p, "appdemo") or blind_card(p) for p in paths]
    else:
        log("vision", "no multimodal model — blind mode (filenames only, no zoom regions)")
        cards = [blind_card(p) for p in paths]
    if not models.writer:
        log("director", "no writer model — offline storyboard from filenames")
        sb = blind_screens_storyboard(paths, args.topic or brief[:60], brand)
        return package(models, brand, dict(sb, facts=facts), facts)
    n = int(args.lines) if getattr(args, "lines", 0) else clamp(len(paths), 4, 12)
    total = n * 3.6
    log("director", f"directing {n} beats (~{total:.0f}s) with {models.writer} ...")
    raw = ask_json(models.writer, P.director_prompt(brand, facts, cards, n, total, brief), P.DIRECTOR_SCHEMA, 0.4, ctx=12288)

    def rebuild(data):
        beats, per, seen = [], {}, set()
        for b in data.get("beats", []):
            sid = b.get("screen_id")
            if not isinstance(sid, int) or not (0 <= sid < len(paths)):
                continue
            regs = cards[sid].get("regions", [])
            reg = b.get("region")
            if not isinstance(reg, int) or not (0 <= reg < len(regs)):
                reg = per.get(sid, 0) if per.get(sid, 0) < len(regs) else -1     # vary the region per reuse
            if (sid, reg) in seen or per.get(sid, 0) >= 2:
                continue
            seen.add((sid, reg)); per[sid] = per.get(sid, 0) + 1
            kpi = None
            scene = "ui"
            if b.get("scene") == "kpi" and clean_line(b.get("kpi_value", "")):
                scene = "kpi"; kpi = {"value": b["kpi_value"], "label": b.get("kpi_label", "")}
            focus = ([regs[reg]["x"], regs[reg]["y"], regs[reg]["w"], regs[reg]["h"]] if 0 <= reg < len(regs) and scene == "ui" else None)
            beats.append({"scene": scene, "img": sid, "kicker": b.get("kicker", f"STEP {len(beats) + 1}"),
                          "caption": b.get("caption", ""), "sub": b.get("sub", ""), "vo": b.get("vo", ""),
                          "focus": focus, "kpi": kpi})
        sb = {"mode": "appdemo", "images": paths, "facts": facts, "beats": beats, **_meta(data, brand, args.topic or "")}
        return normalize(sb)

    sb = rebuild(raw) if isinstance(raw, dict) else None
    if not sb or not sb["beats"]:
        log("director", "model output unusable — offline storyboard")
        sb = blind_screens_storyboard(paths, args.topic or brief[:60], brand)
        sb["facts"] = facts
        return package(models, brand, sb, facts)
    sb = _refine(models, brand, sb, raw, facts, cards, rebuild, memory)
    sb["cards"] = [{k: c.get(k) for k in ("screen", "purpose", "stage", "regions")} for c in cards]
    return package(models, brand, sb, facts)


def plan_photostory(args, brand, models, memory):
    paths = resolve_images(args.images)
    if len(paths) < 2:
        raise SystemExit("photostory needs at least 2 images")
    wish = args.topic or "a story in pictures"
    cards = ([describe(models.vision, p, "photostory") or blind_card(p) for p in paths] if models.vision
             else [blind_card(p) for p in paths])
    if not models.vision:
        log("vision", "no multimodal model — blind mode")
    keep, _ = character_lock(cards, models.embedder)
    sub = [cards[i] for i in keep]
    data = (ask_json(models.writer, P.story_prompt(brand, sub, wish, len(sub)), P.STORY_SCHEMA, 0.6, ctx=8192)
            if models.writer else None)

    def rebuild(d):
        beats, seen = [], set()
        for b in (d or {}).get("beats", []):
            pid = b.get("photo_id")
            if isinstance(pid, int) and 0 <= pid < len(keep) and pid not in seen:
                seen.add(pid)
                beats.append({"scene": "photo", "img": keep[pid], "kicker": b.get("kicker") or f"SCENE {len(beats) + 1:02d}",
                              "caption": b.get("caption", ""), "sub": b.get("sub", ""), "vo": b.get("vo", "")})
        return normalize({"mode": "photostory", "images": paths, "beats": beats, **_meta(d or {}, brand, wish)})

    sb = rebuild(data) if isinstance(data, dict) else None
    if not sb or not sb["beats"]:
        beats = [{"scene": "photo", "img": i, "kicker": f"SCENE {j + 1:02d}", "caption": ""} for j, i in enumerate(keep)]
        sb = normalize({"mode": "photostory", "images": paths, "beats": beats, "title": wish[:60],
                        "cta": brand["identity"].get("cta", "")})
        return package(models, brand, sb, None)
    sb = _refine(models, brand, sb, data, None, None, rebuild, memory, check_numbers=False)
    return package(models, brand, sb, None)


def plan_lines(args, brand, models, memory):
    topic = args.topic
    n = clamp(int(args.lines) if getattr(args, "lines", 0) else 8, 4, 14)
    brief = (getattr(args, "brief", None) or "").strip()
    facts = _facts(models, brief) if brief else None
    if not models.writer:
        log("script", "no writer model — offline demo pack")
        return demo_storyboard(topic, n)
    log("script", f"writing {n} lines about \"{topic}\" ({args.mode}) with {models.writer} ...")
    raw = ask_json(models.writer, P.lines_prompt(brand, topic, args.mode, n, facts), P.LINES_SCHEMA, 0.6, ctx=8192)
    if not isinstance(raw, dict) or not raw.get("lines"):
        log("script", "model output unusable — offline demo pack")
        return demo_storyboard(topic, n)

    def rebuild(d):
        lines = [clean_line(x) for x in d.get("lines", []) if clean_line(x)][:14]
        vos = d.get("vo") or []
        beats = [{"scene": "line", "kicker": f"{i + 1:02d} / {len(lines)}", "caption": ln,
                  "vo": clean_line(vos[i]) if i < len(vos) else ln} for i, ln in enumerate(lines)]
        return normalize({"mode": args.mode, "beats": beats, "facts": facts, **_meta(d, brand, topic)})

    sb = rebuild(raw)
    sb = _refine(models, brand, sb, raw, facts, None, rebuild, memory, check_numbers=bool(facts))
    return package(models, brand, sb, facts)


def plan_clips(args, brand, models, memory):
    from .stt import transcribe
    if not os.path.exists(args.source):
        raise SystemExit(f"source not found: {args.source}")
    segments = transcribe(args.source, getattr(args, "whisper_model", "small"))
    if not segments or not models.writer:
        return None
    n = clamp(int(args.lines) if getattr(args, "lines", 0) else 6, 3, 14)
    data = ask_json(models.writer, P.miner_prompt(brand, segments, args.mode, n), P.MINER_SCHEMA, 0.4, ctx=12288)
    if not isinstance(data, dict):
        return None
    beats = []
    for c in data.get("clips", []):
        ids = sorted({i for i in c.get("ids", []) if isinstance(i, int) and 0 <= i < len(segments)})[:4]
        if not ids:
            continue
        s, e = segments[ids[0]][0], segments[ids[-1]][1]
        if e - s < 1.2 or e - s > 12:
            continue
        quote = clean_line(c.get("quote", "")) or " ".join(segments[i][2] for i in ids)
        beats.append({"scene": "clip", "kicker": mmss(s), "caption": quote[:220], "sub": clean_line(c.get("caption", ""))[:90],
                      "clip": [s, e]})
        if len(beats) >= n:
            break
    if len(beats) < 3:
        return None
    log("miner", f"mined {len(beats)} voice clips from transcript")
    sb = normalize({"mode": args.mode, "source": os.path.abspath(args.source), "beats": beats,
                    **_meta(data, brand, args.topic or "heard this today")})
    return package(models, brand, sb, None)


def plan(args, brand, models, memory):
    """Entry point: choose the planner for the requested mode."""
    mode = args.mode
    if mode == "appdemo":
        return plan_appdemo(args, brand, models, memory)
    if mode == "photostory":
        return plan_photostory(args, brand, models, memory)
    if getattr(args, "source", None):
        sb = plan_clips(args, brand, models, memory)
        if sb:
            return sb
        log("miner", "could not mine clips — falling back to a written script")
        args.source = None
    if not args.topic:
        raise SystemExit("a topic is required for this mode")
    return plan_lines(args, brand, models, memory)


def regen_beat(sb, index, instruction, brand, models):
    """Rewrite a single beat (UI 'regenerate this beat')."""
    sb = normalize(sb)
    beats = sb["beats"]
    if not (0 <= index < len(beats)) or not models.writer:
        return sb
    neigh = [{"caption": b["caption"], "sub": b["sub"]} for j, b in enumerate(beats) if abs(j - index) <= 2 and j != index]
    d = ask_json(models.writer, P.regen_beat_prompt(brand, sb.get("facts"), beats[index], neigh, instruction),
                 P.BEAT_SCHEMA, 0.6, ctx=6144)
    if isinstance(d, dict) and d.get("caption"):
        for k in ("kicker", "caption", "sub", "vo"):
            if d.get(k) is not None:
                beats[index][k] = clean_line(d[k])
        if beats[index]["scene"] == "kpi" and d.get("kpi_value"):
            beats[index]["kpi"] = {"value": clean_line(d["kpi_value"]), "label": clean_line(d.get("kpi_label", ""))}
    return normalize(sb)

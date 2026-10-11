"""Render orchestration: storyboard + brand -> audio -> frames -> MP4 + captions + SRT + JSON."""
import json, os, shutil, tempfile, threading, time

import numpy as np

from . import audio as A
from . import render as R
from .brand import canvas_for
from .storyboard import build_segments, normalize, timeline, to_srt
from .util import OUT_DIR, ffmpeg_bin, log, mmss, now, run, slugify


def load_images(sb, need):
    from PIL import Image
    out = {}
    for i in sorted(need):
        try:
            p = sb["images"][i]
            im = Image.open(p).convert("RGB")
            im.thumbnail((3000, 3000))
            out[i] = im
        except Exception as e:
            log("assets", f"could not load image {i} ({type(e).__name__}) — beat falls back to text")
    return out


def _watermark(args, brand):
    if getattr(args, "no_watermark", False):
        return ""
    w = getattr(args, "watermark", None)
    return brand["identity"].get("watermark", "") if w is None else w


def _caption_text(sb, brand):
    cap = f"{sb['title']}\n\n{sb.get('description', '')}\n\n" + " ".join("#" + h for h in sb.get("hashtags", []))
    pc = sb.get("platform_captions") or {}
    labels = {"linkedin": "LinkedIn", "x": "X / Twitter", "youtube_shorts": "YouTube Shorts", "instagram": "Instagram"}
    for k, lab in labels.items():
        if pc.get(k):
            cap += f"\n\n--- {lab} ---\n{pc[k]}"
    beats = sb.get("beats", [])
    if beats and beats[0]["scene"] != "clip":
        cap += "\n\n--- Beats ---\n" + "\n".join(f"- {b['caption']}" + (f" — {b['sub']}" if b.get("sub") else "") for b in beats)
    else:
        cap += "\n\n--- Clips ---\n" + "\n".join(f"{mmss(b['clip'][0])} — {b['caption']}" for b in beats if b.get("clip"))
    return cap


def render_storyboard(sb, brand, args, models=None, memory=None):
    t0 = time.time()
    sb = normalize(sb)
    quality = "preview" if getattr(args, "fast", False) else (getattr(args, "quality", None) or brand["output"]["quality"])
    w, h, fps = canvas_for(brand, quality)
    work = tempfile.mkdtemp(prefix="typereel_")
    au = brand["audio"]
    no_bed = getattr(args, "no_bed", False)

    # ---- voice-over first: its length drives beat duration
    vo_events, vo_durs = [], {}
    clip_mode = any(b["scene"] == "clip" for b in sb["beats"])
    want_vo = au["voiceover"] and not clip_mode and not getattr(args, "no_vo", False)
    if want_vo:
        backend = A.tts_backend()
        if not backend:
            log("vo", "voice-over requested but no TTS installed (pip install kokoro soundfile) — continuing without voice")
        else:
            log("vo", f"synthesizing narration with {backend} ...")
            for b in sb["beats"]:
                text = b["vo"] or b["caption"]
                wav = os.path.join(work, f"vo_{b['id']}.wav")
                d = A.synth_vo(text, wav, au["voice"], au["voice_speed"], work)
                if d:
                    vo_durs[b["id"]] = d
                    vo_events.append((b["id"], wav))
    segs = build_segments(sb, brand, vo_durs)
    tl_times = timeline(segs)
    duration = sum(s["secs"] for s in segs)
    log("plan", f"{len(segs)} scenes, {duration:.1f}s, {w}x{h}@{fps} ({quality})")

    # ---- AI backdrop (threaded: GPU paints while CPU prepares audio)
    holder, painter = {}, None
    want_ai = brand["visual"]["background"] == "ai" or getattr(args, "bg", "auto") == "ai"
    if getattr(args, "bg", "auto") in ("grid", "gradient"):
        brand["visual"]["background"] = args.bg
        want_ai = False
    if want_ai:
        from .paint import PaintEngine, art_prompt, has_diffusers
        if has_diffusers() or getattr(args, "bg", "") == "ai":
            brand["visual"]["background"] = "ai"
            pr = [int(x) for x in getattr(args, "paint_res", "768x1344").lower().split("x")]
            if w > h:
                pr = pr[::-1]

            def _paint():
                try:
                    eng = PaintEngine(True, args.paint, args.steps, res=(pr[0], pr[1]), quant=args.quant,
                                      reuse=args.reuse_bg, lora=args.lora, compile=args.compile)
                    holder["img"] = eng.image(art_prompt(sb, brand))
                except Exception as e:
                    holder["err"] = e
            painter = threading.Thread(target=_paint, daemon=True)
            painter.start()
        else:
            log("art", "AI backdrop requested but diffusers/GPU not available — using the procedural engine")
            brand["visual"]["background"] = "grid"

    # ---- audio mix
    ev = []
    for bid, wav in vo_events:
        idx = next(i for i, s in enumerate(segs) if s.get("beat") == bid)
        ev.append((tl_times[idx][0] + 0.35, wav))
    clip_events = []
    if clip_mode and sb.get("source") and os.path.exists(sb["source"]):
        for i, s in enumerate(segs):
            if s.get("clip"):
                cp = os.path.join(work, f"clip{i}.wav")
                run([ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{s['clip'][0]:.3f}",
                     "-to", f"{s['clip'][1]:.3f}", "-i", sb["source"], "-vn", "-ac", "1", "-ar", str(A.SR),
                     "-c:a", "pcm_s16le", cp])
                clip_events.append((tl_times[i][0], cp))
        log("audio", f"cut {len(clip_events)} real voice clips")
    voice = A.vo_track(ev + clip_events, duration) if (ev or clip_events) else np.zeros(int(duration * A.SR), dtype=np.float32)
    music_path = getattr(args, "music", None)
    style = getattr(args, "music_style", None) or au["music"]
    if music_path and os.path.exists(music_path):
        music = A.user_music(music_path, duration, work)
    elif no_bed or style == "none":
        music = np.zeros((int(duration * A.SR), 2), dtype=np.float32)
    else:
        log("music", f"synthesizing {style} @ {au['bpm']} BPM{' (builds)' if au['build'] else ''} ...")
        music = A.synth_music(duration, style, au["bpm"], au["build"])
    n = int(duration * A.SR)
    music = music[:n] if len(music) >= n else np.pad(music, ((0, n - len(music)), (0, 0)))
    music = A.duck(music, voice, au["duck_db"]) if np.any(voice) else music
    mixed = A.mix(music, voice, au["music_volume"], 1.0)
    audio_path = None
    if np.any(mixed):
        audio_path = A.write_wav(os.path.join(work, "mix.wav"), mixed)

    if painter:
        painter.join()
        if holder.get("err"):
            log("art", f"paint failed ({type(holder['err']).__name__}) — procedural engine")
    bg = holder.get("img")
    if brand["visual"]["background"] == "ai" and bg is None:
        brand["visual"]["background"] = "grid"
    log("art", f"background: {'AI-painted' if bg is not None else brand['visual']['background']}")

    # ---- frames
    need = {s["img"] for s in segs if s.get("img") is not None}
    images = load_images(sb, need)
    W = R.World(brand, w, h, fps, bg_image=bg, watermark=_watermark(args, brand))
    scenes = R.build_scenes(W, segs, images)
    tl = R.Timeline(W, scenes)
    log("render", f"rendering {tl.total} frames with {max(os.cpu_count() or 2, 2)} threads ...")
    out_dir = getattr(args, "out", None) or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    base = f"{slugify(sb['title'] or 'reel')}-{now()}"
    out_mp4 = os.path.join(out_dir, base + ".mp4")
    log("encode", "streaming frames to ffmpeg ...")
    R.encode(tl, audio_path, out_mp4, final=(quality == "final"), loudness=au["loudness"])

    # ---- sidecars
    cap_path = os.path.join(out_dir, base + ".caption.txt")
    open(cap_path, "w", encoding="utf-8").write(_caption_text(sb, brand))
    srt_path = os.path.join(out_dir, base + ".srt")
    open(srt_path, "w", encoding="utf-8").write(to_srt(segs, use_vo=bool(vo_events)))
    json_path = os.path.join(out_dir, base + ".json")
    meta = {"duration": round(duration, 2), "resolution": f"{w}x{h}", "fps": fps, "quality": quality,
            "writer": getattr(models, "writer", None), "vision": getattr(models, "vision", None),
            "voiceover": bool(vo_events), "music": style if audio_path else "none", "render_seconds": round(time.time() - t0, 1)}
    json.dump({"storyboard": sb, "brand": brand, "meta": meta}, open(json_path, "w", encoding="utf-8"),
              indent=2, ensure_ascii=False, default=str)
    if memory is not None:
        memory.remember(sb["beats"])
    shutil.rmtree(work, ignore_errors=True)
    return {"video": out_mp4, "caption": cap_path, "srt": srt_path, "json": json_path, "seconds": time.time() - t0}

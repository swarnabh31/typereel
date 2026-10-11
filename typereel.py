#!/usr/bin/env python3
"""TypeReel 6 — brand-driven, 100% local video generator (CLI).

  python typereel.py --probe
  python typereel.py "5 stoic rules for a calm mind" --brand bold-social
  python typereel.py --mode appdemo --brief "Recruitment OS ..." --images shots/*.png --brand engineering-saas
  python typereel.py --mode appdemo --images shots/*.png --plan-only        # review/edit the storyboard first
  python typereel.py --storyboard output/x.storyboard.json                  # render an edited storyboard
"""
import argparse, json, os, sys

from trcore import APP, VERSION
from trcore.brand import (ASPECTS, LEGACY_PALETTES, apply_palette_name, list_presets, load_brand, save_brand,
                          validate_brand)
from trcore.util import OUT_DIR, STATE_DIR, ffmpeg_bin, hex_to_rgb, log, now, slugify

MODES = ["quotes", "facts", "flashcards", "story", "hottake", "appdemo", "photostory"]


def build_parser():
    ap = argparse.ArgumentParser(description=f"{APP} v{VERSION}")
    ap.add_argument("topic", nargs="?", default="", help="topic / one-line idea")
    ap.add_argument("--brief", default="", help="longer product/story brief (text, or a path to a .txt/.md file)")
    ap.add_argument("--mode", default="quotes", choices=MODES)
    ap.add_argument("--images", nargs="*", help="screenshots/photos (order = story order) or a folder; use with --mode appdemo|photostory")
    ap.add_argument("--source", help="video/audio to mine for voice clips (ClipMiner)")
    ap.add_argument("--brand", default="engineering-saas", help="preset id or path to a brand .json")
    ap.add_argument("--lines", type=int, default=0, help="number of beats (0 = automatic)")
    ap.add_argument("--aspect", choices=list(ASPECTS), help="override output aspect")
    ap.add_argument("--quality", choices=["preview", "final"], help="override output quality")
    ap.add_argument("--fast", action="store_true", help="alias for --quality preview")
    ap.add_argument("--palette", choices=list(LEGACY_PALETTES), help="override the brand palette with a classic one")
    ap.add_argument("--bg", default="auto", choices=["auto", "ai", "grid", "gradient"], help="backdrop (auto = brand default)")
    ap.add_argument("--cta", help="override the closing call-to-action text")
    ap.add_argument("--font", help="override heading font (family name or file in ./fonts)")
    ap.add_argument("--music", help="audio file to use instead of the synthesized track")
    ap.add_argument("--music-style", choices=["lofi-ambient", "minimal-pulse", "cinematic-drone", "none"])
    ap.add_argument("--vo", dest="vo", action="store_true", help="force voice-over on")
    ap.add_argument("--no-vo", dest="no_vo", action="store_true", help="force voice-over off")
    ap.add_argument("--no-bed", action="store_true", help="silent video (no music)")
    ap.add_argument("--watermark", default=None, help="small footer text (default: brand setting)")
    ap.add_argument("--no-watermark", action="store_true")
    ap.add_argument("--no-memory", action="store_true", help="skip repeat detection")
    ap.add_argument("--writer", help="force a specific Ollama writer model")
    ap.add_argument("--vision", help="force a specific Ollama vision model")
    ap.add_argument("--whisper-model", default="small")
    ap.add_argument("--out", default=OUT_DIR)
    # storyboard workflow
    ap.add_argument("--plan-only", action="store_true", help="write the storyboard JSON and stop (no render)")
    ap.add_argument("--storyboard", help="render this (edited) storyboard JSON instead of planning")
    ap.add_argument("--storyboard-out", help="where --plan-only / --regen-beat writes the storyboard")
    ap.add_argument("--regen-beat", type=int, default=0, help="1-based beat number to rewrite (needs --storyboard)")
    ap.add_argument("--instruction", default="", help="guidance for --regen-beat")
    ap.add_argument("--save-brand", help="write the resolved brand kit to this path and exit")
    # AI backdrop engine
    ap.add_argument("--paint", default="auto", choices=["auto", "flux", "zimage", "qwen", "sd35", "sdxl"])
    ap.add_argument("--paint-res", default="768x1344")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--quant", default="auto", choices=["auto", "none", "8bit", "4bit"])
    ap.add_argument("--reuse-bg", action="store_true", help="theme lock: reuse the last painted backdrop")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--warm", action="store_true", help="download/load the paint engine once, then exit")
    ap.add_argument("--list-brands", action="store_true")
    return ap


def probe():
    from trcore import audio as A
    from trcore.fonts import list_families, resolve_font
    from trcore.llm import (EDITOR_PREFS, EMBED_PREFS, OLLAMA, VISION_PREFS, WRITER_PREFS, list_models, ollama_ok,
                            resolve_model, vision_capable)
    from trcore.paint import PROVIDERS, _hf_downloaded, auto_provider
    from trcore.stt import find_ggml, find_whisper_cpp, has_faster_whisper
    print(f"\n{APP} v{VERSION} — preflight\n")
    print(f"  ffmpeg ....... {ffmpeg_bin() or 'MISSING'}")
    up = ollama_ok()
    print(f"  ollama ....... {'OK at ' + OLLAMA if up else 'DOWN — run: ollama serve (offline demo mode still works)'}")
    if up:
        models = list_models()
        print(f"  writer ....... {resolve_model(WRITER_PREFS, models) or 'not found (offline demo pack)'}")
        print(f"  embedder ..... {resolve_model(EMBED_PREFS, models) or 'not found (memory off)'}")
        vm = resolve_model(VISION_PREFS, models)
        print(f"  vision ....... {(vm + ' (eyes ON)') if (vm and vision_capable(vm)) else 'none multimodal — image modes run blind'}")
    wc, gg = find_whisper_cpp(), find_ggml("small")
    print(f"  stt .......... {('whisper.cpp + ' + os.path.basename(gg)) if (wc and gg) else ('faster-whisper' if has_faster_whisper() else 'NONE — --source disabled')}")
    print(f"  voice-over ... {A.tts_backend() or 'NONE — pip install kokoro soundfile (or piper / pyttsx3)'}")
    try:
        import torch
        print(f"  torch ........ {torch.__version__} cuda={torch.cuda.is_available()}")
        cached = [p for p in PROVIDERS if _hf_downloaded(PROVIDERS[p]['repo'])]
        print(f"  paint ........ cached: {', '.join(cached) or 'none'} | auto -> {auto_provider()}")
    except Exception:
        print("  paint ........ procedural engine only (optional: pip install torch diffusers)")
    print(f"  fonts ........ {', '.join(list_families())}  (default: {resolve_font(None)})")
    print(f"  brands ....... {', '.join(p['id'] for p in list_presets())}\n")


def apply_overrides(brand, a):
    if a.palette:
        apply_palette_name(brand, a.palette)
    if a.aspect:
        brand["output"]["aspect"] = a.aspect
    if a.cta:
        brand["identity"]["cta"] = a.cta
    if a.font:
        for k in ("heading", "body"):
            brand["visual"]["fonts"][k] = a.font
    if a.music_style:
        brand["audio"]["music"] = a.music_style
    if a.vo:
        brand["audio"]["voiceover"] = True
    if a.no_vo:
        brand["audio"]["voiceover"] = False
    return validate_brand(brand)


def main(argv=None):
    a = build_parser().parse_args(argv)
    os.makedirs(STATE_DIR, exist_ok=True)
    if a.list_brands:
        for p in list_presets():
            print(f"{p['id']:<20} {p['name']} — {p['description']}")
        return 0
    if a.probe:
        probe(); return 0
    if a.warm:
        from trcore.paint import PaintEngine
        eng = PaintEngine(True, a.paint, a.steps or None, quant=a.quant)
        print("\n  Paint engine warm." if eng.pipe else "\n  Paint engine unavailable."); return 0
    if not ffmpeg_bin():
        print("ffmpeg not found. Put it on PATH or in ./ffmpeg/bin/ (https://ffmpeg.org).", file=sys.stderr); return 1
    if a.brief and os.path.isfile(a.brief):
        a.brief = open(a.brief, encoding="utf-8").read()

    brand = apply_overrides(load_brand(a.brand), a)
    if a.save_brand:
        print("  Brand kit  :", save_brand(brand, a.save_brand)); return 0

    from trcore import pipeline, planner
    print(f"\n  {APP} v{VERSION} · brand: {brand['name']}")
    models = planner.resolve_models(a.writer, a.vision)
    log("models", f"writer={models.writer or '—'}  vision={models.vision or '—'}  embed={models.embedder or '—'}")
    memory = planner.Memory(not a.no_memory, models.embedder)

    # ---- regenerate one beat
    if a.regen_beat:
        if not a.storyboard:
            print("--regen-beat needs --storyboard", file=sys.stderr); return 1
        sb = json.load(open(a.storyboard, encoding="utf-8"))
        sb = planner.regen_beat(sb, a.regen_beat - 1, a.instruction, brand, models)
        out = a.storyboard_out or a.storyboard
        json.dump(sb, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"\n  Storyboard : {out}"); return 0

    # ---- plan
    if a.storyboard:
        data = json.load(open(a.storyboard, encoding="utf-8"))
        sb = data.get("storyboard", data)             # accepts a full run .json too
        log("plan", f"loaded storyboard: {len(sb.get('beats', []))} beats")
    else:
        if a.mode in ("appdemo", "photostory") and not a.images:
            print(f"--mode {a.mode} needs --images", file=sys.stderr); return 1
        if a.mode not in ("appdemo", "photostory") and not (a.topic or a.source):
            print("give a topic, or --source, or --images with --mode appdemo", file=sys.stderr); return 1
        sb = planner.plan(a, brand, models, memory)

    from trcore.storyboard import normalize
    sb = normalize(sb)
    if a.plan_only:
        out = a.storyboard_out or os.path.join(a.out, f"{slugify(sb['title'] or a.topic)}-{now()}.storyboard.json")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        json.dump(sb, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"\n  Storyboard : {out}\n  Beats      : {len(sb['beats'])}"); return 0

    # ---- render
    res = pipeline.render_storyboard(sb, brand, a, models, memory)
    print(f"\n  Video   : {res['video']}\n  Caption : {res['caption']}\n  Subs    : {res['srt']}\n"
          f"  Data    : {res['json']}\n  Done in {res['seconds']:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

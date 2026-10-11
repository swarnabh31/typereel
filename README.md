# TypeReel

**Brief → brand → storyboard → video.** Describe a product (or drop screenshots, photos, a podcast), choose a
Brand Kit, review an editable storyboard, and get a ready-to-post video with synthesized music and optional
voice-over. 100 % local: Ollama writes, Pillow renders, ffmpeg encodes. No cloud, no keys, no subscriptions.

## What's new in v6

| Area | v5 | v6 |
|---|---|---|
| Brand & style | fixed palettes | **Brand Kits** (JSON): voice, banned words, reference lines, palette, grid/glow/glass, fonts, camera, music, voice-over. 6 presets + import/export |
| Prompting | one generic "viral" prompt | **Staged pipeline**: fact sheet → vision cards (cached) → director → deterministic lint → one targeted revision → platform copy. Schema-constrained JSON |
| Grounding | model could invent claims | Only claims from your fact sheet / screenshots; numbers are verified by the lint |
| Workflow | straight to render | **Storyboard review**: edit every caption/VO/duration/zoom region, rewrite one beat with AI, *then* render |
| Motion | static card, 6 % zoom | Focus-region camera pan/zoom, animated **cursor + click**, spotlight, parallax grid, glass panels, KPI count-up, true cross-dissolves |
| Audio | brown noise | Synthesized **90 BPM-style music that builds**, local **TTS voice-over**, auto-ducking, loudness normalisation |
| Output | 9:16 only | 9:16 · 16:9 · 1:1 · 4:5, plus MP4 + SRT + per-platform captions + project JSON |
| Speed/quality | JPEG frames → x264 | threaded renderer piping **raw frames** to ffmpeg (no double compression) |
| UI | 25-control sidebar | 4-step wizard + live style preview; engine settings tucked away |

## Install

```bash
pip install -r requirements.txt
# ffmpeg: on PATH, or drop a build at ./ffmpeg/bin/   (needs libx264 or libopenh264)
ollama pull gemma3:12b            # writer (any capable model works)
ollama pull qwen3.8:27b           # optional: vision model for screenshots/photos (or gemma4:12b)
ollama pull nomic-embed-text      # optional: repeat detection + character lock
pip install -r requirements-optional.txt   # optional: voice-over, ClipMiner, AI backdrops
```

Run the studio: `streamlit run ui.py` (or `./run.sh` / `run.bat`) → http://localhost:8501

Without Ollama the app still works in an **offline mode** (demo script / filename-based storyboard) and says so in the log.

## The workflow

1. **Brief** — pick *idea / app screenshots / photos / podcast*. For screenshots, paste your pitch or README;
   the AI turns it into a *fact sheet* and may only claim what is in it.
2. **Brand & style** — choose a preset, then edit voice, palette, backdrop, cards, captions, motion and audio.
   The preview on the right re-renders live. Save/import brand kits as JSON.
3. **Storyboard** — the plan appears as an editable table (scene, caption, sub-line, voice-over, seconds,
   zoom region, KPI). Edit freely or rewrite a single beat with AI.
4. **Render** — pick the platform (aspect) and quality; get MP4, SRT, captions per platform and a project file
   you can re-open from the Library.

## CLI

```bash
python typereel.py --probe
python typereel.py --list-brands

# a topic
python typereel.py "5 stoic rules for a calm mind" --brand bold-social --mode quotes

# app tour from screenshots, reviewed before rendering
python typereel.py --mode appdemo --brief pitch.md --images shots/*.png --brand engineering-saas --plan-only
#   -> edit output/<title>.storyboard.json if you like, then:
python typereel.py --storyboard output/<title>.storyboard.json --quality final --aspect 16:9

# rewrite one beat
python typereel.py --storyboard sb.json --regen-beat 3 --instruction "lead with the number"

# podcast clips with real voice
python typereel.py --source podcast.mp4 --mode quotes --lines 6
```

Key flags: `--brand` (preset id or `.json`), `--aspect`, `--quality preview|final` (`--fast`), `--lines`, `--vo/--no-vo`,
`--music FILE`, `--music-style`, `--palette` (classic palettes), `--font`, `--cta`, `--bg ai|grid|gradient`,
`--writer/--vision` (force a model), `--save-brand PATH`. Run `python typereel.py -h` for all.

Every render writes: `*.mp4`, `*.srt`, `*.caption.txt` (all platform copy + hashtags) and `*.json` (storyboard + brand + meta).

## Brand Kits

A brand kit is one JSON file (see `brands/engineering-saas.json`). Sections:

- `identity` — name, tagline, website, CTA, optional watermark
- `voice` — tone words, narration style, **banned words**, style rules, **reference lines**, word limits. This block is injected into every prompt and enforced again by the lint.
- `visual` — palette, backdrop (`grid | gradient | ai | solid`), glow, card style (glass/solid/bare, radius, window bar), caption panel, fonts, grade
- `motion` — camera (`slow_zoom`, `ui_pan`, `parallax`), cursor, spotlight, transition, pacing
- `audio` — music style/BPM/build, voice-over voice & speed, ducking, loudness

Presets: Engineering-first SaaS · Editorial minimal · Playful consumer · Bold social · Luxury noir · Documentary warm.

## How the prompts work

See [`docs/PROMPTS.md`](docs/PROMPTS.md). In short: each stage receives only the context it needs, the brand voice block
everywhere, schema-constrained output, low temperature for planning, and a deterministic quality gate
(`trcore/storyboard.py::lint`) that checks banned terms, word limits, ungrounded numbers, repeated openings and
repeats of earlier videos — then asks the model to fix *only those issues*, once.

## Voice-over, music, subtitles

- **Voice-over** needs one local TTS: `pip install kokoro soundfile` (best), or Piper (`./voices/*.onnx`), or `pyttsx3`.
  Beat length follows the narration; music ducks under it. No TTS installed → renders without voice and tells you.
- **Music** is synthesized (no samples, no licensing): `lofi-ambient`, `minimal-pulse`, `cinematic-drone`; or pass `--music file`.
- **SRT** is written next to every video.

## Optional engines

- **ClipMiner** (`--source`): whisper.cpp + a `ggml-*.bin` in `./models/`, or `pip install faster-whisper`.
- **AI backdrop** (`Backdrop: ai`): FLUX.1-schnell by default via Diffusers; one painting, Ken-Burns'd per scene.
  Tip: set `OLLAMA_KEEP_ALIVE=0` while painting — the app also unloads Ollama models itself before loading the painter.

## Fonts

Inter ships in `fonts/` (OFL). `python download_fonts.py` adds more Google Fonts; drop any `.ttf/.otf` into `fonts/`
and it appears in the Brand Kit font pickers.

## Project layout

```
typereel.py          CLI
ui.py                Streamlit studio (4-step wizard)
trcore/              brand · prompts · planner · storyboard (+lint) · render · audio · paint · stt · pipeline
brands/              preset brand kits (JSON)
fonts/               Inter + any fonts you add
tests/               pytest suite (fake Ollama server included)
docs/PROMPTS.md      the prompt architecture
```

Run the tests: `pytest tests -q` (needs ffmpeg for the render tests).

## Troubleshooting

- *"no supported H.264 encoder"* → use a full ffmpeg build (libx264) at `./ffmpeg/bin/`.
- *Screenshots come out blind (no zoom regions)* → pull a multimodal model (`ollama pull gemma3:12b` or `qwen3.8:27b`).
- *Text too small in the video* → raise **Caption text size** in Brand → Visual, or shorten the caption limit in Voice.
- *Voice-over silent* → run `--probe`; it lists the TTS backend it found.
- *Slow render* → use **preview** quality while iterating; rendering is threaded and scales with CPU cores.

Licences: see `LICENSE` (code, MIT) and `THIRD_PARTY.md` (fonts, models, tools).

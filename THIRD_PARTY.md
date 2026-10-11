# Third-party components & licences

TypeReel's own code is MIT. Everything below is **optional or bundled separately** and carries its own terms —
check them before you redistribute videos or this package commercially. (This is a summary, not legal advice.)

| Component | Used for | Licence (verify upstream) |
|---|---|---|
| Inter (bundled in `fonts/`) | default typeface | SIL OFL 1.1 — see `fonts/OFL-Inter.txt` |
| Other Google Fonts (`download_fonts.py`) | optional typefaces | SIL OFL 1.1 (each family) |
| Pillow, NumPy, Streamlit, pandas | core | HPND / BSD / Apache-2.0 / BSD |
| FFmpeg (you install it) | encoding | LGPL/GPL depending on the build |
| Ollama + the models you pull | writing / vision | each model has its own licence (Gemma, Qwen, Llama …) |
| Kokoro-82M | optional voice-over | Apache-2.0 |
| Piper voices | optional voice-over | per-voice licence |
| faster-whisper / whisper.cpp | optional ClipMiner STT | MIT |
| FLUX.1-schnell | optional AI backdrop (default) | Apache-2.0 |
| Z-Image-Turbo, Qwen-Image | optional AI backdrop | check the model card |
| Stable Diffusion 3.5 / SDXL | optional AI backdrop | Stability community / OpenRAIL licences |

**Music** is synthesised by TypeReel itself (`trcore/audio.py`) — no samples, nothing to license.
If you supply your own track with `--music`, you are responsible for its rights.
**ClipMiner** reuses real audio from a file you supply: only mine content you have the right to reuse.

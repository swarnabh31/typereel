"""Audio: procedural music (numpy), local TTS adapters, VO ducking, final mix.

No cloud, no licensed samples: the music is synthesised, so it is safe to publish.
"""
import math, os, shutil, struct, subprocess, wave

import numpy as np

from .util import log, run, ffmpeg_bin, ROOT

SR = 44100

# ------------------------------------------------------------------ wav io
def write_wav(path, arr):
    """arr: float32 (n,) mono or (n,2) stereo in [-1,1] -> 16-bit PCM."""
    a = np.asarray(arr, dtype=np.float32)
    ch = 1 if a.ndim == 1 else a.shape[1]
    pcm = (np.clip(a, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(ch); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


def read_wav(path):
    """Canonical 16-bit wav -> float32 array (n,) or (n,2)."""
    with wave.open(path, "rb") as w:
        n, ch, sw = w.getnframes(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(n)
    if sw != 2:
        raise ValueError("expected 16-bit wav")
    a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return a.reshape(-1, ch) if ch > 1 else a


def to_canonical(src, dst, channels=1):
    run([ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-i", src,
         "-ac", str(channels), "-ar", str(SR), "-c:a", "pcm_s16le", dst])
    return dst


def duration(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


# ------------------------------------------------------------------ music synth
def _hz(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


STYLES = {
    #  progression: (bass_root_midi, [pad notes])   bars_per_chord, layers
    "lofi-ambient": dict(prog=[(45, [57, 60, 64]), (41, [53, 57, 60]), (48, [55, 60, 64]), (43, [55, 59, 62])],
                         bars=1, arp=True, hat=True, kick=True, pad=0.9),
    "minimal-pulse": dict(prog=[(45, [57, 60, 64]), (48, [55, 60, 64]), (43, [55, 59, 62]), (41, [53, 57, 60])],
                          bars=1, arp=False, hat=True, kick=True, pad=0.7),
    "cinematic-drone": dict(prog=[(36, [48, 55, 60]), (41, [53, 57, 65]), (43, [50, 55, 62]), (36, [48, 55, 63])],
                            bars=2, arp=False, hat=False, kick=False, pad=1.0),
}


def _env_ramp(t, start, end, lo=0.0, hi=1.0):
    return np.clip((t - start) / max(end - start, 1e-6), 0, 1) * (hi - lo) + lo


def synth_music(dur, style="lofi-ambient", bpm=90, build=True):
    """Stereo float32 array of length dur seconds. Layers enter over time so it 'builds' subtly."""
    n = int(dur * SR)
    out = np.zeros((n, 2), dtype=np.float32)
    if style == "none" or n <= 0:
        return out
    cfg = STYLES.get(style, STYLES["lofi-ambient"])
    t_all = np.arange(n) / SR
    beat = 60.0 / bpm
    bar = beat * 4
    seg_len = bar * cfg["bars"]
    # build curves (constant when build=False)
    def layer(start_frac, end_frac, hi=1.0):
        if not build:
            return np.full(n, hi, dtype=np.float32)
        return _env_ramp(t_all, dur * start_frac, dur * end_frac, 0.0, hi).astype(np.float32)
    pad_gain = (0.65 + 0.35 * layer(0.0, 0.9)) if build else np.full(n, 1.0, dtype=np.float32)

    # --- pad + bass, per chord segment (overlap-added with soft attack/release)
    k = 0
    while k * seg_len < dur + seg_len:
        bass, notes = cfg["prog"][k % len(cfg["prog"])]
        t0 = k * seg_len
        i0 = int(t0 * SR); i1 = min(n, int((t0 + seg_len + 0.9) * SR))
        if i0 >= n:
            break
        tt = (np.arange(i0, i1) / SR) - t0
        atk, rel = 0.7, 0.9
        env = np.minimum(tt / atk, 1.0) * np.clip((seg_len + rel - tt) / rel, 0, 1)
        env = env * env * (3 - 2 * env)
        pad = np.zeros(len(tt), dtype=np.float32)
        for j, m in enumerate(notes):
            f = _hz(m)
            pad += np.sin(2 * np.pi * f * tt) + np.sin(2 * np.pi * f * 1.0045 * tt + j) + 0.25 * np.sin(2 * np.pi * 2 * f * tt)
        pad *= (0.045 * cfg["pad"] / max(len(notes), 1)) * env
        wob = 0.5 + 0.5 * np.sin(2 * np.pi * 0.11 * tt + k)      # slow stereo drift
        out[i0:i1, 0] += pad * (0.75 + 0.25 * wob) * pad_gain[i0:i1]
        out[i0:i1, 1] += pad * (1.0 - 0.25 * wob) * pad_gain[i0:i1]
        sub = np.sin(2 * np.pi * _hz(bass) * tt) * np.clip(1 - tt / (seg_len + 0.5), 0, 1) * np.minimum(tt / 0.05, 1)
        out[i0:i1] += (0.20 * sub * env)[:, None] * 1.0
        k += 1

    # --- arpeggio (plucks), enters ~20% and grows
    if cfg["arp"]:
        arp_gain = layer(0.18, 0.85, 1.0)
        step = beat / 2
        pattern = [0, 1, 2, 1, 2, 1, 0, 2]
        idx = 0
        t = 0.0
        while t < dur:
            seg = int(t // seg_len)
            _, notes = cfg["prog"][seg % len(cfg["prog"])]
            m = notes[pattern[idx % len(pattern)] % len(notes)] + 12
            i0 = int(t * SR); L = int(0.55 * SR); i1 = min(n, i0 + L)
            if i1 > i0:
                tt = np.arange(i1 - i0) / SR
                p = np.sin(2 * np.pi * _hz(m) * tt) * np.exp(-tt / 0.22) * np.minimum(tt / 0.006, 1)
                g = 0.07 * arp_gain[min(i0, n - 1)]
                pan = 0.5 + 0.35 * math.sin(idx * 1.3)
                out[i0:i1, 0] += (p * g * (1 - pan) * 1.4).astype(np.float32)
                out[i0:i1, 1] += (p * g * pan * 1.4).astype(np.float32)
            t += step; idx += 1
        d = int(beat * 0.75 * SR)              # dotted-eighth echo for width
        echo = np.zeros_like(out); echo[d:] = out[:-d] * 0.28 if d < n else 0
        out += echo * 0.5

    # --- kick + hat
    if cfg["kick"]:
        kg = layer(0.30, 0.75, 1.0)
        t = 0.0; b = 0
        while t < dur:
            if b % 4 in (0, 2):
                i0 = int(t * SR); L = int(0.28 * SR); i1 = min(n, i0 + L)
                if i1 > i0:
                    tt = np.arange(i1 - i0) / SR
                    f = 48 + 70 * np.exp(-tt / 0.03)
                    ph = 2 * np.pi * np.cumsum(f) / SR
                    kick = np.sin(ph) * np.exp(-tt / 0.09)
                    out[i0:i1] += (0.22 * kg[min(i0, n - 1)] * kick)[:, None]
            t += beat; b += 1
    if cfg["hat"]:
        hg = layer(0.45, 0.9, 1.0)
        rng = np.random.default_rng(3)
        t = beat / 2
        while t < dur:
            i0 = int(t * SR); L = int(0.05 * SR); i1 = min(n, i0 + L)
            if i1 > i0:
                nz = rng.standard_normal(i1 - i0).astype(np.float32)
                nz = np.diff(nz, prepend=0)                                   # crude high-pass
                env = np.exp(-np.arange(i1 - i0) / SR / 0.012)
                out[i0:i1] += (0.05 * hg[min(i0, n - 1)] * nz * env)[:, None]
            t += beat

    # --- master: fades + normalise
    fi, fo = min(1.6, dur / 4), min(2.4, dur / 3)
    fade = np.ones(n, dtype=np.float32)
    fade[: int(fi * SR)] = np.linspace(0, 1, int(fi * SR))
    fade[-int(fo * SR):] *= np.linspace(1, 0, int(fo * SR))
    out *= fade[:, None]
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * 0.8).astype(np.float32)


def user_music(src, dur, workdir):
    """Load any audio file as stereo, loop/trim to dur with a 0.5 s crossfade on loops."""
    wav = to_canonical(src, os.path.join(workdir, "usermusic.wav"), channels=2)
    a = read_wav(wav)
    n = int(dur * SR)
    if len(a) == 0:
        return np.zeros((n, 2), dtype=np.float32)
    if len(a) >= n:
        out = a[:n].copy()
    else:
        reps = [a]
        total = len(a)
        xf = min(int(0.5 * SR), len(a) // 4)
        out = a.copy()
        while len(out) < n:
            nxt = a.copy()
            if xf:
                ramp = np.linspace(0, 1, xf)[:, None]
                out[-xf:] = out[-xf:] * (1 - ramp) + nxt[:xf] * ramp
                out = np.concatenate([out, nxt[xf:]])
            else:
                out = np.concatenate([out, nxt])
        out = out[:n]
    fo = int(min(1.5, dur / 3) * SR)
    out[-fo:] *= np.linspace(1, 0, fo)[:, None]
    fi = int(min(0.8, dur / 4) * SR)
    out[:fi] *= np.linspace(0, 1, fi)[:, None]
    return out


# ------------------------------------------------------------------ TTS
VOICE_PRESETS = {                      # Kokoro-82M voice ids (blends are comma-separated)
    "neutral": "af_nicole,am_michael",
    "calm-female": "af_heart",
    "calm-male": "am_michael",
    "warm-female": "af_bella",
    "deep-male": "am_onyx",
}
_KOKORO = None


def tts_backend():
    if os.environ.get("TYPEREEL_TTS") == "fake":
        return "fake"
    try:
        import kokoro  # noqa: F401
        import soundfile  # noqa: F401
        return "kokoro"
    except Exception:
        pass
    if _piper_model() and (shutil.which("piper") or _has_mod("piper")):
        return "piper"
    if _has_mod("pyttsx3"):
        return "pyttsx3"
    return None


def _has_mod(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False


def _piper_model():
    m = os.environ.get("PIPER_MODEL")
    if m and os.path.exists(m):
        return m
    import glob
    hits = glob.glob(os.path.join(ROOT, "voices", "*.onnx"))
    return hits[0] if hits else None


def synth_vo(text, out_wav, voice="neutral", speed=0.95, workdir=None):
    """Synthesize `text` to a canonical mono wav. Returns duration seconds, or None on failure."""
    backend = tts_backend()
    if not backend or not text.strip():
        return None
    workdir = workdir or os.path.dirname(out_wav)
    raw = out_wav + ".raw.wav"
    try:
        if backend == "fake":                       # deterministic test voice: tone burst per word
            words = max(len(text.split()), 1)
            dur = words * 0.34 / max(speed, 0.5)
            t = np.arange(int(dur * SR)) / SR
            a = 0.25 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
            write_wav(out_wav, a.astype(np.float32))
            return duration(out_wav)
        if backend == "kokoro":
            global _KOKORO
            import soundfile as sf
            from kokoro import KPipeline
            if _KOKORO is None:
                _KOKORO = KPipeline(lang_code="a")
            vid = VOICE_PRESETS.get(voice, voice)
            try:
                chunks = [np.asarray(getattr(a, "numpy", lambda: a)()) for _, _, a in _KOKORO(text, voice=vid, speed=speed)]
            except Exception:
                vid = vid.split(",")[0]
                chunks = [np.asarray(getattr(a, "numpy", lambda: a)()) for _, _, a in _KOKORO(text, voice=vid, speed=speed)]
            sf.write(raw, np.concatenate(chunks), 24000)
        elif backend == "piper":
            cmd = [shutil.which("piper") or "piper", "--model", _piper_model(), "--output_file", raw,
                   "--length_scale", f"{1.0 / max(speed, 0.5):.2f}"]
            subprocess.run(cmd, input=text, text=True, check=True, capture_output=True)
        elif backend == "pyttsx3":
            import pyttsx3
            e = pyttsx3.init(); e.setProperty("rate", int(175 * speed))
            e.save_to_file(text, raw); e.runAndWait()
        to_canonical(raw, out_wav, 1)
        os.remove(raw)
        return duration(out_wav)
    except Exception as ex:
        log("vo", f"TTS failed ({type(ex).__name__}: {str(ex)[:80]}) — continuing without voice")
        return None


# ------------------------------------------------------------------ mixing
def vo_track(events, total):
    """events: [(offset_sec, wav_path)] -> mono float32 track of `total` seconds."""
    n = int(total * SR)
    track = np.zeros(n, dtype=np.float32)
    for off, path in events:
        a = read_wav(path)
        a = a if a.ndim == 1 else a.mean(axis=1)
        i0 = int(off * SR)
        if i0 >= n:
            continue
        seg = a[: n - i0]
        track[i0:i0 + len(seg)] += seg
    return track


def duck(music, vo, duck_db=-14.0):
    """Lower music under speech: smoothed VO activity -> gain curve (deterministic side-chain)."""
    if not np.any(vo):
        return music
    env = np.convolve(np.abs(vo), np.ones(int(0.04 * SR)) / int(0.04 * SR), mode="same")
    active = (env > 0.015).astype(np.float32)
    k = int(0.35 * SR)                                  # attack/release smoothing
    smooth = np.convolve(active, np.ones(k) / k, mode="same")
    smooth = np.clip(smooth * 1.6, 0, 1)
    g = 1.0 - smooth * (1.0 - 10 ** (duck_db / 20.0))
    return music * g[:, None]


def mix(music, vo, music_gain=0.55, vo_gain=1.0):
    n = max(len(music), len(vo))
    out = np.zeros((n, 2), dtype=np.float32)
    out[: len(music)] += music * music_gain
    if len(vo):
        out[: len(vo)] += (vo * vo_gain)[:, None]
    peak = float(np.max(np.abs(out))) if n else 0
    if peak > 0.98:
        out *= 0.98 / peak
    return out

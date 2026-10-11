"""Speech-to-text for ClipMiner: whisper.cpp or faster-whisper. Timestamps come from STT, never from the LLM."""
import glob, json, os, shutil, tempfile

from .util import log, run, ffmpeg_bin, ROOT

WHISPER_BINS = ["whisper-cli", "whisper-cpp", "whisper", "main"]


def find_whisper_cpp():
    for n in WHISPER_BINS:
        p = shutil.which(n)
        if p:
            return p
    return None


def find_ggml(pref):
    cands = []
    if os.environ.get("WHISPER_MODEL"):
        cands.append(os.environ["WHISPER_MODEL"])
    cands += (glob.glob(os.path.join(ROOT, "models", "ggml-*.bin")) + glob.glob("ggml-*.bin") +
              glob.glob(os.path.expanduser("~/.cache/whisper-cpp/ggml-*.bin")))
    if pref:
        hit = [c for c in cands if pref in os.path.basename(c)]
        if hit:
            return hit[0]
    return cands[0] if cands else None


def has_faster_whisper():
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def transcribe(src, model_pref):
    tmp = tempfile.mkdtemp(prefix="tr_stt_"); wav = os.path.join(tmp, "in16k.wav")
    log("stt", "extracting audio track with ffmpeg ...")
    run([ffmpeg_bin(), "-y", "-hide_banner", "-loglevel", "error", "-i", src, "-vn", "-ac", "1", "-ar", "16000", wav])
    binp, gg = find_whisper_cpp(), find_ggml(model_pref)
    if binp and gg:
        log("stt", f"whisper.cpp ({os.path.basename(gg)}) transcribing ...")
        prefix = os.path.join(tmp, "out")
        try:
            run([binp, "-m", gg, "-f", wav, "-l", "auto", "-np", "-oj", "-of", prefix])
            data = json.load(open(prefix + ".json", encoding="utf-8"))
            segs = []
            for s in data.get("transcription", []):
                t = s.get("text", "").strip()
                if t:
                    off = s.get("offsets", {})
                    segs.append((off.get("from", 0) / 1000.0, off.get("to", 0) / 1000.0, t))
            if segs:
                return segs
        except Exception as e:
            log("stt", f"whisper.cpp failed ({type(e).__name__}) — trying faster-whisper")
    try:
        from faster_whisper import WhisperModel
        log("stt", f"faster-whisper ({model_pref or 'small'}) transcribing ...")
        m = WhisperModel(model_pref or "small", device="auto", compute_type="default")
        segments, _ = m.transcribe(wav)
        return [(s.start, s.end, s.text.strip()) for s in segments if s.text.strip()]
    except Exception:
        return None

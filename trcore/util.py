"""Small shared helpers (logging, paths, colour, easing, subprocess)."""
import math, os, re, shutil, subprocess, sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_DIR = os.path.join(ROOT, ".typereel")
OUT_DIR = os.path.join(ROOT, "output")


def log(step, msg):
    print(f"  [{step}] {msg}", flush=True)


def slugify(s, n=40):
    s = re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")
    return s[:n] or "reel"


def now():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def mmss(t):
    return f"{int(t)//60:02d}:{int(t)%60:02d}"


def state_path(*parts):
    p = os.path.join(STATE_DIR, *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p


def ffmpeg_bin():
    """Prefer ./ffmpeg/bin/ffmpeg(.exe) over PATH (avoids PATH build mismatches)."""
    for name in ("ffmpeg.exe", "ffmpeg"):
        p = os.path.join(ROOT, "ffmpeg", "bin", name)
        if os.path.exists(p):
            return p
    return shutil.which("ffmpeg")


def run(cmd, timeout=None):
    """Run a command; raise RuntimeError with the stderr tail on failure."""
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-6:]
        raise RuntimeError(f"{os.path.basename(str(cmd[0]))} failed: " + " | ".join(tail))
    return r


def hex_to_rgb(s):
    s = (s or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return None
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        return None


def rgb_to_hex(c):
    return "#%02X%02X%02X" % tuple(int(max(0, min(255, v))) for v in c)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def lerp(a, b, t):
    return a + (b - a) * t


def smooth(t):
    """smoothstep ease-in-out on [0,1]."""
    t = clamp(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def ease_out(t):
    t = clamp(t, 0.0, 1.0)
    return 1 - (1 - t) ** 3


def mix_rgb(a, b, t):
    return tuple(int(lerp(a[i], b[i], t)) for i in range(3))


def word_count(s):
    return len(re.findall(r"[\w'’%$€£.\-]+", str(s)))

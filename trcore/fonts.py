"""Font discovery + loading. Drop .ttf/.otf files in ./fonts; variable fonts (Inter) get weight set."""
import glob, os, re, threading

from .util import ROOT

FONT_DIR = os.path.join(ROOT, "fonts")
FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
]
_tl = threading.local()          # FreeType faces are not thread-safe -> one cache per thread


def font_dir():
    os.makedirs(FONT_DIR, exist_ok=True)
    return FONT_DIR


def _family(fname):
    n = os.path.splitext(os.path.basename(fname))[0]
    n = re.sub(r"\[.*?\]", "", n)
    n = re.sub(r"[-_ ]?(Regular|Bold|SemiBold|Medium|Light|Black|ExtraBold|Italic)$", "", n, flags=re.I)
    return n


def list_font_files():
    return sorted(glob.glob(os.path.join(font_dir(), "*.ttf")) + glob.glob(os.path.join(font_dir(), "*.otf")))


def list_families():
    fams = sorted({_family(p) for p in list_font_files()})
    return fams or ["Default"]


def list_fonts():
    """File names (back-compat with the old UI picker)."""
    return [os.path.basename(p) for p in list_font_files()] + ["Default (system)"]


def resolve_font(name):
    """name = family ('Inter'), file name, or absolute path. Falls back to a system font."""
    if name and name not in ("Default", "Default (system)", "Default (Arial)", "auto"):
        if os.path.exists(name):
            return name
        p = os.path.join(font_dir(), name)
        if os.path.exists(p):
            return p
        want = re.sub(r"\W", "", _family(name)).lower()
        for f in list_font_files():
            if re.sub(r"\W", "", _family(f)).lower() == want:
                return f
        for f in list_font_files():
            if want and want in re.sub(r"\W", "", os.path.basename(f)).lower():
                return f
    for f in list_font_files():           # bundled default (Inter) beats other fonts, which beat system fonts
        if _family(f).lower() == "inter":
            return f
    for f in list_font_files():
        return f
    for c in FONT_CANDIDATES:
        if os.path.exists(c):
            return c
    return None


def get_font(size, path=None, weight=700):
    from PIL import ImageFont
    size = max(int(size), 8)
    p = path or resolve_font(None)
    cache = getattr(_tl, "c", None)
    if cache is None:
        cache = _tl.c = {}
    key = (p, size, weight)
    if key in cache:
        return cache[key]
    f = None
    if p:
        try:
            f = ImageFont.truetype(p, size)
            try:
                axes = f.get_variation_axes()
                vals = []
                for a in axes:
                    nm = a["name"].decode() if isinstance(a["name"], bytes) else str(a["name"])
                    nm = nm.lower()
                    if "weight" in nm or nm == "wght":
                        vals.append(max(a["minimum"], min(a["maximum"], weight)))
                    else:
                        vals.append(a["default"])
                if vals:
                    f.set_variation_by_axes(vals)
            except Exception:
                pass
        except Exception:
            f = None
    if f is None:
        try:
            f = ImageFont.load_default(size=size)
        except TypeError:
            f = ImageFont.load_default()
    cache[key] = f
    return f


def render_font_preview(name, sample="Reel 05", width=260, height=72):
    from PIL import Image, ImageDraw
    p = resolve_font(name)
    img = Image.new("RGB", (width, height), (24, 24, 30))
    d = ImageDraw.Draw(img)
    font = get_font(int(height * 0.5), p)
    tw = d.textlength(sample, font=font)
    d.text(((width - tw) / 2, height * 0.16), sample, font=font, fill=(245, 245, 250))
    d.text((8, height - 22), str(name), font=get_font(11, p, 500), fill=(150, 150, 160))
    return img


def wrap_text(draw, text, font, max_w):
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if not cur or draw.textlength(cand, font=font) <= max_w:
            cur = cand
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines

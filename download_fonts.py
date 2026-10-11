"""Install popular Google Fonts (all SIL OFL) into ./fonts so they show up in the Brand Kit font pickers.

Source: raw.githubusercontent.com/google/fonts (no bot-wall).
Usage:  python download_fonts.py            # all
        python download_fonts.py Lilita Sora   # a subset (case-insensitive names)
"""
import os, sys, re, urllib.request, urllib.parse

BASE = "https://raw.githubusercontent.com/google/fonts/main/ofl/{dir}/{file}"
# (display name, repo dir, preferred file). 'file' None = auto-pick best static/variable.
FONTS = [
    ("Inter", "inter", "Inter[opsz,wght].ttf"),
    ("Space Grotesk", "spacegrotesk", "SpaceGrotesk[wght].ttf"),
    ("DM Sans", "dmsans", "DMSans[opsz,wght].ttf"),
    ("JetBrains Mono", "jetbrainsmono", "JetBrainsMono[wght].ttf"),
    ("Playfair Display", "playfairdisplay", "PlayfairDisplay[wght].ttf"),
    ("IBM Plex Sans", "ibmplexsans", "IBMPlexSans[wdth,wght].ttf"),
    ("Lilita One", "lilitaone", "LilitaOne-Regular.ttf"),
    ("Bricolage Grotesque", "bricolagegrotesque", "BricolageGrotesque[opsz,wdth,wght].ttf"),
    ("Outfit", "outfit", "Outfit[wght].ttf"),
    ("Sora", "sora", "Sora[wght].ttf"),
    ("Zilla Slab", "zillaslab", "ZillaSlab-SemiBold.ttf"),
    ("Cairo", "cairo", "Cairo[slnt,wght].ttf"),
    ("Chakra Petch", "chakrapetch", "ChakraPetch-Bold.ttf"),
    ("Manrope", "manrope", "Manrope[wght].ttf"),
]
DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def save_name(fname):
    n = os.path.basename(fname)
    n = re.sub(r"\[.*?\]", "", n)              # strip [wght]/[opsz,wdth,wght]/[slnt,wght]
    n = n.rsplit(".", 1)[0]                     # drop existing .ttf/.otf
    n = re.sub(r"-(Regular|Bold|SemiBold|Medium|Light)$", "", n)
    return n + ".ttf"

def is_font(b):
    return b[:4] in (b"\x00\x01\x00\x00", b"OTTO", b"true", b"typ1") or b[:4] == b"wOFF"

def pick(names):
    if not names:
        return FONTS
    sel = []
    for want in names:
        w = want.lower().replace(" ", "")
        for f in FONTS:
            if w in f[0].lower().replace(" ", "") or w in f[1]:
                sel.append(f)
    return sel

def grab(dir_name, fname):
    url = BASE.format(dir=dir_name, file=urllib.parse.quote(fname, safe=""))
    return urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60).read()

def main():
    os.makedirs(DIR, exist_ok=True)
    done = skip = fail = 0
    for name, d, fname in pick(sys.argv[1:]):
        out = os.path.join(DIR, save_name(fname))
        if os.path.exists(out) and is_font(open(out, "rb").read(16)):
            print(f"skip  {name} (already {os.path.basename(out)})"); skip += 1; continue
        try:
            body = grab(d, fname)
            if len(body) < 8000 or not is_font(body):
                raise RuntimeError(f"not a valid font ({len(body)}B, magic={body[:4]!r})")
            with open(out, "wb") as fh: fh.write(body)
            print(f"ok    {name} -> {os.path.basename(out)} ({len(body)//1024} KB)"); done += 1
        except Exception as e:
            print(f"FAIL  {name}: {e}"); fail += 1
    total = len([x for x in os.listdir(DIR) if x.lower().endswith((".ttf", ".otf")) and is_font(open(os.path.join(DIR, x), "rb").read(16))])
    print(f"\n{done} installed, {skip} already present, {fail} failed.  Valid .ttf in ./fonts: {total}")
    return 1 if fail and not done else 0

if __name__ == "__main__":
    sys.exit(main())

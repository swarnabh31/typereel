"""Frame engine. Pure-Pillow, thread-safe scenes, raw frames piped straight into ffmpeg.

Scenes: title · ui (focus-region camera, cursor, spotlight, glass card) · kpi (count-up)
        photo (Ken-Burns) · line / clip (text beats) · cta.
Every scene exposes frame(k) -> RGB image, with k allowed to run past `total` so transitions
can cross-dissolve two *moving* scenes (no frozen frames, no jump cuts).
"""
import math, os, random, re, subprocess, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .brand import palette_rgb
from .fonts import get_font, resolve_font, wrap_text
from .util import clamp, ease_out, ffmpeg_bin, lerp, log, mix_rgb, run, smooth

RES = Image.BICUBIC


def _lut(scale):
    return [int(i * scale) for i in range(256)]


def fade_alpha(k, total, fps):
    fi = max(int(0.40 * fps), 2); fo = max(int(0.32 * fps), 2)
    fi = min(fi, max(total // 3, 1)); fo = min(fo, max(total // 3, 1))
    a = 1.0
    if k < fi:
        a = k / fi
    elif k >= total - fo:
        a = max((total - 1 - k) / fo, 0.0)
    return smooth(a)


def rounded_mask(size, radius, square_top=0):
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    d.rounded_rectangle([0, 0, size[0] - 1, size[1] - 1], radius=max(radius, 0), fill=255)
    if square_top:
        d.rectangle([0, 0, size[0] - 1, square_top], fill=255)
    return m


def split_rgba(im):
    r, g, b, a = im.split()
    return Image.merge("RGB", (r, g, b)), a


# ====================================================================== world
class World:
    """Everything shared & read-only across scenes/threads."""

    def __init__(self, brand, w, h, fps, bg_image=None, watermark=""):
        self.brand, self.w, self.h, self.fps = brand, w, h, fps
        self.u = min(w, h) / 1080.0
        self.portrait = h >= w
        self.pal = palette_rgb(brand)
        vis = brand["visual"]
        self.vis = vis
        self.mode = vis["background"]
        if self.mode == "ai" and bg_image is None:
            self.mode = "grid"
        self.cam = set(brand["motion"]["camera"])
        self.fonts = {k: resolve_font(vis["fonts"].get(k)) for k in ("heading", "body", "numerals")}
        self.g1 = self._gradient(self.pal["a"], self.pal["b"])
        self.g2 = self._gradient(self.pal["a2"], self.pal["b2"])
        self.solid = Image.new("RGB", (w, h), self.pal["a"])
        self.tint = self.g1
        self.tint_amt = vis["grade"]["tint"]
        self._vig = None
        # parallax grid layer (built once)
        self.pg = int(60 * self.u)
        self.grid_a, self.grid_color = self._build_grid() if self.mode == "grid" else (None, None)
        # AI backdrop (Ken-Burns canvas, cover-fit with headroom)
        self.bg_big = None
        if bg_image is not None and self.mode == "ai":
            s = max(w * 1.25 / bg_image.width, h * 1.25 / bg_image.height)
            self.bg_big = bg_image.convert("RGB").resize((int(bg_image.width * s), int(bg_image.height * s)), Image.LANCZOS)
        self.watermark = None
        if watermark:
            f = get_font(24 * self.u, self.fonts["body"], 500)
            d = ImageDraw.Draw(Image.new("L", (4, 4)))
            tw = int(d.textlength(watermark, font=f)) + 8
            lay = Image.new("RGBA", (tw, int(40 * self.u)), (0, 0, 0, 0))
            ImageDraw.Draw(lay).text((4, 4), watermark, font=f, fill=self.pal["text"] + (120,))
            self.watermark = split_rgba(lay) + ((w - tw) // 2, h - int(78 * self.u))

    def _gradient(self, top, bot):
        col = Image.new("RGB", (1, self.h)); px = col.load()
        for y in range(self.h):
            t = y / max(self.h - 1, 1)
            px[0, y] = tuple(int(top[i] + (bot[i] - top[i]) * t) for i in range(3))
        return col.resize((self.w, self.h))

    def _build_grid(self):
        W2, H2 = self.w + 2 * self.pg, self.h + 2 * self.pg
        step = max(int(self.vis["grid"]["size"] * self.u), 16)
        lines = Image.new("L", (W2, H2), 0)
        d = ImageDraw.Draw(lines)
        for x in range(0, W2, step):
            d.line([(x, 0), (x, H2)], fill=255, width=max(int(self.u), 1))
        for y in range(0, H2, step):
            d.line([(0, y), (W2, y)], fill=255, width=max(int(self.u), 1))
        fade = Image.new("L", (64, 113)); px = fade.load()
        for yy in range(113):
            for xx in range(64):
                r = min(math.hypot((xx - 32) / 32, (yy - 40) / 62), 1.0)
                px[xx, yy] = int(255 * (1 - r ** 1.6))
        fade = fade.resize((W2, H2), Image.BILINEAR)
        alpha = ImageChops.multiply(lines, fade).point(_lut(self.vis["grid"]["opacity"] * 1.0))
        col = mix_rgb(self.pal["accent"], self.pal["text"], 0.45)
        return alpha, Image.new("RGB", (self.w, self.h), col)

    def vignette(self):
        if self._vig is None:
            m = Image.new("L", (80, 142)); px = m.load()
            amt = self.vis["grade"]["vignette"]
            for y in range(142):
                for x in range(80):
                    dd = min(math.hypot((x - 40) / 40, (y - 71) / 71) / 1.25, 1.0)
                    px[x, y] = int(255 * (dd ** 2) * amt)
            self._vig = m.resize((self.w, self.h), Image.BILINEAR)
        return self._vig

    def grade(self, frame):
        if self.tint_amt > 0:
            frame = Image.blend(frame, self.tint, self.tint_amt)
        dark = frame.point(_lut(0.70))
        return Image.composite(dark, frame, self.vignette())


# ====================================================================== background
class Background:
    def __init__(self, W, seed):
        self.W, self.seed = W, seed
        rng = random.Random(seed * 7919 + 13)
        w, h = W.w, W.h
        self.anchor = (rng.uniform(0.25, 0.75) * w, rng.uniform(0.10, 0.45) * h)
        self.drift = (rng.uniform(-40, 40) * W.u, rng.uniform(-60, 60) * W.u)
        self.glow = None
        g = W.vis["glow"]
        if g > 0 and W.mode in ("grid", "gradient"):
            pad = int(0.25 * w)
            gm = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
            r = int(w * 0.55)
            x, y = self.anchor[0] + pad, self.anchor[1] + pad
            ImageDraw.Draw(gm).ellipse([x - r, y - r, x + r, y + r], fill=255)
            gm = gm.filter(ImageFilter.GaussianBlur(int(w * 0.12))).point(_lut(g))
            self.glow, self.glow_pad = gm, pad
            self.glow_img = Image.new("RGB", (w, h), W.pal["accent"])
        self.kb_zoom = 1.0 + 0.04 * (seed % 4)

    def render(self, t, p):
        W = self.W
        w, h = W.w, W.h
        par = "parallax" in W.cam
        if W.mode == "ai" and W.bg_big is not None:
            z = min(self.kb_zoom + 0.10 * p, 1.35)
            bw, bh = W.bg_big.size
            cw = bw / z; ch = cw * h / w
            if ch > bh:
                ch = bh; cw = ch * w / h
            cx = bw / 2 + self.drift[0] * p; cy = bh / 2 + self.drift[1] * p
            x0 = clamp(cx - cw / 2, 0, bw - cw); y0 = clamp(cy - ch / 2, 0, bh - ch)
            im = W.bg_big.transform((w, h), Image.EXTENT, (x0, y0, x0 + cw, y0 + ch), RES)
            return Image.blend(im, W.tint, 0.20)
        if W.mode == "solid":
            im = W.solid.copy()
        else:
            k = 0.5 + 0.5 * math.sin(2 * math.pi * t / 9.0 + self.seed)
            im = Image.blend(W.g1, W.g2, k)
        D = (p - 0.5) * 36 * W.u if par else 0.0
        if self.glow is not None:
            ox = int(self.glow_pad + D * 0.5); oy = int(self.glow_pad + D * 0.3)
            m = self.glow.crop((ox, oy, ox + w, oy + h))
            im.paste(self.glow_img, (0, 0), m)
        if W.grid_a is not None:
            ox = int(W.pg + D * 0.22 + t * 1.2 * W.u); oy = int(W.pg + D * 0.12)
            m = W.grid_a.crop((ox, oy, ox + w, oy + h))
            im.paste(W.grid_color, (0, 0), m)
        return im


# ====================================================================== text
class TextBlock:
    """Pre-rendered kicker + caption + sub-line with optional glass/solid panel."""

    def __init__(self, W, kicker="", caption="", sub="", place="lower", size=1.0, big=False,
                 style=None, align_y=None):
        self.W = W
        cap = W.vis["caption"]
        self.style = style or cap["style"]
        u, w, h = W.u, W.w, W.h
        head, body = W.fonts["heading"], W.fonts["body"]
        meas = ImageDraw.Draw(Image.new("L", (8, 8)))
        max_w = int(w * (0.84 if W.portrait else 0.60))
        fs = (92 if big else 62) * u * size * (cap["size"] if not big else 1.0)
        lines = []
        for _ in range(12):
            font = get_font(fs, head, 700)
            lines = wrap_text(meas, caption, font, max_w) if caption else []
            if len(lines) <= 4 or fs < 30 * u:
                break
            fs *= 0.9
        lines = lines[:5]
        lh = int(fs * 1.2)
        sfs = max(fs * 0.50, 26 * u)
        sfont = get_font(sfs, body, 500)
        slines = wrap_text(meas, sub, sfont, max_w)[:3] if sub else []
        slh = int(sfs * 1.38)
        kfs = 27 * u
        kfont = get_font(kfs, body, 700)
        ktxt = " ".join(kicker.upper()) if kicker else ""
        ktxt = re.sub(r"\s{2,}", "   ", ktxt)
        gap = int(14 * u)
        kh = int(kfs * 1.9) if ktxt else 0
        block_h = kh + lh * len(lines) + ((gap + slh * len(slines)) if slines else 0)
        widest = max([meas.textlength(t, font=font) for t in lines] +
                     [meas.textlength(t, font=sfont) for t in slines] +
                     [meas.textlength(ktxt, font=kfont) if ktxt else 0] + [10])
        px_, py_ = int(48 * u), int(38 * u)
        pw, ph = int(widest + 2 * px_), int(block_h + 2 * py_)
        if align_y is None:
            align_y = 0.5 if (place == "center" or big) else (0.80 if W.portrait else 0.86)
        cy = int(h * align_y)
        x0 = int((w - pw) / 2); y0 = int(cy - ph / 2)
        y0 = int(clamp(y0, 20 * u, h - ph - 20 * u))
        self.rect = (x0, y0, x0 + pw, y0 + ph)
        m = int(36 * u)
        lay = Image.new("RGBA", (pw + 2 * m, ph + 2 * m), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        pal = W.pal
        if self.style in ("glass", "solid"):
            br = int(30 * u)
            if self.style == "glass":
                d.rounded_rectangle([m, m, m + pw, m + ph], radius=br, outline=pal["text"] + (46,), width=max(int(1.6 * u), 1))
                d.line([(m + br, m + 1), (m + pw - br, m + 1)], fill=pal["text"] + (80,), width=max(int(u), 1))
            self.mask = rounded_mask((pw, ph), br)
            self.bgcol = hex_rgb(cap["bg"])
            self.op = cap["opacity"] / 255.0 if self.style == "glass" else min(1.0, cap["opacity"] / 200.0 + 0.2)
        else:
            self.mask = None
            sh = Image.new("RGBA", lay.size, (0, 0, 0, 0))
        shadow_on = (self.style == "none") and not pal["light"]
        ty = m + py_
        tcol, mut, acc = pal["text"], pal["muted"], pal["accent"]
        if ktxt:
            self._text_centered(d, ktxt, kfont, m + pw / 2, ty, acc + (240,), shadow_on, sh if self.mask is None else None)
            ty += kh
        for t in lines:
            self._text_centered(d, t, font, m + pw / 2, ty, tcol + (255,), shadow_on, sh if self.mask is None else None)
            ty += lh
        if slines:
            ty += gap
            for t in slines:
                self._text_centered(d, t, sfont, m + pw / 2, ty, mut + (235,), shadow_on, sh if self.mask is None else None)
                ty += slh
        if self.mask is None and shadow_on:
            sh = sh.filter(ImageFilter.GaussianBlur(int(7 * u)))
            sh.putalpha(sh.getchannel("A").point(_lut(0.85)))
            lay = Image.alpha_composite(sh, lay)
        self.rgb, self.a = split_rgba(lay)
        self.m = m
        self.slide = int(26 * u)

    @staticmethod
    def _text_centered(d, text, font, cx, y, fill, shadow, sh_layer):
        tw = d.textlength(text, font=font)
        if shadow and sh_layer is not None:
            ImageDraw.Draw(sh_layer).text((cx - tw / 2, y + 4), text, font=font, fill=(0, 0, 0, 200))
        d.text((cx - tw / 2, y), text, font=font, fill=fill)

    def draw(self, frame, k, total):
        W = self.W
        e = fade_alpha(k, total, W.fps)
        if e <= 0.004:
            return frame
        dy = int(self.slide * (1 - e))
        x0, y0, x1, y1 = self.rect
        y0 += dy; y1 += dy
        if self.mask is not None:
            reg = frame.crop((x0, y0, x1, y1))
            rw, rh = reg.size
            if self.style == "glass":
                reg = reg.resize((max(rw // 6, 1), max(rh // 6, 1)), Image.BILINEAR).filter(
                    ImageFilter.GaussianBlur(3)).resize((rw, rh), Image.BILINEAR)
            tinted = Image.blend(reg, Image.new("RGB", (rw, rh), self.bgcol), self.op)
            frame.paste(tinted, (x0, y0), self.mask.point(_lut(e)))
        frame.paste(self.rgb, (x0 - self.m, y0 - self.m), self.a.point(_lut(e)))
        return frame


def hex_rgb(h):
    from .util import hex_to_rgb
    return hex_to_rgb(h) or (10, 12, 30)


# ====================================================================== scenes
class _Scene:
    def __init__(self, W, seg, idx):
        self.W, self.seg, self.idx = W, seg, idx
        self.total = max(int(round(seg["secs"] * W.fps)), 4)
        self.bg = Background(W, idx)

    def prog(self, k):
        return clamp(k / max(self.total - 1, 1), 0.0, 1.0)


class TitleScene(_Scene):
    def __init__(self, W, seg, idx, end=False):
        super().__init__(W, seg, idx)
        self.end = end
        self.text = TextBlock(W, seg.get("kicker", ""), seg["text"], seg.get("sub", ""), "center",
                              big=True, style="none", align_y=0.46)
        self.barw = int(150 * W.u)

    def frame(self, k):
        W = self.W
        p = self.prog(k)
        im = self.bg.render(k / W.fps, p)
        im = W.grade(im) if W.mode == "ai" else im
        e = fade_alpha(k, self.total, W.fps)
        if e > 0.01:
            x0, y0, x1, y1 = self.text.rect
            bw = int(self.barw * smooth(min(k / (0.9 * W.fps), 1.0)) * e)
            by = y1 + int(18 * W.u)
            if bw > 2:
                ImageDraw.Draw(im).rounded_rectangle(
                    [W.w // 2 - bw // 2, by, W.w // 2 + bw // 2, by + max(int(5 * W.u), 3)],
                    radius=3, fill=W.pal["accent"])
        return self.text.draw(im, k, self.total)


class LineScene(_Scene):
    def __init__(self, W, seg, idx):
        super().__init__(W, seg, idx)
        place = W.vis["caption"]["position"]
        self.text = TextBlock(W, seg.get("kicker", ""), seg["text"], seg.get("sub", ""), place,
                              size=1.1, align_y=0.5 if place == "center" else None)

    def frame(self, k):
        im = self.bg.render(k / self.W.fps, self.prog(k))
        if self.W.mode == "ai":
            im = self.W.grade(im)
        return self.text.draw(im, k, self.total)


class PhotoScene(_Scene):
    def __init__(self, W, seg, idx, img):
        super().__init__(W, seg, idx)
        self.img = img
        rng = random.Random(idx * 31 + 5)
        self.d = (rng.randint(-30, 30), rng.randint(-40, 40))
        self.text = TextBlock(W, seg.get("kicker", ""), seg["text"], seg.get("sub", ""), "lower")

    def frame(self, k):
        W = self.W
        p = self.prog(k)
        iw, ih = self.img.size
        s = max(W.w / iw, W.h / ih)
        z = 1.06 + (0.10 if "slow_zoom" in W.cam else 0.02) * p
        cw, ch = W.w / (s * z), W.h / (s * z)
        cx = clamp(iw / 2 + self.d[0] * p * 2, cw / 2, iw - cw / 2)
        cy = clamp(ih / 2 + self.d[1] * p * 2, ch / 2, ih - ch / 2)
        im = self.img.transform((W.w, W.h), Image.EXTENT, (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2), RES)
        return self.text.draw(W.grade(im), k, self.total)


_NUM = re.compile(r"^(?P<pre>[^\d\-]*)(?P<num>-?\d[\d,]*\.?\d*)(?P<suf>.*)$")


class KPIScene(_Scene):
    def __init__(self, W, seg, idx):
        super().__init__(W, seg, idx)
        kp = seg["kpi"]
        self.value, self.label = kp["value"], kp.get("label", "")
        m = _NUM.match(self.value)
        self.parts = None
        if m:
            num = m.group("num")
            self.parts = (m.group("pre"), float(num.replace(",", "")), len(num.split(".")[1]) if "." in num else 0,
                          "," in num, m.group("suf"))
        self.text = TextBlock(W, seg.get("kicker", ""), seg["text"], seg.get("sub", ""), "lower")
        self.fs = (250 if W.portrait else 230) * W.u
        r = int(W.w * 0.34)
        gl = Image.new("L", (2 * r, 2 * r), 0)
        ImageDraw.Draw(gl).ellipse([r // 3, r // 3, 2 * r - r // 3, 2 * r - r // 3], fill=255)
        self.glow_a = gl.filter(ImageFilter.GaussianBlur(int(r * 0.35))).point(_lut(0.38))
        self.glow_rgb = Image.new("RGB", gl.size, W.pal["accent"])
        self.r = r

    def _fmt(self, c):
        if not self.parts:
            return self.value
        pre, val, dec, comma, suf = self.parts
        v = val * c
        body = f"{v:,.{dec}f}" if comma else f"{v:.{dec}f}"
        return f"{pre}{body}{suf}"

    def frame(self, k):
        W = self.W
        t = k / W.fps
        p = self.prog(k)
        im = self.bg.render(t, p)
        cy = int(W.h * (0.38 if W.portrait else 0.40))
        e = fade_alpha(k, self.total, W.fps)
        im.paste(self.glow_rgb, (W.w // 2 - self.r, cy - self.r), self.glow_a.point(_lut(e)))
        c = ease_out(clamp(t / min(1.5, 0.55 * self.total / W.fps), 0, 1))
        txt = self._fmt(c) if c < 0.999 else self.value
        font = get_font(self.fs, W.fonts["numerals"], 800)
        d = ImageDraw.Draw(im)
        # fixed-width layout from the final string so digits don't jitter
        full_w = d.textlength(self.value, font=font)
        tw = d.textlength(txt, font=font)
        lay = Image.new("RGBA", (W.w, int(self.fs * 1.6)), (0, 0, 0, 0))
        ld = ImageDraw.Draw(lay)
        y = int(self.fs * 0.15)
        ld.text(((W.w - tw) / 2, y + int(6 * W.u)), txt, font=font, fill=(0, 0, 0, 120))
        ld.text(((W.w - tw) / 2, y), txt, font=font, fill=W.pal["text"] + (255,))
        if self.label:
            lf = get_font(46 * W.u, W.fonts["body"], 600)
            lw = ld.textlength(self.label.upper(), font=lf)
            ld.text(((W.w - lw) / 2, y + self.fs * 1.17), self.label.upper(), font=lf, fill=W.pal["muted"] + (255,))
        bar = int(120 * W.u * smooth(min(t / 1.0, 1)))
        ld.rounded_rectangle([W.w // 2 - bar // 2, y + int(self.fs * 1.02), W.w // 2 + bar // 2, y + int(self.fs * 1.02) + max(int(5 * W.u), 3)],
                             radius=3, fill=W.pal["accent"] + (255,))
        rgb, a = split_rgba(lay)
        im.paste(rgb, (0, cy - int(self.fs * 0.75)), a.point(_lut(e)))
        return self.text.draw(im, k, self.total)


def _cursor_sprite(u):
    s = 4
    sc = 1.55 * u * s
    pts = [(0, 0), (0, 17), (4.6, 13.2), (7.6, 20.2), (10.2, 19.1), (7.3, 12.4), (13, 12.4)]
    pts = [(x * sc, y * sc) for x, y in pts]
    W_, H_ = int(15 * sc) + 6, int(22 * sc) + 6
    im = Image.new("RGBA", (W_, H_), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.polygon([(x + 3, y + 3) for x, y in pts], fill=(255, 255, 255, 255), outline=(10, 10, 14, 255))
    d.line([(x + 3, y + 3) for x, y in pts + [pts[0]]], fill=(10, 10, 14, 255), width=int(1.4 * s))
    return im.resize((max(W_ // s, 4), max(H_ // s, 4)), Image.LANCZOS)


class UIScene(_Scene):
    """Screenshot card with focus-region camera, eased cursor, click ripple, ring and spotlight."""

    def __init__(self, W, seg, idx, img):
        super().__init__(W, seg, idx)
        brand = W.brand
        card = W.vis["card"]
        u, w, h = W.u, W.w, W.h
        self.focus = seg.get("focus")
        self.cursor_on = brand["motion"]["cursor"] and bool(self.focus)
        self.spot_on = brand["motion"]["spotlight"] and bool(self.focus)
        self.pan_on = "ui_pan" in W.cam
        self.zoom_on = "slow_zoom" in W.cam
        ar = img.width / img.height
        style = card["style"]
        has_bar = style != "bare" and (card["chrome"] == "browser" or (card["chrome"] == "auto" and ar > 1.15))
        self.bar = int(38 * u) if has_bar else 0
        self.pad = int(12 * u) if style != "bare" else 0
        self.text = TextBlock(W, seg.get("kicker", ""), seg["text"], seg.get("sub", ""), "lower")
        top_m, gap = int(0.06 * h), int(0.035 * h)
        avail = max(self.text.rect[1] - gap - top_m, int(0.25 * h))        # room above the caption panel
        box_w = int(w * (0.88 if W.portrait else 0.64)) - 2 * self.pad
        box_h = min(int(h * (0.50 if W.portrait else 0.60)), avail) - 2 * self.pad - self.bar
        # card aspect: follow the screenshot, but never thinner than 1.3:1 in portrait, so very wide UIs
        # become a bigger card the camera pans across instead of a thin strip.
        A = clamp(ar, 0.75, 1.3 if W.portrait else 1.9)
        cw = box_w; ch = int(cw / A)
        if ch > box_h:
            ch = box_h; cw = int(ch * A)
        self.cw, self.ch = max(cw // 2 * 2, 64), max(ch // 2 * 2, 64)
        self.A = self.cw / self.ch
        if ar > self.A:                       # base view = full height, centred crop of the width
            bw0, bh0 = img.height * self.A, img.height
        else:
            bw0, bh0 = img.width, img.width / self.A
        s_ = min(1.0, self.cw * 3 / bw0)
        self.src = img if s_ >= 0.999 else img.resize((max(int(img.width * s_), 1), max(int(img.height * s_), 1)), Image.LANCZOS)
        self.bw, self.bh = bw0 * s_, bh0 * s_
        self.card_w = self.cw + 2 * self.pad
        self.card_h = self.ch + self.bar + 2 * self.pad
        self.cx0 = w // 2
        self.cy0 = int(top_m + avail / 2)
        rad = int(card["radius"] * u)
        self.rad = rad
        # frame (glass / solid) with browser chrome
        fr = Image.new("RGBA", (self.card_w, self.card_h), (0, 0, 0, 0))
        fd = ImageDraw.Draw(fr)
        if style == "glass":
            fd.rounded_rectangle([0, 0, self.card_w - 1, self.card_h - 1], radius=rad + self.pad // 2, fill=(255, 255, 255, 30),
                                 outline=(255, 255, 255, 95), width=max(int(1.6 * u), 1))
        elif style == "solid":
            fd.rounded_rectangle([0, 0, self.card_w - 1, self.card_h - 1], radius=rad + self.pad // 2, fill=(12, 12, 18, 255))
        if self.bar:
            for i, col in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
                x = self.pad + int(16 * u) + i * int(22 * u)
                cy_ = self.pad + self.bar // 2
                r = int(5.5 * u)
                fd.ellipse([x - r, cy_ - r, x + r, cy_ + r], fill=col + (235,))
        self.fr_rgb, self.fr_a = split_rgba(fr)
        self.mask = rounded_mask((self.cw, self.ch), max(rad - self.pad // 2, 2), square_top=(rad if self.bar else 0))
        # shadow + glow
        gm = int(70 * u)
        sh = Image.new("RGBA", (self.card_w + 2 * gm, self.card_h + 2 * gm), (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle([gm, gm, gm + self.card_w, gm + self.card_h], radius=rad + self.pad, fill=(0, 0, 0, 165))
        sh = sh.filter(ImageFilter.GaussianBlur(int(26 * u)))
        self.sh_rgb, self.sh_a = split_rgba(sh)
        self.gm = gm
        self.glow = None
        if card["glow"] > 0:
            gl = Image.new("L", sh.size, 0)
            ImageDraw.Draw(gl).rounded_rectangle([gm - int(8 * u), gm - int(8 * u), gm + self.card_w + int(8 * u), gm + self.card_h + int(8 * u)],
                                                 radius=rad + self.pad + int(8 * u), fill=255)
            gl = gl.filter(ImageFilter.GaussianBlur(int(34 * u))).point(_lut(card["glow"] * 0.55))
            self.glow = (Image.new("RGB", gl.size, W.pal["accent"]), gl)
        self.sprite = _cursor_sprite(u) if self.cursor_on else None
        self.rng = random.Random(idx * 97 + 3)
        # camera targets
        if self.focus:
            fx, fy, fw, fh = self.focus
            self.tx, self.ty = fx + fw / 2, fy + fh / 2
            sw_, sh_ = self.src.size
            self.zt = clamp(min(self.bw / max(fw * sw_ * 1.35, 1), self.bh / max(fh * sh_ * 1.35, 1)), 1.12, 2.6) if self.pan_on else 1.06
            ang = self.rng.uniform(0, 2 * math.pi)
            self.start = (clamp(self.tx + 0.30 * math.cos(ang), 0.05, 0.95), clamp(self.ty + 0.28 * abs(math.sin(ang)) + 0.10, 0.08, 0.95))
            self.ctrl = ((self.start[0] + self.tx) / 2 + self.rng.uniform(-0.12, 0.12), (self.start[1] + self.ty) / 2 + self.rng.uniform(-0.14, 0.04))
        else:
            self.tx, self.ty, self.zt = 0.5, 0.5, 1.07
            self.d = (self.rng.uniform(-0.05, 0.05), self.rng.uniform(-0.04, 0.04))

    # ---- camera
    def _cam(self, p):
        ph = smooth((p - 0.10) / 0.50) if self.focus else p
        if self.focus:
            zoom = lerp(1.0, self.zt, ph if self.pan_on else ph * 0.5) + (0.05 * max(p - 0.62, 0) / 0.38 if self.zoom_on else 0)
            cx, cy = lerp(0.5, self.tx, ph), lerp(0.5, self.ty, ph)
        else:
            zoom = 1.0 + (0.07 * p if self.zoom_on else 0.0)
            cx, cy = 0.5 + self.d[0] * p, 0.5 + self.d[1] * p
        sw, sh = self.src.size
        cw_s, ch_s = self.bw / zoom, self.bh / zoom
        x0 = clamp(cx * sw - cw_s / 2, 0, sw - cw_s); y0 = clamp(cy * sh - ch_s / 2, 0, sh - ch_s)
        return zoom, (x0, y0, x0 + cw_s, y0 + ch_s)

    def _to_px(self, nx, ny, box):
        sw, sh = self.src.size
        x0, y0, x1, y1 = box
        return (nx * sw - x0) / (x1 - x0) * self.cw, (ny * sh - y0) / (y1 - y0) * self.ch

    def _content(self, p, t):
        W = self.W
        u = W.u
        zoom, box = self._cam(p)
        content = self.src.transform((self.cw, self.ch), Image.EXTENT, box, RES)
        if not self.focus:
            return content
        fx, fy, fw, fh = self.focus
        rx0, ry0 = self._to_px(fx, fy, box)
        rx1, ry1 = self._to_px(fx + fw, fy + fh, box)
        pad = 10 * u
        rect = [rx0 - pad, ry0 - pad, rx1 + pad, ry1 + pad]
        if self.spot_on:
            s = smooth((p - 0.40) / 0.25) * 0.42
            if s > 0.01:
                sm = Image.new("L", (self.cw // 8 + 1, self.ch // 8 + 1), int(255 * s))
                ImageDraw.Draw(sm).rounded_rectangle([v / 8 for v in rect], radius=int(14 * u / 8) + 1, fill=0)
                sm = sm.filter(ImageFilter.GaussianBlur(2)).resize((self.cw, self.ch), Image.BILINEAR)
                content.paste((4, 5, 14), (0, 0), sm)
        ov = Image.new("RGBA", (self.cw, self.ch), (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        acc = W.pal["accent"]
        ring = smooth((p - 0.56) / 0.10)
        if ring > 0.01:
            od.rounded_rectangle(rect, radius=int(14 * u), fill=acc + (int(28 * ring),),
                                 outline=acc + (int(235 * ring),), width=max(int(3 * u), 2))
        if self.cursor_on and self.sprite is not None:
            c = smooth((p - 0.10) / 0.42)
            sx, sy = self.start; cx_, cy_ = self.ctrl
            bx = (1 - c) ** 2 * sx + 2 * (1 - c) * c * cx_ + c * c * self.tx
            by = (1 - c) ** 2 * sy + 2 * (1 - c) * c * cy_ + c * c * self.ty
            px, py = self._to_px(bx, by, box)
            # click ripple
            rp = (p - 0.54) / 0.20
            if 0 <= rp <= 1:
                rr = (8 + 40 * ease_out(rp)) * u
                od.ellipse([px - rr, py - rr, px + rr, py + rr], outline=(255, 255, 255, int(200 * (1 - rp))), width=max(int(3 * u), 2))
            ca = smooth((p - 0.05) / 0.07)
            if ca > 0.02:
                px_i = int(clamp(px, 0, self.cw - 4)); py_i = int(clamp(py, 0, self.ch - 4))
                spr = self.sprite
                if ca < 0.99:
                    r_, g_, b_, a_ = spr.split()
                    spr = Image.merge("RGBA", (r_, g_, b_, a_.point(_lut(ca))))
                ov.alpha_composite(spr, (px_i, py_i))
        return Image.alpha_composite(content.convert("RGBA"), ov).convert("RGB")

    def frame(self, k):
        W = self.W
        u = W.u
        t = k / W.fps
        p = self.prog(k)
        im = self.bg.render(t, p)
        ent = smooth(min(t / 0.8, 1.0))
        par = "parallax" in W.cam
        D = (p - 0.5) * 36 * u if par else 0.0
        cx = self.cx0 + int(D * 0.45)
        cy = self.cy0 + int((1 - ent) * 26 * u) + int(D * 0.22)
        x0, y0 = cx - self.card_w // 2, cy - self.card_h // 2
        ea = fade_alpha(k, self.total, W.fps)
        card_a = max(ent, 0.0) * (1.0 if k < self.total - int(0.3 * W.fps) else ea)
        lut = _lut(card_a)
        if self.glow is not None:
            im.paste(self.glow[0], (x0 - self.gm, y0 - self.gm), self.glow[1].point(lut))
        im.paste(self.sh_rgb, (x0 - self.gm, y0 - self.gm + int(14 * u)), self.sh_a.point(lut))
        if self.fr_a.getbbox():
            im.paste(self.fr_rgb, (x0, y0), self.fr_a.point(lut))
        content = self._content(p, t)
        m = self.mask.point(lut) if card_a < 0.999 else self.mask
        im.paste(content, (x0 + self.pad, y0 + self.pad + self.bar), m)
        return self.text.draw(im, k, self.total)


# ====================================================================== timeline + encode
def build_scenes(W, segs, images):
    """segs from storyboard.build_segments; images = {index: PIL RGB}."""
    scenes = []
    for i, s in enumerate(segs):
        kind = s["kind"]
        if kind == "title":
            sc = TitleScene(W, s, i)
        elif kind == "cta":
            sc = TitleScene(W, s, i, end=True)
        elif kind == "ui" and s.get("img") in images:
            sc = UIScene(W, s, i, images[s["img"]])
        elif kind == "kpi" and s.get("kpi"):
            sc = KPIScene(W, s, i)
        elif kind == "photo" and s.get("img") in images:
            sc = PhotoScene(W, s, i, images[s["img"]])
        else:
            sc = LineScene(W, s, i)
        scenes.append(sc)
    return scenes


class Timeline:
    def __init__(self, W, scenes):
        self.W, self.scenes = W, scenes
        m = W.brand["motion"]
        self.kind = m["transition"]
        self.xf = max(2, int(m["transition_sec"] * W.fps))
        self.total = sum(s.total for s in scenes)
        self.black = Image.new("RGB", (W.w, W.h), W.pal["a"])

    def jobs(self):
        return [(si, k) for si, s in enumerate(self.scenes) for k in range(s.total)]

    def frame_at(self, job, gi):
        si, k = job
        sc = self.scenes[si]
        im = sc.frame(k)
        xf = min(self.xf, sc.total // 2, (self.scenes[si - 1].total // 2) if si else 99)
        if si > 0 and k < xf:
            prev = self.scenes[si - 1]
            pim = prev.frame(prev.total + k)
            t = smooth((k + 1) / (xf + 1))
            if self.kind == "slide":
                canvas = Image.new("RGB", im.size)
                off = int(self.W.w * t)
                canvas.paste(pim, (-off, 0)); canvas.paste(im, (self.W.w - off, 0))
                im = canvas
            else:
                im = Image.blend(pim, im, t)
        fin = int(0.35 * self.W.fps)
        if gi < fin:
            im = Image.blend(self.black, im, smooth(gi / fin))
        elif gi >= self.total - fin:
            im = Image.blend(self.black, im, smooth((self.total - 1 - gi) / fin))
        if self.W.watermark:
            rgb, a, x, y = self.W.watermark
            im.paste(rgb, (x, y), a)
        return im


def _encoders(ff):
    try:
        out = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20)
        return {l.split()[1] for l in out.stdout.splitlines() if l.startswith(" V")}
    except Exception:
        return set()


def video_codec(ff, final=True):
    enc = _encoders(ff)
    if "libx264" in enc:
        return ["-c:v", "libx264", "-preset", "medium" if final else "veryfast", "-crf", "18" if final else "23"]
    if "libopenh264" in enc:
        return ["-c:v", "libopenh264", "-b:v", "8M" if final else "4M"]
    if "h264_mf" in enc:
        return ["-c:v", "h264_mf", "-b:v", "8M" if final else "4M"]
    if "h264_videotoolbox" in enc:
        return ["-c:v", "h264_videotoolbox", "-b:v", "8M"]
    if "hevc_mf" in enc:
        return ["-c:v", "hevc_mf", "-b:v", "6M"]
    raise RuntimeError("no supported H.264 encoder in ffmpeg (need libx264, libopenh264 or h264_mf)")


def encode(tl, audio_path, out_path, final=True, loudness=-16.0, workers=None):
    """Render all frames in a thread pool and stream them (raw RGB) into one ffmpeg process."""
    ff = ffmpeg_bin()
    if not ff:
        raise RuntimeError("ffmpeg not found (put it on PATH or in ./ffmpeg/bin/)")
    W = tl.W
    total_s = tl.total / W.fps
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W.w}x{W.h}", "-r", str(W.fps), "-i", "-"]
    if audio_path:
        cmd += ["-i", audio_path, "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k",
                "-af", f"loudnorm=I={loudness:.0f}:TP=-1.5:LRA=11,apad", "-ar", "44100"]
    cmd += video_codec(ff, final) + ["-pix_fmt", "yuv420p", "-t", f"{total_s:.3f}", "-movflags", "+faststart", out_path]
    errf = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=errf)
    jobs = tl.jobs()
    n = len(jobs)
    workers = workers or max(os.cpu_count() or 2, 2)
    batch = workers * 3
    t0 = time.time()
    last_pct = -1
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for b0 in range(0, n, batch):
                chunk = jobs[b0:b0 + batch]
                frames = list(ex.map(lambda a: tl.frame_at(a[1], b0 + a[0]).tobytes(), list(enumerate(chunk))))
                for fb in frames:
                    proc.stdin.write(fb)
                pct = int(100 * min(b0 + batch, n) / n)
                if pct // 10 != last_pct // 10:
                    log("render", f"{min(b0 + batch, n)}/{n} frames ({pct}%)")
                    last_pct = pct
        proc.stdin.close()
    except BrokenPipeError:
        pass
    rc = proc.wait()
    if rc != 0:
        errf.seek(0)
        raise RuntimeError("ffmpeg failed: " + errf.read().decode(errors="ignore")[-400:])
    log("time", f"render+encode {time.time() - t0:.1f}s ({n} frames, {workers} threads)")
    return out_path

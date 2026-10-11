"""TypeReel Studio — Streamlit cockpit. 4 steps: Brief -> Brand & Style -> Storyboard -> Render.
All heavy work runs through the CLI (single source of truth); this file only collects choices."""
import base64, copy, glob, hashlib, io, json, os, re, subprocess, sys, time

import streamlit as st

from trcore import APP, VERSION
from trcore import audio as A
from trcore import brand as B
from trcore.fonts import list_families
from trcore.storyboard import build_segments, normalize
from trcore.util import OUT_DIR, ROOT, STATE_DIR

CLI = os.path.join(ROOT, "typereel.py")
JOBS = os.path.join(STATE_DIR, "jobs")
IMG_DIR = os.path.join(STATE_DIR, "images")
for d in (JOBS, IMG_DIR, OUT_DIR):
    os.makedirs(d, exist_ok=True)

st.set_page_config(page_title=f"{APP} Studio", page_icon="🎬", layout="wide")
st.markdown("""<style>
.block-container{max-width:1240px;padding-top:4.5rem}
.tr-title{font-size:2.1rem;font-weight:800;letter-spacing:-.02em;margin:0;background:linear-gradient(90deg,#7C5CFF,#C084FC);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.tr-sub{color:#8a8a99;margin:.1rem 0 1rem}
.tr-step{padding:.35rem .8rem;border-radius:999px;font-size:.85rem;border:1px solid #3a3a4a;color:#9a9aaa}
.tr-step.on{background:#7C5CFF;color:white;border-color:#7C5CFF}
.tr-swatch{height:44px;border-radius:10px;margin-bottom:6px}
</style>""", unsafe_allow_html=True)

KINDS = {"💡 An idea": "idea", "🖥️ App screenshots": "screens", "📷 Photos": "photos", "🎙️ Podcast / video": "podcast"}
SCENES = ["ui", "kpi", "photo", "line", "clip"]
TEXT_MODES = ["quotes", "facts", "flashcards", "story", "hottake"]
PLATFORMS = {"Reels / Shorts / TikTok (9:16)": "9:16", "LinkedIn / YouTube / Web (16:9)": "16:9",
             "Square feed (1:1)": "1:1", "Instagram portrait (4:5)": "4:5"}
fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", lambda f: f)


# ------------------------------------------------------------------ state
def init():
    ss = st.session_state
    ss.setdefault("step", 1)
    ss.setdefault("kind_sel", "💡 An idea")      # plain state, NOT a widget key (widget keys vanish when off-screen)
    ss.setdefault("images", [])
    ss.setdefault("source", None)
    ss.setdefault("music_file", None)
    ss.setdefault("nonce", 0)
    ss.setdefault("up_nonce", 0)
    ss.setdefault("preset", "engineering-saas")
    ss.setdefault("base", B.load_brand("engineering-saas"))
    ss.setdefault("brand", copy.deepcopy(ss["base"]))
    ss.setdefault("sb", None)
    ss.setdefault("result", None)
    ss.setdefault("models", {"writer": "", "vision": ""})
    ss.setdefault("paint", {"engine": "auto", "steps": 0, "quant": "auto", "res": "768x1344", "lock": False})


def bk(name):
    return f"{name}_{st.session_state.nonce}"


def set_brand_base(brand):
    st.session_state.base = B.validate_brand(brand)
    st.session_state.brand = copy.deepcopy(st.session_state.base)
    st.session_state.nonce += 1


def go(step):
    ss = st.session_state
    if ss.step == 2 and step != 2:                      # widgets of step 2 are discarded when it is left:
        ss.base = B.validate_brand(copy.deepcopy(ss.brand))   # re-seed them from the edited brand
        ss.nonce += 1
    ss.step = step
    st.rerun()


# ------------------------------------------------------------------ CLI runner
STAGE = {"models": .03, "facts": .08, "vision": .18, "director": .35, "script": .35, "miner": .35, "stt": .2,
         "lint": .55, "plan": .06, "vo": .12, "music": .2, "audio": .22, "art": .3, "assets": .35, "encode": .4,
         "memory": .02, "cast": .25}


def run_cli(args, label):
    cmd = [sys.executable, "-u", CLI] + args
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    bar = st.progress(0.0, text=label)
    box = st.empty()
    lines, frac = [], 0.0
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env=env, bufsize=1)
    for raw in p.stdout:
        line = raw.rstrip()
        if not line:
            continue
        lines.append(line)
        m = re.match(r"\s*\[(\w+)\]\s*(.*)", line)
        if m:
            tag, msg = m.group(1), m.group(2)
            pm = re.search(r"\((\d+)%\)", msg)
            if tag == "render" and pm:
                frac = 0.45 + 0.5 * int(pm.group(1)) / 100
            elif tag in STAGE:
                frac = max(frac, STAGE[tag])
            bar.progress(min(frac, 0.99), text=f"{label} — {msg[:90]}")
        box.code("\n".join(lines[-12:]), language=None)
    rc = p.wait()
    bar.progress(1.0 if rc == 0 else min(frac, 0.99), text=label + (" — done" if rc == 0 else " — failed"))
    return rc, lines


def grab(lines, key):
    for l in lines:
        if l.strip().startswith(key):
            return l.split(":", 1)[1].strip()
    return None


# ------------------------------------------------------------------ images
def save_upload(f):
    data = f.getvalue()
    h = hashlib.sha1(data).hexdigest()[:10]
    path = os.path.join(IMG_DIR, f"{h}-{re.sub(r'[^A-Za-z0-9._-]', '_', f.name)}")
    if not os.path.exists(path):
        with open(path, "wb") as fh:
            fh.write(data)
    return path


@st.cache_data(show_spinner=False, max_entries=200)
def thumb(path, px=360):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((px, px))
    buf = io.BytesIO(); im.save(buf, format="JPEG", quality=82)
    return buf.getvalue()


def thumb_uri(path):
    try:
        return "data:image/jpeg;base64," + base64.b64encode(thumb(path, 160)).decode()
    except Exception:
        return None


def rerun_frag():
    try:
        st.rerun(scope="fragment")
    except Exception:
        st.rerun()


@fragment
def image_grid():
    imgs = st.session_state.images
    if not imgs:
        return
    st.caption("Story order = left→right, top→bottom. Use ◀ ▶ to reorder, ✕ to remove, 🔍 to enlarge.")
    cols_n = 5
    for r in range(0, len(imgs), cols_n):
        cols = st.columns(cols_n)
        for j, col in enumerate(cols):
            i = r + j
            if i >= len(imgs):
                break
            with col, st.container(border=True):
                st.image(thumb(imgs[i]), width="stretch")
                st.caption(f"**{i + 1}.** {os.path.basename(imgs[i]).split('-', 1)[-1][:22]}")
                b1, b2, b3, b4 = st.columns(4)
                if b1.button("◀", key=f"l{i}", disabled=i == 0, help="move earlier"):
                    imgs[i - 1], imgs[i] = imgs[i], imgs[i - 1]; rerun_frag()
                if b2.button("▶", key=f"r{i}", disabled=i == len(imgs) - 1, help="move later"):
                    imgs[i + 1], imgs[i] = imgs[i], imgs[i + 1]; rerun_frag()
                if b3.button("✕", key=f"x{i}", help="remove"):
                    imgs.pop(i); rerun_frag()
                try:
                    with b4.popover("🔍"):
                        st.image(imgs[i], width="stretch")
                except Exception:
                    pass


# ------------------------------------------------------------------ brand editor
def apply_classic(name):
    pal = B.LEGACY_PALETTES[name]
    ss = st.session_state
    ss[bk("c_bg0")], ss[bk("c_bg1")] = pal["bg"]
    ss[bk("c_bg20")], ss[bk("c_bg21")] = pal["bg2"]
    ss[bk("c_acc")], ss[bk("c_txt")] = pal["accent"], pal["text"]


def lines(s):
    return [x.strip() for x in re.split(r"[\n,]", s) if x.strip()]


def brand_editor(base):
    b = copy.deepcopy(base)
    i, v, vis, mo, au = b["identity"], b["voice"], b["visual"], b["motion"], b["audio"]
    with st.expander("🏷️ Identity", expanded=False):
        c1, c2 = st.columns(2)
        i["name"] = c1.text_input("Brand / product name", i["name"], key=bk("i_name"))
        i["tagline"] = c2.text_input("Tagline", i["tagline"], key=bk("i_tag"))
        c3, c4, c5 = st.columns(3)
        i["cta"] = c3.text_input("Call to action", i["cta"], key=bk("i_cta"))
        i["website"] = c4.text_input("Website / handle (shown on end card)", i["website"], key=bk("i_web"))
        i["watermark"] = c5.text_input("Footer watermark (optional)", i["watermark"], key=bk("i_wm"))
    with st.expander("🗣️ Voice & tone — this is what the AI is told to write like", expanded=True):
        v["tone"] = lines(st.text_input("Tone keywords (comma-separated)", ", ".join(v["tone"]), key=bk("v_tone")))
        v["person"] = st.text_input("Narration style", v["person"], key=bk("v_person"))
        v["banned"] = lines(st.text_area("Never use (words / characters, comma or newline separated)", ", ".join(v["banned"]), height=70, key=bk("v_ban")))
        v["style_rules"] = lines(st.text_area("Style rules (one per line)", "\n".join(v["style_rules"]), height=110, key=bk("v_rules")))
        v["examples"] = lines(st.text_area("Reference lines the AI should match in register (one per line)", "\n".join(v["examples"]), height=90, key=bk("v_ex")))
        c1, c2, c3 = st.columns(3)
        v["max_caption_words"] = c1.slider("Max caption words", 3, 14, v["max_caption_words"], key=bk("v_mc"))
        v["max_sub_words"] = c2.slider("Max sub-line words", 4, 24, v["max_sub_words"], key=bk("v_ms"))
        v["language"] = c3.text_input("Language", v["language"], key=bk("v_lang"))
    with st.expander("🎨 Visual language", expanded=False):
        pal = vis["palette"]
        st.caption("Quick start from a classic palette:")
        pc = st.columns(len(B.LEGACY_PALETTES))
        for col, name in zip(pc, B.LEGACY_PALETTES):
            col.button(name, key=bk("cl_" + name), on_click=apply_classic, args=(name,), width="stretch")
        c = st.columns(6)
        pal["bg"][0] = c[0].color_picker("Bg top", pal["bg"][0], key=bk("c_bg0"))
        pal["bg"][1] = c[1].color_picker("Bg bottom", pal["bg"][1], key=bk("c_bg1"))
        pal["bg2"][0] = c[2].color_picker("Alt top", pal["bg2"][0], key=bk("c_bg20"))
        pal["bg2"][1] = c[3].color_picker("Alt bottom", pal["bg2"][1], key=bk("c_bg21"))
        pal["accent"] = c[4].color_picker("Accent", pal["accent"], key=bk("c_acc"))
        pal["text"] = c[5].color_picker("Text", pal["text"], key=bk("c_txt"))
        c1, c2, c3 = st.columns(3)
        bgs = ["grid", "gradient", "ai", "solid"]
        vis["background"] = c1.selectbox("Backdrop", bgs, bgs.index(vis["background"]), key=bk("bgmode"),
                                         help="grid = subtle engineering grid + glow · ai = painted by a local diffusion model (optional install)")
        vis["glow"] = c2.slider("Accent glow", 0.0, 1.0, vis["glow"], 0.05, key=bk("glow"))
        vis["grade"]["vignette"] = c3.slider("Vignette", 0.0, 1.0, vis["grade"]["vignette"], 0.05, key=bk("vig"))
        if vis["background"] == "grid":
            c1, c2 = st.columns(2)
            vis["grid"]["size"] = c1.slider("Grid size", 24, 160, vis["grid"]["size"], key=bk("gsize"))
            vis["grid"]["opacity"] = c2.slider("Grid strength", 0.0, 0.4, vis["grid"]["opacity"], 0.01, key=bk("gop"))
        if vis["background"] == "ai":
            vis["art_style"] = st.text_area("AI backdrop art direction", vis["art_style"], height=70, key=bk("art"))
        st.markdown("**Cards (screenshots)**")
        c1, c2, c3, c4 = st.columns(4)
        cs = ["glass", "solid", "bare"]; ch = ["auto", "browser", "none"]
        vis["card"]["style"] = c1.selectbox("Card style", cs, cs.index(vis["card"]["style"]), key=bk("cs"))
        vis["card"]["radius"] = c2.slider("Corner radius", 0, 80, vis["card"]["radius"], key=bk("cr"))
        vis["card"]["glow"] = c3.slider("Card glow", 0.0, 1.0, vis["card"]["glow"], 0.05, key=bk("cg"))
        vis["card"]["chrome"] = c4.selectbox("Window bar", ch, ch.index(vis["card"]["chrome"]), key=bk("cc"))
        st.markdown("**Captions**")
        c1, c2, c3, c4 = st.columns(4)
        ps = ["glass", "solid", "none"]; pos = ["lower", "center"]
        vis["caption"]["style"] = c1.selectbox("Panel", ps, ps.index(vis["caption"]["style"]), key=bk("ps"))
        vis["caption"]["opacity"] = c2.slider("Panel opacity", 0, 255, vis["caption"]["opacity"], key=bk("po"))
        vis["caption"]["size"] = c3.slider("Text size", 0.6, 1.6, vis["caption"]["size"], 0.05, key=bk("psz"))
        vis["caption"]["position"] = c4.selectbox("Position", pos, pos.index(vis["caption"]["position"]), key=bk("ppos"))
        fams = list_families()
        def pick(label, cur, key):
            opts = fams if cur in fams else fams + [cur]
            return st.selectbox(label, opts, opts.index(cur), key=bk(key))
        c1, c2, c3 = st.columns(3)
        with c1: vis["fonts"]["heading"] = pick("Heading font", vis["fonts"]["heading"], "fh")
        with c2: vis["fonts"]["body"] = pick("Body font", vis["fonts"]["body"], "fb")
        with c3: vis["fonts"]["numerals"] = pick("KPI numerals font", vis["fonts"]["numerals"], "fn")
        st.caption("More fonts: sidebar → Engine & tools → Download Google Fonts (or drop .ttf into ./fonts).")
    with st.expander("🎬 Motion", expanded=False):
        cams = {"slow_zoom": "Slow zoom", "ui_pan": "Pan across UI (focus regions)", "parallax": "Gentle parallax"}
        sel = st.multiselect("Camera", list(cams), [x for x in mo["camera"] if x in cams], format_func=cams.get, key=bk("cam"))
        mo["camera"] = sel
        c1, c2 = st.columns(2)
        mo["cursor"] = c1.checkbox("Cursor-driven reveals (animated pointer + click)", mo["cursor"], key=bk("cur"))
        mo["spotlight"] = c2.checkbox("Spotlight the focus region", mo["spotlight"], key=bk("spot"))
        c1, c2, c3 = st.columns(3)
        tr = ["dissolve", "slide"]; pc_ = ["calm", "standard", "snappy"]
        mo["transition"] = c1.selectbox("Transition", tr, tr.index(mo["transition"]), key=bk("tr"), help="Both are true cross-transitions — never a jump cut.")
        mo["transition_sec"] = c2.slider("Transition length (s)", 0.1, 1.2, mo["transition_sec"], 0.05, key=bk("trs"))
        mo["pace"] = c3.selectbox("Pacing", pc_, pc_.index(mo["pace"]), key=bk("pace"))
    with st.expander("🔊 Audio", expanded=False):
        ms = ["lofi-ambient", "minimal-pulse", "cinematic-drone", "none"]
        c1, c2, c3 = st.columns(3)
        au["music"] = c1.selectbox("Music (synthesized locally)", ms, ms.index(au["music"]), key=bk("mus"))
        au["bpm"] = c2.slider("Tempo (BPM)", 60, 140, au["bpm"], key=bk("bpm"))
        au["build"] = c3.checkbox("Build subtly toward the end", au["build"], key=bk("build"))
        c1, c2 = st.columns(2)
        au["music_volume"] = c1.slider("Music level", 0.0, 1.0, au["music_volume"], 0.05, key=bk("mv"))
        up = c2.file_uploader("…or use your own track (mp3/wav)", type=["mp3", "wav", "m4a", "aac", "flac"], key=bk("musup"))
        if up is not None:
            p = os.path.join(STATE_DIR, "music-" + re.sub(r"[^A-Za-z0-9._-]", "_", up.name))
            open(p, "wb").write(up.getvalue()); st.session_state.music_file = p
        if st.session_state.music_file:
            c2.caption(f"Using: {os.path.basename(st.session_state.music_file)}")
            if c2.button("Remove custom track", key=bk("rmmus")):
                st.session_state.music_file = None; st.rerun()
        st.markdown("**Voice-over**")
        au["voiceover"] = st.checkbox("Narrate every beat (local text-to-speech)", au["voiceover"], key=bk("vo"))
        backend = A.tts_backend()
        if au["voiceover"] and not backend:
            st.warning("No local TTS found. Install one: `pip install kokoro soundfile` (best), or Piper / pyttsx3. The video renders without voice until then.")
        elif backend:
            st.caption(f"TTS engine: **{backend}**")
        vs = list(A.VOICE_PRESETS)
        c1, c2, c3 = st.columns(3)
        au["voice"] = c1.selectbox("Voice", vs, vs.index(au["voice"]) if au["voice"] in vs else 0, key=bk("voice"), help="neutral = a blend of a female and a male voice")
        au["voice_speed"] = c2.slider("Speaking speed", 0.6, 1.4, au["voice_speed"], 0.05, key=bk("vsp"))
        au["duck_db"] = c3.slider("Duck music under voice (dB)", -30, 0, int(au["duck_db"]), key=bk("duck"))
        au["loudness"] = st.slider("Final loudness (LUFS)", -24, -9, int(au["loudness"]), key=bk("lufs"), help="-16 suits most social platforms; -14 is louder")
    return B.validate_brand(b)


@st.cache_data(show_spinner=False, max_entries=24)
def style_preview(brand_json, img_path):
    """Render two still frames (title + a UI card or text beat) with the current brand."""
    from PIL import Image
    from trcore import render as R
    brand = json.loads(brand_json)
    brand["visual"]["background"] = "gradient" if brand["visual"]["background"] == "ai" else brand["visual"]["background"]
    brand["output"]["aspect"] = "9:16"
    w, h, fps = 540, 960, 24
    W = R.World(brand, w, h, fps)
    title = R.TitleScene(W, {"kind": "title", "text": brand["identity"].get("tagline") or "Your headline lands here", "kicker": brand["identity"].get("name", ""), "secs": 3}, 0)
    if img_path and os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB"); img.thumbnail((1400, 1400))
        sc = R.UIScene(W, {"kind": "ui", "text": "A caption in your brand voice", "sub": "Supporting detail, one short line", "kicker": "STEP 1", "secs": 4, "focus": [0.1, 0.1, 0.45, 0.3]}, 1, img)
    else:
        sc = R.LineScene(W, {"kind": "line", "text": "A caption in your brand voice", "sub": "Supporting detail, one short line", "kicker": "01 / 06", "secs": 4}, 1)
    a = title.frame(int(title.total * 0.6))
    b = sc.frame(int(sc.total * 0.62))
    out = Image.new("RGB", (w * 2 + 12, h), (20, 20, 26)); out.paste(a, (0, 0)); out.paste(b, (w + 12, 0))
    buf = io.BytesIO(); out.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


# ------------------------------------------------------------------ steps
def stepper():
    labels = ["1 · Brief", "2 · Brand & style", "3 · Storyboard", "4 · Render"]
    cols = st.columns(len(labels))
    for n, (c, lab) in enumerate(zip(cols, labels), 1):
        if c.button(lab, key=f"nav{n}", width="stretch", type="primary" if st.session_state.step == n else "secondary"):
            go(n)


def step_brief():
    ss = st.session_state
    kinds = list(KINDS)
    kind = st.radio("What do you have?", kinds, index=kinds.index(ss.kind_sel), horizontal=True)
    ss.kind_sel = kind
    k = KINDS[kind]
    if k == "idea":
        ss["topic"] = st.text_input("Topic", ss.get("topic", ""), placeholder="e.g. 5 stoic rules for a calm mind")
        c1, c2 = st.columns([2, 1])
        ss["mode"] = c1.selectbox("Format", TEXT_MODES, TEXT_MODES.index(ss.get("mode", "quotes")) if ss.get("mode", "quotes") in TEXT_MODES else 0)
        ss["lines"] = c2.number_input("Beats", 4, 14, int(ss.get("lines") or 8))
        ss["brief"] = st.text_area("Facts to ground the script (optional)", ss.get("brief", ""), height=100,
                                   help="Anything the AI may rely on. Without it, the writer stays general.")
    elif k == "screens":
        ss["brief"] = st.text_area("Describe the product — paste your pitch, README or landing copy", ss.get("brief", ""), height=190,
                                   help="The AI extracts a fact sheet from this and is only allowed to make claims found in it.")
        c1, c2 = st.columns([3, 1])
        ss["topic"] = c1.text_input("Working title (optional)", ss.get("topic", ""))
        ss["lines"] = c2.number_input("Beats (0 = automatic)", 0, 14, int(ss.get("lines") or 0))
        ss["mode"] = "appdemo"
    elif k == "photos":
        ss["topic"] = st.text_input("What's the story?", ss.get("topic", ""), placeholder="e.g. a winter climb at dawn")
        ss["lines"] = 0; ss["mode"] = "photostory"
    else:
        c1, c2 = st.columns([2, 1])
        ss["topic"] = c1.text_input("Topic (optional)", ss.get("topic", ""))
        ss["mode"] = c2.selectbox("Mine for", ["quotes", "facts"])
        ss["lines"] = st.number_input("Clips", 3, 12, int(ss.get("lines") or 6))
        up = st.file_uploader("Drop a video or audio file", type=["mp4", "mov", "mkv", "webm", "mp3", "wav", "m4a"], key=f"srcup{ss.up_nonce}")
        if up is not None:
            p = os.path.join(STATE_DIR, "source-" + re.sub(r"[^A-Za-z0-9._-]", "_", up.name))
            open(p, "wb").write(up.getvalue()); ss.source = p
        if ss.source:
            st.success(f"Source: {os.path.basename(ss.source)}")
    if k in ("screens", "photos"):
        label = "Drop screenshots" if k == "screens" else "Drop photos"
        ups = st.file_uploader(label + " (add as many as you like — you can reorder below)", type=["png", "jpg", "jpeg", "webp"],
                               accept_multiple_files=True, key=f"imgup{ss.up_nonce}")
        if ups:
            for f in ups:
                p = save_upload(f)
                if p not in ss.images:
                    ss.images.append(p)
            ss.up_nonce += 1
            st.rerun()
        image_grid()
    # validation + next
    ok, why = True, ""
    if k == "idea" and not ss.get("topic", "").strip(): ok, why = False, "Enter a topic."
    if k == "screens" and (len(ss.images) < 2 or not (ss.get("brief", "").strip() or ss.get("topic", "").strip())): ok, why = False, "Add at least 2 screenshots and a short description."
    if k == "photos" and (len(ss.images) < 2 or not ss.get("topic", "").strip()): ok, why = False, "Add at least 2 photos and a one-line story."
    if k == "podcast" and not ss.source: ok, why = False, "Upload a video or audio file."
    if not ok:
        st.info(why)
    if st.button("Next: choose brand & style →", type="primary", disabled=not ok):
        if k == "screens" and not ss.get("topic"):
            ss["topic"] = (ss.get("brief", "").strip().split("\n")[0])[:60]
        ss.sb = None
        go(2)


def step_brand():
    ss = st.session_state
    st.markdown("#### Start from a preset, then make it yours")
    presets = B.list_presets()
    cols = st.columns(3)
    for n, p in enumerate(presets):
        with cols[n % 3], st.container(border=True):
            pb = B.load_brand(p["id"])
            pal = pb["visual"]["palette"]
            st.markdown(f"<div class='tr-swatch' style='background:linear-gradient(135deg,{pal['bg'][0]},{pal['bg'][1]} 60%,{pal['accent']})'></div>", unsafe_allow_html=True)
            st.markdown(f"**{p['name']}**")
            st.caption(p["description"])
            if st.button("Use this" if ss.preset != p["id"] else "✓ Selected", key=f"pre_{p['id']}", width="stretch",
                         type="primary" if ss.preset == p["id"] else "secondary"):
                ss.preset = p["id"]; set_brand_base(pb); st.rerun()
    c1, c2 = st.columns([3, 1])
    imp = c1.file_uploader("…or import a brand kit (.json)", type=["json"], key=f"imp{ss.nonce}")
    if imp is not None:
        try:
            set_brand_base(json.loads(imp.getvalue().decode("utf-8"))); ss.preset = "custom"; st.rerun()
        except Exception as e:
            st.error(f"Not a valid brand kit: {e}")
    left, right = st.columns([3, 1.25])
    with left:
        brand = brand_editor(ss.base)
        ss.brand = brand
    with right:
        st.markdown("**Live style preview**")
        first_img = ss.images[0] if ss.images else None
        try:
            st.image(style_preview(json.dumps(brand, sort_keys=True), first_img), width="stretch")
            st.caption("Left: title card · right: a caption over " + ("your first screenshot." if first_img else "a text beat."))
        except Exception as e:
            st.caption(f"Preview unavailable: {e}")
        st.download_button("⬇ Save this brand kit", json.dumps(brand, indent=2, ensure_ascii=False),
                           file_name=f"{B.validate_brand(brand)['identity']['name'] or 'brand'}-brand.json".replace(" ", "_"),
                           mime="application/json", width="stretch")
    b1, b2 = st.columns([1, 4])
    if b1.button("← Back"): go(1)
    if b2.button("Next: generate the storyboard →", type="primary"): go(3)


def job_paths():
    ss = st.session_state
    if "job" not in ss:
        ss["job"] = time.strftime("%Y%m%d-%H%M%S")
    return (os.path.join(JOBS, f"{ss.job}.brand.json"), os.path.join(JOBS, f"{ss.job}.storyboard.json"))


def common_args():
    ss = st.session_state
    bp, _ = job_paths()
    B.save_brand(ss.brand, bp)
    args = ["--brand", bp]
    m = ss.models
    if m.get("writer"): args += ["--writer", m["writer"]]
    if m.get("vision"): args += ["--vision", m["vision"]]
    return args


def plan_args():
    ss = st.session_state
    k = KINDS[ss.kind_sel]
    a = ["--mode", ss.get("mode", "quotes"), "--plan-only", "--storyboard-out", job_paths()[1]]
    if ss.get("brief", "").strip():
        bf = os.path.join(JOBS, f"{ss.job}.brief.txt"); open(bf, "w", encoding="utf-8").write(ss["brief"]); a += ["--brief", bf]
    if ss.get("lines"): a += ["--lines", str(int(ss["lines"]))]
    if k in ("screens", "photos"): a += ["--images"] + ss.images
    if k == "podcast": a += ["--source", ss.source]
    if ss.get("topic"): a += ["--", ss["topic"]]          # '--' keeps a topic that starts with '-' from being read as a flag
    return a


def plan_problem():
    ss = st.session_state
    k = KINDS[ss.kind_sel]
    if k in ("screens", "photos") and len([p for p in ss.images if os.path.exists(p)]) < 2:
        return "Add at least 2 images in step 1 (the uploaded files are no longer available)."
    if k == "podcast" and not (ss.source and os.path.exists(ss.source)):
        return "Upload a video/audio file in step 1."
    if k == "idea" and not ss.get("topic", "").strip():
        return "Enter a topic in step 1."
    return None


def sb_table(sb):
    import pandas as pd
    rows = []
    for b in sb["beats"]:
        rows.append({"#": b["id"], "Preview": thumb_uri(sb["images"][b["img"]]) if (b["img"] is not None and b["img"] < len(sb["images"])) else None,
                     "Scene": b["scene"], "Img": b["img"], "Kicker": b["kicker"], "Caption": b["caption"], "Sub-line": b["sub"],
                     "Voice-over": b["vo"], "Secs": b["secs"], "Focus x,y,w,h": ",".join(str(v) for v in b["focus"]) if b["focus"] else "",
                     "KPI value | label": f"{b['kpi']['value']} | {b['kpi']['label']}" if b["kpi"] else ""})
    return pd.DataFrame(rows)


def table_to_sb(df, sb):
    out = copy.deepcopy(sb)
    old = {b["id"]: b for b in sb["beats"]}
    beats = []
    for _, r in df.iterrows():
        def g(k):
            v = r.get(k)
            return "" if v is None or (isinstance(v, float) and v != v) else v
        prev = old.get(int(r["#"]) if g("#") != "" else -1, {})
        img = g("Img")
        beats.append({"scene": g("Scene") or "line", "img": int(img) if img != "" else None, "kicker": str(g("Kicker")),
                      "caption": str(g("Caption")), "sub": str(g("Sub-line")), "vo": str(g("Voice-over")),
                      "secs": g("Secs") or None, "focus": g("Focus x,y,w,h") or None, "kpi": g("KPI value | label") or None,
                      "clip": prev.get("clip")})
    out["beats"] = beats
    return normalize(out)


def step_storyboard():
    ss = st.session_state
    bp, sp = job_paths()
    c1, c2 = st.columns([1, 3])
    gen = c1.button("✨ Generate storyboard" if not ss.sb else "↻ Regenerate everything", type="primary")
    c2.caption("The AI plans the whole film as an editable table. Fix any line before rendering — no render time is spent until you approve.")
    if gen:
        problem = plan_problem()
        if problem:
            st.error(problem)
        else:
            rc, lines_ = run_cli(common_args() + plan_args(), "Planning")
            if rc == 0 and os.path.exists(sp):
                ss.sb = json.load(open(sp, encoding="utf-8")); st.rerun()
            else:
                msg = next((l.strip() for l in reversed(lines_) if l.strip() and not l.strip().startswith("[")), "see the log above")
                st.error(f"Planning failed: {msg}")
    sb = ss.sb
    if not sb:
        b1, _ = st.columns([1, 5])
        if b1.button("← Back"): go(2)
        return
    t1, t2 = st.columns([3, 2])
    sb["title"] = t1.text_input("Opening title card", sb.get("title", ""))
    sb["cta"] = t2.text_input("Closing call-to-action", sb.get("cta", ""))
    if sb.get("lint"):
        st.warning("Quality notes (not blocking): " + " · ".join(sb["lint"][:4]))
    else:
        st.success("Quality gate passed — no banned words, length limits met, numbers grounded in your facts.")
    edited = st.data_editor(
        sb_table(sb), num_rows="dynamic", width="stretch", hide_index=True, key=f"ed{ss.get('edver', 0)}",
        column_config={"#": st.column_config.NumberColumn(disabled=True, width="small"),
                       "Preview": st.column_config.ImageColumn(width="small"),
                       "Scene": st.column_config.SelectboxColumn(options=SCENES, width="small"),
                       "Img": st.column_config.NumberColumn(min_value=0, step=1, width="small", help="0-based screenshot/photo index"),
                       "Caption": st.column_config.TextColumn(width="large"), "Sub-line": st.column_config.TextColumn(width="large"),
                       "Voice-over": st.column_config.TextColumn(width="large"),
                       "Secs": st.column_config.NumberColumn(min_value=1.0, max_value=20.0, step=0.5, width="small", help="blank = automatic"),
                       "Focus x,y,w,h": st.column_config.TextColumn(width="medium", help="normalized 0–1 region the camera zooms to")})
    ss.sb = table_to_sb(edited, sb)
    segs = build_segments(ss.sb, ss.brand)
    st.caption(f"≈ **{sum(s['secs'] for s in segs):.0f} s** · {len(ss.sb['beats'])} beats + title + end card")
    with st.expander("✍️ Rewrite one beat with AI"):
        r1, r2, r3 = st.columns([1, 4, 1.4])
        n = r1.number_input("Beat #", 1, max(len(ss.sb["beats"]), 1), 1)
        ins = r2.text_input("Instruction (optional)", placeholder="e.g. make it about speed · shorter · lead with the number")
        if r3.button("Rewrite beat", width="stretch"):
            json.dump(ss.sb, open(sp, "w", encoding="utf-8"), ensure_ascii=False)
            rc, _ = run_cli(common_args() + ["--storyboard", sp, "--regen-beat", str(int(n)), "--instruction", ins], "Rewriting beat")
            if rc == 0:
                ss.sb = json.load(open(sp, encoding="utf-8")); ss["edver"] = ss.get("edver", 0) + 1; st.rerun()
    with st.expander("📣 Post copy & hashtags"):
        ss.sb["description"] = st.text_area("Description", ss.sb.get("description", ""), height=70)
        tags = st.text_input("Hashtags (space separated, no #)", " ".join(ss.sb.get("hashtags", [])))
        ss.sb["hashtags"] = [t.lstrip("#") for t in tags.split() if t]
    b1, b2 = st.columns([1, 4])
    if b1.button("← Back"): go(2)
    if b2.button("Looks good — go to render →", type="primary"): go(4)


def step_render():
    ss = st.session_state
    if not ss.sb:
        st.info("Generate a storyboard first."); 
        if st.button("← Go to storyboard"): go(3)
        return
    bp, sp = job_paths()
    brand = ss.brand
    c1, c2, c3 = st.columns([2, 1, 1])
    labels = list(PLATFORMS)
    cur = next((i for i, (k, v) in enumerate(PLATFORMS.items()) if v == brand["output"]["aspect"]), 0)
    plat = c1.selectbox("Where will it be posted?", labels, cur)
    quality = c2.radio("Quality", ["preview", "final"], index=0 if brand["output"]["quality"] == "preview" else 1, horizontal=True,
                       help="preview = 720p/24fps, ~3× faster · final = 1080p/30fps")
    segs = build_segments(ss.sb, brand)
    c3.metric("Length", f"≈ {sum(s['secs'] for s in segs):.0f} s")
    vo = brand["audio"]["voiceover"]
    st.caption(f"Brand **{brand['name']}** · {len(ss.sb['beats'])} beats · music: {brand['audio']['music']} @ {brand['audio']['bpm']} BPM · "
               f"voice-over: {'on (' + (A.tts_backend() or 'no TTS installed') + ')' if vo else 'off'}")
    if st.button("🎬 Render video", type="primary"):
        brand["output"]["aspect"] = PLATFORMS[plat]; brand["output"]["quality"] = quality
        ss.brand = brand
        json.dump(ss.sb, open(sp, "w", encoding="utf-8"), ensure_ascii=False)
        a = common_args() + ["--storyboard", sp, "--aspect", PLATFORMS[plat], "--quality", quality]
        if ss.music_file: a += ["--music", ss.music_file]
        p = ss.paint
        if brand["visual"]["background"] == "ai":
            a += ["--bg", "ai", "--paint", p["engine"], "--quant", p["quant"], "--paint-res", p["res"]]
            if p["steps"]: a += ["--steps", str(p["steps"])]
            if p["lock"]: a += ["--reuse-bg"]
        rc, lines_ = run_cli(a, "Rendering")
        if rc == 0 and grab(lines_, "Video"):
            ss.result = {k: grab(lines_, v) for k, v in (("video", "Video"), ("caption", "Caption"), ("srt", "Subs"), ("json", "Data"))}
        else:
            st.error("Render failed — see the log above.")
    r = ss.result
    if r and r.get("video") and os.path.exists(r["video"]):
        show_result(r, ss.sb)
    b1, _ = st.columns([1, 5])
    if b1.button("← Back"): go(3)


def show_result(r, sb):
    left, right = st.columns([1, 1.4])
    with left:
        st.video(r["video"])
    with right:
        st.success(f"Saved: {os.path.basename(r['video'])}")
        c = st.columns(4)
        c[0].download_button("⬇ MP4", open(r["video"], "rb").read(), file_name=os.path.basename(r["video"]), mime="video/mp4", width="stretch")
        if r.get("srt") and os.path.exists(r["srt"]):
            c[1].download_button("⬇ Subtitles", open(r["srt"], "rb").read(), file_name=os.path.basename(r["srt"]), width="stretch")
        if r.get("json") and os.path.exists(r["json"]):
            c[2].download_button("⬇ Project", open(r["json"], "rb").read(), file_name=os.path.basename(r["json"]), mime="application/json", width="stretch",
                                 help="storyboard + brand kit — re-open it from the Library")
        if r.get("caption") and os.path.exists(r["caption"]):
            c[3].download_button("⬇ Caption", open(r["caption"], "rb").read(), file_name=os.path.basename(r["caption"]), width="stretch")
        pc = (sb or {}).get("platform_captions") or {}
        names = [(k, n) for k, n in (("linkedin", "LinkedIn"), ("x", "X"), ("youtube_shorts", "YouTube Shorts"), ("instagram", "Instagram")) if pc.get(k)]
        if names:
            tabs = st.tabs([n for _, n in names])
            for t, (k, _) in zip(tabs, names):
                with t:
                    st.code(pc[k], language=None)


def library():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "*.mp4")), key=os.path.getmtime, reverse=True)
    if not files:
        st.info("Your reels will show up here."); return
    for f in files[:30]:
        base = f[:-4]
        with st.expander(f"🎞️ {os.path.basename(f)}"):
            c1, c2 = st.columns([1, 1.6])
            c1.video(f)
            with c2:
                if os.path.exists(base + ".caption.txt"):
                    st.code(open(base + ".caption.txt", encoding="utf-8").read(), language=None)
                b = st.columns(3)
                b[0].download_button("⬇ MP4", open(f, "rb").read(), file_name=os.path.basename(f), key="dl" + f, width="stretch")
                if os.path.exists(base + ".json") and b[1].button("✏️ Open in editor", key="ed" + f, width="stretch"):
                    d = json.load(open(base + ".json", encoding="utf-8"))
                    st.session_state.sb = d["storyboard"]; set_brand_base(d["brand"]); st.session_state.preset = "custom"
                    st.session_state.pop("job", None); st.session_state.step = 3
                    st.session_state.images = [p for p in d["storyboard"].get("images", []) if os.path.exists(p)]
                    st.rerun()
                if b[2].button("🗑 Delete", key="rm" + f, width="stretch"):
                    for ext in (".mp4", ".caption.txt", ".json", ".srt"):
                        if os.path.exists(base + ext): os.remove(base + ext)
                    st.rerun()


def sidebar():
    ss = st.sidebar
    ss.markdown(f"### 🎬 {APP} `v{VERSION}`")
    with ss.expander("Engine & tools"):
        if st.button("Run preflight probe", width="stretch"):
            out = subprocess.run([sys.executable, CLI, "--probe"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
            st.code(out.stdout + out.stderr, language=None)
        m = st.session_state.models
        m["writer"] = st.text_input("Writer model (blank = auto)", m["writer"])
        m["vision"] = st.text_input("Vision model (blank = auto)", m["vision"])
        if st.button("Download more Google Fonts", width="stretch"):
            out = subprocess.run([sys.executable, os.path.join(ROOT, "download_fonts.py")], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
            st.code(out.stdout[-900:], language=None)
    with ss.expander("AI backdrop (optional)"):
        st.caption("Only used when a brand's Backdrop is set to **ai**. Needs `pip install torch diffusers`.")
        p = st.session_state.paint
        engines = ["auto", "flux", "zimage", "qwen", "sd35", "sdxl"]
        p["engine"] = st.selectbox("Engine", engines, engines.index(p["engine"]))
        p["res"] = st.selectbox("Resolution", ["768x1344", "576x1024", "1024x1792"], ["768x1344", "576x1024", "1024x1792"].index(p["res"]))
        p["steps"] = st.slider("Steps (0 = engine default)", 0, 50, p["steps"])
        qs = ["auto", "none", "8bit", "4bit"]
        p["quant"] = st.selectbox("VRAM quantization", qs, qs.index(p["quant"]))
        p["lock"] = st.checkbox("Theme lock (reuse last backdrop)", p["lock"])
        if st.button("Warm paint engine (one-time download)", width="stretch"):
            rc, _ = run_cli(["--warm", "--paint", p["engine"], "--quant", p["quant"]], "Warming")
    if ss.button("Start over", width="stretch"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


def main():
    init()
    sidebar()
    st.markdown(f"<p class='tr-title'>{APP} Studio</p><p class='tr-sub'>Brief → brand → storyboard → video. 100% local — no uploads, no subscriptions.</p>", unsafe_allow_html=True)
    tab_create, tab_lib = st.tabs(["Create", "Library"])
    with tab_create:
        stepper()
        st.divider()
        {1: step_brief, 2: step_brand, 3: step_storyboard, 4: step_render}[st.session_state.step]()
    with tab_lib:
        library()


main()

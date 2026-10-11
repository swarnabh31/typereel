"""Master prompts + JSON schemas for every LLM stage.

Design rules: (1) each stage gets only the context it needs, (2) the brand voice block is
injected everywhere, (3) outputs are schema-constrained, (4) facts are the ONLY allowed claims.
"""
import json

from .brand import voice_block

_STR = {"type": "string"}
_STRS = {"type": "array", "items": _STR}
_NUM = {"type": "number"}

# ------------------------------------------------------------------ schemas
FACT_SCHEMA = {"type": "object", "properties": {
    "product_name": _STR, "one_liner": _STR, "audience": _STR, "problem": _STR,
    "claims": _STRS, "differentiators": _STRS,
    "metrics": {"type": "array", "items": {"type": "object", "properties": {"value": _STR, "label": _STR},
                                           "required": ["value", "label"]}}},
    "required": ["product_name", "one_liner", "claims"]}

SCREEN_SCHEMA = {"type": "object", "properties": {
    "screen": _STR, "purpose": _STR,
    "stage": {"type": "string", "enum": ["onboarding", "core", "result", "settings", "unknown"]},
    "visible_text": _STRS,
    "numbers": {"type": "array", "items": {"type": "object", "properties": {"value": _STR, "label": _STR},
                                           "required": ["value", "label"]}},
    "regions": {"type": "array", "items": {"type": "object", "properties": {
        "label": _STR, "x": _NUM, "y": _NUM, "w": _NUM, "h": _NUM}, "required": ["label", "x", "y", "w", "h"]}}},
    "required": ["screen", "purpose", "regions"]}

PHOTO_SCHEMA = {"type": "object", "properties": {
    "subjects": _STRS, "setting": _STR, "action": _STR, "mood": _STR, "colors": _STRS, "one_liner": _STR},
    "required": ["subjects", "one_liner"]}

_BEAT_COMMON = {"kicker": _STR, "caption": _STR, "sub": _STR, "vo": _STR}
DIRECTOR_SCHEMA = {"type": "object", "properties": {
    "title": _STR, "cta": _STR, "hashtags": _STRS, "description": _STR, "art": _STR,
    "beats": {"type": "array", "items": {"type": "object", "properties": dict(
        _BEAT_COMMON, screen_id={"type": "integer"}, region={"type": "integer"},
        scene={"type": "string", "enum": ["ui", "kpi"]}, kpi_value=_STR, kpi_label=_STR),
        "required": ["screen_id", "caption"]}}},
    "required": ["title", "beats", "cta"]}

STORY_SCHEMA = {"type": "object", "properties": {
    "title": _STR, "cta": _STR, "hashtags": _STRS, "description": _STR, "art": _STR,
    "beats": {"type": "array", "items": {"type": "object", "properties": dict(
        _BEAT_COMMON, photo_id={"type": "integer"}), "required": ["photo_id", "caption"]}}},
    "required": ["title", "beats", "cta"]}

LINES_SCHEMA = {"type": "object", "properties": {
    "title": _STR, "lines": _STRS, "hashtags": _STRS, "description": _STR, "art": _STR, "vo": _STRS},
    "required": ["title", "lines"]}

MINER_SCHEMA = {"type": "object", "properties": {
    "title": _STR, "hashtags": _STRS, "description": _STR, "art": _STR,
    "clips": {"type": "array", "items": {"type": "object", "properties": {
        "ids": {"type": "array", "items": {"type": "integer"}}, "quote": _STR, "caption": _STR},
        "required": ["ids"]}}},
    "required": ["title", "clips"]}

PACK_SCHEMA = {"type": "object", "properties": {
    "linkedin": _STR, "x": _STR, "youtube_shorts": _STR, "instagram": _STR, "hashtags": _STRS},
    "required": ["linkedin", "x", "instagram"]}

BEAT_SCHEMA = {"type": "object", "properties": dict(_BEAT_COMMON, kpi_value=_STR, kpi_label=_STR),
               "required": ["caption"]}

MODES = {
    "quotes": "punchy, quotable one-liners of wisdom. Each line must stand alone.",
    "facts": "surprising, specific, little-known facts. Each line is one self-contained fact.",
    "flashcards": "learning beats alternating: a short concept, then its sharp one-line explanation.",
    "story": "a micro-story: line 1 is the hook, middle lines escalate, the last line is the payoff.",
    "hottake": "bold contrarian takes, each followed by a one-line justification.",
}
MINE_FLAVOR = {"quotes": "verbatim quotable moments — opinions, wisdom, one-liners worth repeating",
               "facts": "concrete factual claims: numbers, names, surprising specifics"}

CORE_RULES = ("RULES: Use ONLY information present in FACTS or ASSETS. Never invent numbers, customers, "
              "features or claims. No exclamation marks. Plain, specific language. Output ONLY the JSON.")


# ------------------------------------------------------------------ stage prompts
def fact_sheet_prompt(brief):
    return f"""Extract a FACT SHEET from the product brief below. Include ONLY what the brief states.
If the brief contains no numbers, "metrics" must be an empty list. Do not embellish.

BRIEF:
\"\"\"{brief[:6000]}\"\"\"

Fields: product_name, one_liner (<=14 words), audience (who it is for), problem (the pain it removes),
claims (3-7 short, verifiable statements), differentiators (0-4), metrics (value + label pairs, only if stated)."""


SCREEN_PROMPT = """You are analysing ONE screenshot of a software application for a film editor.
Describe only what is actually visible. Coordinates are NORMALIZED to the image (0..1, origin top-left):
x,y = top-left corner of the region, w,h = its size.
- screen: the name of this view; purpose: what the user accomplishes here (<=14 words)
- stage: onboarding | core | result | settings | unknown
- visible_text: up to 12 short strings of text you can READ in the UI (titles, labels, buttons)
- numbers: any KPI/metric numbers visible, as {value, label} (copy digits exactly; none if none)
- regions: 1-3 regions most worth zooming into (a KPI card, a chart, a column, a primary button). Give a label each."""

PHOTO_PROMPT = """Describe this photograph for a film editor. Be literal and specific.
subjects: short tags (e.g. 'young woman, red jacket'); setting; action; mood; colors (2-3); one_liner (cinematic, <=16 words)."""


def _cards_listing(cards):
    out = []
    for i, c in enumerate(cards):
        regs = "; ".join(f"{j}={r.get('label', '?')}" for j, r in enumerate(c.get("regions", [])[:3])) or "none"
        nums = ", ".join(f"{n.get('value')} {n.get('label')}" for n in c.get("numbers", [])[:4]) or "none"
        txt = " | ".join(c.get("visible_text", [])[:8])
        out.append(f"[{i}] {c.get('stage', '?')} · {c.get('screen', '?')} — {c.get('purpose', '?')}\n"
                   f"     visible text: {txt or '—'}\n     numbers: {nums}\n     zoom regions: {regs}")
    return "\n".join(out)


def director_prompt(brand, facts, cards, n_beats, total_sec, user_brief=""):
    f = json.dumps(facts, ensure_ascii=False) if facts else "(none — rely on ASSETS)"
    v = brand["voice"]
    return f"""ROLE: You are the creative director of a product film for {brand['identity'].get('name') or 'this product'}.

BRAND VOICE
{voice_block(brand)}

FACTS (the only claims you may make)
{f}

ASSETS — numbered screens ({len(cards)})
{_cards_listing(cards)}

BRIEF FROM THE USER
{user_brief[:1500] or '(none)'}

TASK: Direct a {total_sec:.0f}-second film of about {n_beats} beats following a logical product workflow.
- "title": the opening hook — the buyer's problem or the outcome, <= 8 words.
- Each beat: screen_id (a screen number), region (index of the zoom region to pan to, or -1), scene "ui",
  kicker (STEP 1 / STEP 2 ... or a 1-2 word label), caption (<= {v['max_caption_words']} words, the benefit shown),
  sub (<= {v['max_sub_words']} words, the supporting detail), vo (one natural spoken sentence, <= 20 words, no hype).
- A screen may be used twice only with a DIFFERENT region. Prefer one beat per screen.
- If a screen shows real KPI numbers, you may add a beat with scene "kpi" using kpi_value (digits exactly as shown or stated in FACTS) and kpi_label.
- Last: "cta" = a calm call to action <= 6 words (default: "{brand['identity'].get('cta', 'Learn more')}").
- Also write hashtags (5-8, no #), a one-sentence description, and "art" (one-sentence background art direction in this style: {brand['visual'].get('art_style', '')}).
{CORE_RULES}"""


def story_prompt(brand, cards, wish, n_beats):
    listing = "\n".join(
        f"[{i}] subjects: {c.get('subjects', '?')} | {c.get('setting', '?')} | {c.get('action', '?')} | "
        f"mood: {c.get('mood', '?')} | {c.get('one_liner', '?')}" for i, c in enumerate(cards))
    v = brand["voice"]
    return f"""ROLE: You are a cinematic storyteller cutting a {n_beats * 3}-second photo film.

BRAND VOICE
{voice_block(brand)}

USER'S WISH: "{wish}"

PHOTOS
{listing}

TASK: Choose photos in narrative order (each at most once) and write one beat per photo:
photo_id, kicker (SCENE n), caption (<= {v['max_caption_words'] + 2} words, present tense), sub (optional, <= {v['max_sub_words']} words), vo (one sentence).
CAST RULE: keep ONE consistent protagonist and refer to them with the SAME words in every caption.
Also: "title" (<= 6 words), "cta" (<= 6 words), hashtags (5-8, no #), description (one sentence), art (background direction).
{CORE_RULES}"""


def lines_prompt(brand, topic, mode, n, facts=None, avoid=None):
    v = brand["voice"]
    f = f"\nFACTS (only claims you may make):\n{json.dumps(facts, ensure_ascii=False)}\n" if facts else ""
    av = ("\nAVOID lines that resemble these earlier ones:\n" + "\n".join(f"- {a}" for a in avoid[:8]) + "\n") if avoid else ""
    return f"""ROLE: You are the head writer for short-form video.

BRAND VOICE
{voice_block(brand)}
{f}{av}
TOPIC: "{topic}"
FORM: {MODES.get(mode, MODES['quotes'])}

TASK: Write EXACTLY {n} caption lines, each <= {v['max_caption_words']} words, plain language, no emojis, no hashtags inside lines.
Line 1 must hook. The last line must land. Also: "title" (<= 6 words), "vo" (the same {n} lines, rewritten as natural spoken sentences),
hashtags (5-8, no #), description (one sentence), art (one-sentence background direction in this style: {brand['visual'].get('art_style', '')}).
Output ONLY the JSON."""


def miner_prompt(brand, segments, mode, n):
    if len(segments) > 160:
        step = len(segments) / 160.0
        segments = [segments[int(i * step)] for i in range(160)]
    listing = "\n".join(f"[{i}] {a:.1f}-{b:.1f}s {t}" for i, (a, b, t) in enumerate(segments))
    return f"""Transcript segments from a recording:
{listing}

BRAND VOICE (for captions only)
{voice_block(brand)}

Pick the {n} best {MINE_FLAVOR.get(mode, MINE_FLAVOR['quotes'])}.
For each pick 1-4 consecutive segment ids, give the verbatim quote (join those segments' text exactly) and a caption of <= 12 words.
Also: title (<= 6 words), hashtags (5-8, no #), description, art. Output ONLY the JSON."""


def revise_prompt(brand, storyboard, issues, facts=None):
    f = json.dumps(facts, ensure_ascii=False) if facts else "(none)"
    return f"""You previously wrote this storyboard JSON. A quality check found problems.

BRAND VOICE
{voice_block(brand)}

FACTS
{f}

STORYBOARD
{json.dumps(storyboard, ensure_ascii=False)}

PROBLEMS TO FIX
{chr(10).join('- ' + i for i in issues)}

Fix ONLY the listed problems. Keep every other field, the order and the number of beats unchanged.
Return the full corrected JSON in the same schema. {CORE_RULES}"""


def regen_beat_prompt(brand, facts, beat, neighbours, instruction):
    return f"""Rewrite ONE beat of a product film.

BRAND VOICE
{voice_block(brand)}

FACTS
{json.dumps(facts, ensure_ascii=False) if facts else '(none)'}

NEIGHBOUR BEATS (for flow, do not repeat them)
{json.dumps(neighbours, ensure_ascii=False)}

CURRENT BEAT
{json.dumps({k: beat.get(k) for k in ('kicker', 'caption', 'sub', 'vo')}, ensure_ascii=False)}

INSTRUCTION: {instruction or 'Write a stronger, more specific version.'}
Return JSON with kicker, caption (<= {brand['voice']['max_caption_words']} words), sub (<= {brand['voice']['max_sub_words']} words), vo. {CORE_RULES}"""


def pack_prompt(brand, sb, facts):
    beats = [b.get("caption", "") for b in sb.get("beats", [])]
    return f"""Write platform-specific post copy for a finished short product/brand video.

BRAND VOICE
{voice_block(brand)}

VIDEO: "{sb.get('title', '')}" — beats: {json.dumps(beats, ensure_ascii=False)}
FACTS: {json.dumps(facts, ensure_ascii=False) if facts else '(none)'}
CTA: {sb.get('cta', '')} {brand['identity'].get('website', '')}

Return JSON: linkedin (2-3 calm, professional sentences, no hashtags in body), x (<= 240 chars), youtube_shorts (<= 100 chars title-style),
instagram (1-2 sentences), hashtags (5-8, no #). {CORE_RULES}"""

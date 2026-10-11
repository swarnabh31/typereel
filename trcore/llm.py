"""Ollama client: structured JSON output, model resolution, embeddings."""
import json, math, os, re, urllib.error, urllib.request

from .util import log

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
if not OLLAMA.startswith("http"):
    OLLAMA = "http://" + OLLAMA

WRITER_PREFS = ["qwen3.8:27b", "gemma4:12b", "gemma4", "gemma3:12b", "gemma3", "qwen3:14b", "qwen3", "llama3.1", "gemma"]
EDITOR_PREFS = ["nimble:9b", "nimble"]
EMBED_PREFS = ["nomic-embed-text:latest", "nomic-embed-text"]
VISION_PREFS = ["qwen3.8:27b", "gemma4:12b", "gemma4", "gemma3:12b", "gemma3",
                "qwen2.5vl", "llava:latest", "minicpm-v", "llama3.2-vision", "moondream"]
VISION_FAMILIES = ("gemma3", "gemma4", "llava", "minicpm-v", "qwen2vl", "qwen2.5vl",
                   "moondream", "bakllava", "qwen3", "mllama", "llama3.2-vision")


def _req(path, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(OLLAMA + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def ollama_get(path, timeout=10):
    return _req(path, None, timeout)


def ollama_post(path, payload, timeout=600):
    return _req(path, payload, timeout)


def ollama_ok():
    try:
        ollama_get("/api/tags", 3)
        return True
    except Exception:
        return False


def list_models():
    try:
        return [m["name"] for m in ollama_get("/api/tags").get("models", [])]
    except Exception:
        return []


def resolve_model(prefs, installed, override=None):
    """Exact match first, then prefix match, walking the preference list."""
    if override:
        for m in installed:
            if m == override or m.startswith(override):
                return m
        return override if not installed else None
    for want in prefs:
        for m in installed:
            if m == want:
                return m
    for want in prefs:
        base = want.split(":")[0]
        for m in installed:
            if m.startswith(want) or m.split(":")[0] == base:
                return m
    return None


def vision_capable(model):
    if not model:
        return False
    try:
        info = ollama_post("/api/show", {"model": model}, timeout=15)
        if "vision" in (info.get("capabilities") or []):
            return True
        fam = " ".join((info.get("details", {}) or {}).get("families") or []) + " " + model
        if "clip" in fam.lower() or "mllama" in fam.lower():
            return True
    except Exception:
        pass
    return any(f in model.lower() for f in VISION_FAMILIES)


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def extract_json(s):
    if not s:
        return None
    s = _THINK_RE.sub("", s).strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    for a, b in (("{", "}"), ("[", "]")):
        i, j = s.find(a), s.rfind(b)
        if i != -1 and j > i:
            try:
                return json.loads(s[i:j + 1])
            except Exception:
                continue
    return None


def llm(model, prompt, temperature=0.4, images=None, ctx=8192, schema=None, system=None,
        timeout=900):
    """One generation call. Uses Ollama structured outputs when a schema is given and
    degrades gracefully (schema -> plain json mode) on older servers."""
    base = {"model": model, "prompt": prompt, "stream": False, "keep_alive": "10m",
            "options": {"temperature": temperature, "num_ctx": ctx}}
    if system:
        base["system"] = system
    if images:
        base["images"] = images
    variants = []
    if schema is not None:
        variants += [dict(base, format=schema, think=False), dict(base, format=schema),
                     dict(base, format="json")]
    else:
        variants += [dict(base, think=False), dict(base)]
    last = None
    for payload in variants:
        try:
            return ollama_post("/api/generate", payload, timeout).get("response", "")
        except urllib.error.HTTPError as e:      # older server rejects a field -> next variant
            last = e
            continue
    raise last if last else RuntimeError("llm call failed")


def ask_json(model, prompt, schema=None, temperature=0.4, images=None, ctx=8192, system=None, retries=1):
    """llm() + parse. Retries once with a stricter reminder when the JSON is unusable."""
    p = prompt
    for attempt in range(retries + 1):
        try:
            data = extract_json(llm(model, p, temperature, images, ctx, schema, system))
        except Exception as e:
            log("llm", f"call failed ({type(e).__name__}: {str(e)[:80]})")
            data = None
        if isinstance(data, (dict, list)):
            return data
        p = prompt + "\n\nIMPORTANT: your previous answer was not valid JSON. Return ONLY the JSON object."
        temperature = max(0.1, temperature - 0.2)
    return None


def unload_ollama():
    try:
        running = ollama_get("/api/ps").get("models", [])
        for m in running:
            ollama_post("/api/generate", {"model": m["name"], "prompt": "", "keep_alive": 0}, timeout=30)
        if running:
            log("art", f"VRAM freed: unloaded {len(running)} Ollama model(s) BEFORE paint load")
    except Exception:
        pass


def embed(model, text):
    try:
        return ollama_post("/api/embed", {"model": model, "input": [text]}, timeout=120)["embeddings"][0]
    except Exception:
        return ollama_post("/api/embeddings", {"model": model, "prompt": text}, timeout=120).get("embedding")


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0

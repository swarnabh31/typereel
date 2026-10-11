"""AI background painter (optional): FLUX.1-schnell / Z-Image / Qwen-Image / SD3.5 / SDXL via Diffusers.
Ported from TypeReel v5 — behaviour unchanged. Falls back silently to the procedural engine."""
import glob, hashlib, os, time

from .util import log, STATE_DIR
from .llm import unload_ollama

QWEN_IMAGE = "Qwen/Qwen-Image-2.1"

PROVIDERS = {
    "flux":   dict(repo="black-forest-labs/FLUX.1-schnell", cls="FluxPipeline",
                   steps=4,  res=(768, 1344), guide=0.0, dl=23,
                   qsubs=[("transformer", ["FluxTransformer2DModel"]),
                          ("text_encoder_2", ["T5EncoderModel"])]),
    "zimage": dict(repo="Tongyi-MAI/Z-Image-Turbo", cls="ZImagePipeline",
                   steps=8,  res=(768, 1344), guide=0.0, dl=12,
                   qsubs=[("transformer", ["ZImageTransformer2DModel", "QwenImageTransformer2DModel"])]),
    "qwen":   dict(repo=QWEN_IMAGE, cls="DiffusionPipeline",
                   steps=20, res=(768, 1344), guide=None, dl=40,
                   qsubs=[("transformer", ["QwenImageTransformer2DModel"])]),
    "sd35":   dict(repo="stabilityai/stable-diffusion-3.5-medium", cls="StableDiffusion3Pipeline",
                   steps=28, res=(768, 1344), guide=4.5, dl=10,
                   qsubs=[("transformer", ["SD3Transformer2DModel"])]),
    "sdxl":   dict(repo="stabilityai/stable-diffusion-xl-base-1.0", cls="StableDiffusionXLPipeline",
                   steps=4,  res=(768, 1344), guide=1.0, dl=7,
                   qsubs=[("unet", ["UNet2DConditionModel"])],
                   lora=("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_lora.sdxl.safetensors")),
}
NEED_GB = {"flux": 34, "zimage": 15, "qwen": 40, "sd35": 15, "sdxl": 8}

def _hf_downloaded(repo):
    d = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                     "models--" + repo.replace("/", "--"))
    return bool(os.path.isdir(d) and glob.glob(os.path.join(d, "snapshots", "*", "*")))

def auto_provider():
    for p in ("flux", "zimage", "qwen"):
        if _hf_downloaded(PROVIDERS[p]["repo"]): return p
    return "flux"

class PaintEngine:
    """One paint API, five backends. FLUX.1-schnell default: 4 steps, no login, Apache-2.0."""

    def __init__(self, want, provider="auto", steps=None, res=None, quant="auto",
                 reuse=False, lora=None, compile=False):
        self.pipe, self.lora_arg, self.reuse_path = None, lora, None
        if not want: return
        if reuse:
            hit = self._newest_cache()
            if hit:
                self.reuse_path = hit
                self.provider, self.steps, self.res, self.guide = "cache", 0, (0, 0), None
                log("art", f"theme lock: reusing {os.path.basename(hit)} — paint skipped")
                return
        try:
            import torch, diffusers as DF
            import transformers as TF
            unload_ollama()                                   # BEFORE load = no CPU sharding
            props = torch.cuda.get_device_properties(0)
            free_gb = torch.cuda.mem_get_info(0)[0] / 1024**3
            cap = torch.cuda.get_device_capability(0)
            if cap[0] >= 12 and tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) < (2, 7):
                log("art", "WARNING: Blackwell + torch<2.7 — pip install -U torch --index-url https://download.pytorch.org/whl/cu128")
            self.provider = provider if provider != "auto" else auto_provider()
            cfgp = PROVIDERS[self.provider]
            self.steps = steps or cfgp["steps"]
            self.res = res or cfgp["res"]
            self.guide = cfgp["guide"]
            log("art", f"GPU {props.name} {props.total_memory/1024**3:.0f} GB ({free_gb:.0f} free) — provider: {self.provider}")
            if not _hf_downloaded(cfgp["repo"]):
                log("art", f"first run: downloading {cfgp['repo']} (~{cfgp['dl']} GB, one-time)")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            cls = getattr(DF, cfgp["cls"], None) or DF.DiffusionPipeline
            need = NEED_GB.get(self.provider, 24)
            quant = self._pick(quant, free_gb, need)
            comp_kw = {}
            if quant != "none":
                try:
                    try: from diffusers import BitsAndBytesConfig as BNB
                    except Exception: BNB = TF.BitsAndBytesConfig
                    bcfg = BNB(load_in_4bit=(quant == "4bit"), load_in_8bit=(quant == "8bit"),
                               bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
                    got = 0
                    for sub, names in cfgp.get("qsubs", []):
                        for nm in names:
                            C = getattr(DF, nm, None) or getattr(TF, nm, None)
                            if C is None: continue
                            try:
                                comp_kw[sub] = C.from_pretrained(
                                    cfgp["repo"], subfolder=sub, quantization_config=bcfg,
                                    torch_dtype=torch.bfloat16)
                                got += 1; break
                            except Exception: continue
                    if got: log("art", f"{got} component(s) quantized {quant} — resident in VRAM")
                except Exception as e:
                    log("art", f"quant unavailable ({type(e).__name__}) — bf16")
            self.pipe = cls.from_pretrained(cfgp["repo"], torch_dtype=torch.bfloat16, **comp_kw)
            try:
                c = getattr(self.pipe, "transformer", None) or getattr(self.pipe, "unet", None)
                if c is not None: c.set_attention_backend("sdpa")
            except Exception: pass
            try:
                if quant != "none" or free_gb >= need + 4: self.pipe.to("cuda")
                else:
                    self.pipe.enable_model_cpu_offload()
                    log("art", "warning: CPU offload active — try --quant 4bit")
            except Exception:
                self.pipe.enable_model_cpu_offload()
            try: self.pipe.vae.enable_tiling()
            except Exception: pass
            lora_spec = lora or cfgp.get("lora")
            if lora_spec:
                try:
                    if isinstance(lora_spec, str) and os.path.exists(lora_spec):
                        self.pipe.load_lora_weights(os.path.dirname(lora_spec),
                                                    weight_name=os.path.basename(lora_spec))
                    elif isinstance(lora_spec, str):
                        self.pipe.load_lora_weights(lora_spec)
                    else:
                        self.pipe.load_lora_weights(lora_spec[0], weight_name=lora_spec[1])
                    try: self.pipe.fuse_lora()
                    except Exception: pass
                    log("art", "LoRA fused")
                except Exception as e:
                    log("art", f"LoRA skipped ({type(e).__name__})")
            if compile:
                try:
                    attr = "transformer" if getattr(self.pipe, "transformer", None) is not None else "unet"
                    setattr(self.pipe, attr, torch.compile(getattr(self.pipe, attr), mode="reduce-overhead"))
                    log("art", "torch.compile on (first paint pays compile cost)")
                except Exception: pass
            log("art", f"paint engine ready: {self.provider} {quant}, {self.res[0]}x{self.res[1]}, {self.steps} steps")
        except Exception as e:
            log("art", f"paint unavailable ({type(e).__name__}) — gradient engine")

    @staticmethod
    def _pick(quant, free_gb, need):
        if quant != "auto": return quant
        if free_gb >= need: return "none"
        return "8bit" if free_gb >= need*0.55 else "4bit"

    @staticmethod
    def _newest_cache():
        hits = glob.glob(os.path.join(STATE_DIR, "bgcache", "*.jpg"))
        return max(hits, key=os.path.getmtime) if hits else None

    def image(self, prompt):
        if self.reuse_path:
            from PIL import Image
            return Image.open(self.reuse_path).convert("RGB")
        if not self.pipe: return None
        os.makedirs(os.path.join(STATE_DIR, "bgcache"), exist_ok=True)
        key = hashlib.sha1((self.provider + prompt + f"{self.res}").encode()).hexdigest()[:16]
        path = os.path.join(STATE_DIR, "bgcache", key + ".jpg")
        if os.path.exists(path):
            log("art", "background from cache — paint skipped")
            from PIL import Image
            return Image.open(path).convert("RGB")
        log("art", f"painting via {self.provider} ({self.steps} steps @ {self.res[0]}x{self.res[1]}) ...")
        t0 = time.time()
        kw = dict(height=self.res[1], width=self.res[0], num_inference_steps=self.steps)
        if self.guide is not None: kw["guidance_scale"] = self.guide
        img = self.pipe(prompt=prompt, **kw).images[0].convert("RGB")
        dt = time.time() - t0
        log("art", f"paint done in {dt:.1f}s ({dt/max(self.steps,1):.2f}s/it) — cached for reuse")
        img.save(path, quality=92)
        return img


def art_prompt(sb, brand):
    """Background prompt: the director's art line + the brand's art style, never text/people/logos."""
    base = sb.get("art") or "atmospheric abstract backdrop"
    style = brand["visual"].get("art_style", "")
    return (f"{base}. {style}. vertical cinematic background, atmospheric depth, muted colors, "
            "high detail, no text, no watermark, no logos, no people")


def has_diffusers():
    try:
        import diffusers, torch  # noqa: F401
        return torch.cuda.is_available() or torch.backends.mps.is_available() or bool(os.environ.get("TYPEREEL_AI_CPU"))
    except Exception:
        return False

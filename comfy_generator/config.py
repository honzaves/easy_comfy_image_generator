"""
comfy_generator.config
======================
All tuneable constants, model definitions, and the Ollama enhancement prompt.
Edit this file to point at your own ComfyUI / Ollama instances or to add models.
"""

# ── Service endpoints ──────────────────────────────────────────────────────────

COMFYUI_URL  = "http://127.0.0.1:8188"
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3:27b"

# ── Token limits ───────────────────────────────────────────────────────────────

T5_LIMIT   = 512   # T5-XXL max tokens (hard cap for Flux)
CLIP_LIMIT = 77    # CLIP-L max tokens (warning threshold)

# ── Resolution presets ─────────────────────────────────────────────────────────
# Each entry: (label, width, height, aspect_ratio_label)

RESOLUTIONS: list[tuple[str, int, int, str]] = [
    ("512×512",   512,  512,  "1:1"),
    ("768×512",   768,  512,  "3:2"),
    ("832×512",   832,  512,  "16:9"),
    ("1024×576",  1024, 576,  "16:9"),
    ("1024×1024", 1024, 1024, "1:1"),
    ("1216×768",  1216, 768,  "16:10"),
    ("1344×768",  1344, 768,  "16:9"),
    ("1536×640",  1536, 640,  "21:9"),
    ("1920×1088", 1920, 1088, "16:9 XL"),
    ("1280×1280", 1280, 1280, "1:1 +"),
    ("1536×1536", 1536, 1536, "1:1 ++"),
    ("1600×1600", 1600, 1600, "1:1 max"),
]

# ── Model definitions ──────────────────────────────────────────────────────────
# Each key is a model_key string used throughout the app.
#
# The "arch" field selects which ComfyUI workflow graph is built (see
# workflows.py).  Three architectures are supported:
#
#   "flux"  — split components loaded via UNETLoader + VAELoader + DualCLIPLoader.
#             Fields: unet, steps, guidance.  cfg is fixed at 1.0; the negative
#             prompt is ignored, so FluxGuidance carries conditioning instead.
#   "sd15"  — all-in-one Stable Diffusion 1.5 checkpoint (one .safetensors file),
#             loaded via CheckpointLoaderSimple.  Native 512×512.
#   "sdxl"  — all-in-one SDXL checkpoint, also CheckpointLoaderSimple.  Native
#             1024×1024.
#
# Checkpoint models ("sd15"/"sdxl") additionally carry: ckpt (filename in
# ComfyUI's models/checkpoints/), cfg, sampler, scheduler, native (w, h) and an
# optional per-model negative prompt.  Edit any of these to retune a model.
#
# NOTE ON CLASSIFICATION: the arch / native size of the checkpoint models below
# was inferred from each file's name, not verified against its metadata.  If a
# model misbehaves (e.g. duplication at its supposed native size), correcting
# its "arch"/"native" here is the only change needed.

# Default negative prompt for SD/SDXL checkpoints.  Flux ignores negatives.
DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, "
    "extra digit, fewer digits, cropped, worst quality, low quality, "
    "jpeg artifacts, signature, watermark, blurry, deformed"
)

MODELS: dict[str, dict] = {
    "flux-schnell": {
        "arch":     "flux",
        "unet":     "flux1-schnell.safetensors",
        "steps":    4,
        "guidance": 1.0,
        "label":    "Flux Schnell  ·  fast",
        "desc":     "4 steps  ·  ~2 min  ·  good quality",
    },
    "flux-dev": {
        "arch":     "flux",
        "unet":     "flux1-dev.safetensors",
        "steps":    20,
        "guidance": 3.5,
        "label":    "Flux Dev  ·  quality",
        "desc":     "20 steps  ·  ~5 min  ·  best quality",
    },

    # ── SD 1.5 checkpoints (native 512×512) ─────────────────────────────────────
    "sd15-base": {
        "arch":      "sd15",
        "ckpt":      "v1-5-pruned-emaonly.safetensors",
        "steps":     25,
        "cfg":       7.0,
        "sampler":   "dpmpp_2m",
        "scheduler": "karras",
        "native":    (512, 512),
        "negative":  DEFAULT_NEGATIVE,
        "label":     "SD 1.5  ·  base",
        "desc":      "25 steps  ·  512px native  ·  SD 1.5",
    },
    "dreamshaper-8": {
        "arch":      "sd15",
        "ckpt":      "dreamshaper_8.safetensors",
        "steps":     25,
        "cfg":       7.0,
        "sampler":   "dpmpp_2m",
        "scheduler": "karras",
        "native":    (512, 512),
        "negative":  DEFAULT_NEGATIVE,
        "label":     "DreamShaper 8  ·  versatile",
        "desc":      "25 steps  ·  512px native  ·  SD 1.5",
    },
    "rev-animated": {
        "arch":      "sd15",
        "ckpt":      "revAnimated_v2Rebirth.safetensors",
        "steps":     25,
        "cfg":       7.0,
        "sampler":   "dpmpp_2m",
        "scheduler": "karras",
        "native":    (512, 512),
        "negative":  DEFAULT_NEGATIVE,
        "label":     "ReV Animated  ·  stylised",
        "desc":      "25 steps  ·  512px native  ·  SD 1.5",
    },
    "slate-pencil": {
        "arch":      "sd15",
        "ckpt":      "slatePencilMix_v10.safetensors",
        "steps":     25,
        "cfg":       7.0,
        "sampler":   "dpmpp_2m",
        "scheduler": "karras",
        "native":    (512, 512),
        "negative":  DEFAULT_NEGATIVE,
        "label":     "Slate Pencil Mix  ·  illustrative",
        "desc":      "25 steps  ·  512px native  ·  SD 1.5",
    },
    "realistic-vision-hyper": {
        "arch":      "sd15",
        "ckpt":      "realisticVisionV60B1_v51HyperVAE.safetensors",
        "steps":     6,
        "cfg":       2.0,
        "sampler":   "dpmpp_sde",
        "scheduler": "karras",
        "native":    (512, 512),
        "negative":  DEFAULT_NEGATIVE,
        "label":     "Realistic Vision V6  ·  Hyper",
        "desc":      "6 steps  ·  fast  ·  512px native  ·  SD 1.5 (Hyper)",
    },

    # ── SDXL checkpoints (native 1024×1024) ─────────────────────────────────────
    "juggernaut-xl": {
        "arch":      "sdxl",
        "ckpt":      "juggernautXL_ragnarokBy.safetensors",
        "steps":     30,
        "cfg":       6.0,
        "sampler":   "dpmpp_2m",
        "scheduler": "karras",
        "native":    (1024, 1024),
        "negative":  DEFAULT_NEGATIVE,
        "label":     "Juggernaut XL  ·  photoreal",
        "desc":      "30 steps  ·  1024px native  ·  SDXL",
    },
    "paragon-xl": {
        "arch":      "sdxl",
        "ckpt":      "paragonV10_v10VAE.safetensors",
        "steps":     30,
        "cfg":       6.0,
        "sampler":   "dpmpp_2m",
        "scheduler": "karras",
        "native":    (1024, 1024),
        "negative":  DEFAULT_NEGATIVE,
        "label":     "Paragon V10  ·  general",
        "desc":      "30 steps  ·  1024px native  ·  SDXL",
    },
}

# ── Ollama prompt-enhancement system prompts ───────────────────────────────────
# One template per model family — Flux and SD/SDXL want very different prompts.
# Both take a single {description} field.  Use enhance_prompt_for() to pick.

ENHANCE_PROMPT_FLUX = """You are an expert at writing image generation prompts for Flux, \
a state-of-the-art text-to-image model.

Flux works best with:
- Natural language sentences, not keyword lists
- Specific lighting descriptions (e.g. "lit by a single lantern casting warm orange light")
- Cinematographic language (e.g. "wide establishing shot", "close-up", "low angle")
- Concrete visual details rather than abstract quality boosters
- Style anchors like "in the style of a 1970s sci-fi paperback cover" or "photorealistic, 35mm film"
- Avoiding: "masterpiece", "highly detailed", "8k", "best quality" — Flux ignores these

Rewrite the following image description as an optimised Flux prompt.
Return ONLY the improved prompt text — no explanation, no preamble, no quotes.

Description to improve:
{description}
"""

ENHANCE_PROMPT_SD = """You are an expert at writing image generation prompts for \
Stable Diffusion (SD 1.5 / SDXL) models.

SD/SDXL work best with:
- Comma-separated keywords and short phrases, NOT long flowing sentences
- The main subject first, then descriptive details, then style, then quality tags
- Concrete style/medium tags (e.g. "oil painting", "35mm photo", "concept art", "cinematic lighting")
- Quality boosters DO help here — include a few like "highly detailed, masterpiece, best quality, sharp focus"
- Staying concise: the CLIP text encoder only reads ~77 tokens, so avoid filler
- Avoiding full grammatical sentences and narrative phrasing

Rewrite the following image description as an optimised Stable Diffusion prompt.
Return ONLY the improved comma-separated prompt text — no explanation, no preamble, no quotes.

Description to improve:
{description}
"""

# Backwards-compatible alias (Flux was the original sole target).
ENHANCE_PROMPT = ENHANCE_PROMPT_FLUX


def enhance_prompt_for(model_key: str) -> str:
    """Return the Ollama enhancement template matching the model's architecture."""
    arch = MODELS[model_key]["arch"]
    return ENHANCE_PROMPT_FLUX if arch == "flux" else ENHANCE_PROMPT_SD

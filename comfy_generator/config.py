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
]

# ── Model definitions ──────────────────────────────────────────────────────────
# Each key is a model_key string used throughout the app.

MODELS: dict[str, dict] = {
    "flux-schnell": {
        "unet":     "flux1-schnell.safetensors",
        "steps":    4,
        "guidance": 1.0,
        "label":    "Flux Schnell  ·  fast",
        "desc":     "4 steps  ·  ~2 min  ·  good quality",
    },
    "flux-dev": {
        "unet":     "flux1-dev.safetensors",
        "steps":    20,
        "guidance": 3.5,
        "label":    "Flux Dev  ·  quality",
        "desc":     "20 steps  ·  ~5 min  ·  best quality",
    },
}

# ── Ollama prompt-enhancement system prompt ────────────────────────────────────

ENHANCE_PROMPT = """You are an expert at writing image generation prompts for Flux, \
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

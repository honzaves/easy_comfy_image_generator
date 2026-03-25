# ComfyUI Generator

A PyQt6 desktop application for generating and inpainting images via a local [ComfyUI](https://github.com/comfyanonymous/ComfyUI) instance running Flux, with optional prompt enhancement powered by [Ollama](https://ollama.com).

---

## Features

| Feature | Details |
|---|---|
| **Text-to-image generation** | Flux Schnell (fast) or Flux Dev (quality) |
| **Prompt enhancement** | Sends your rough description to Ollama (Gemma 3 27B by default) and rewrites it as an optimised Flux prompt |
| **Inpainting** | Paint a mask over any region of a generated or imported image, describe what should replace it, and regenerate |
| **Open from disk** | Load any PNG / JPEG / WebP into the viewer and use the Improve tool on it |
| **Batch generation** | Generate 1–6 images in one click; thumbnail strip lets you browse results |
| **Token counter** | Live T5-XXL token estimate with colour-coded bar and CLIP-L overflow warning |
| **9 resolution presets** | From 512×512 to 1920×1088 |

---

## Requirements

- macOS (the *Reveal in Finder* button uses `open -R`; everything else is cross-platform)
- Python 3.11+
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) running locally on `http://127.0.0.1:8188`
- The following models placed in ComfyUI's model folders:
  - `models/unet/flux1-schnell.safetensors` (or `flux1-dev.safetensors`)
  - `models/vae/ae.safetensors`
  - `models/clip/clip_l.safetensors`
  - `models/clip/t5xxl_fp16.safetensors`
- [Ollama](https://ollama.com) running locally on `http://localhost:11434` with `gemma3:27b` pulled  
  *(optional — only needed for the ✦ Enhance button)*

---

## Installation

```bash
# 1. Clone or unzip the project
cd comfy_generator

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run
python main.py
```

Or install as an editable package:

```bash
pip install -e .
comfy-generator
```

---

## Configuration

All tuneable settings live in **`comfy_generator/config.py`**:

| Constant | Default | Description |
|---|---|---|
| `COMFYUI_URL` | `http://127.0.0.1:8188` | ComfyUI server address |
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Ollama server address |
| `OLLAMA_MODEL` | `gemma3:27b` | Model used for prompt enhancement |
| `MODELS` | Schnell + Dev | Add / remove Flux model variants here |
| `RESOLUTIONS` | 9 presets | Add custom width×height pairs here |

---

## Project layout

```
comfy_generator/
├── comfy_generator/        # Python package
│   ├── __init__.py
│   ├── __main__.py         # python -m comfy_generator
│   ├── config.py           # Constants, model definitions, prompts
│   ├── styles.py           # Dark palette + Qt stylesheet
│   ├── workflows.py        # ComfyUI workflow dict builders
│   ├── workers.py          # QObject background workers (network I/O)
│   ├── widgets.py          # Custom Qt widgets
│   └── main_window.py      # MainWindow — UI assembly + event handlers
├── main.py                 # Entry point
├── README.md
├── CLAUDE.md               # Notes for AI-assisted development
├── pyproject.toml
└── requirements.txt
```

---

## Usage

1. Start ComfyUI and (optionally) Ollama.
2. Launch the app: `python main.py`.
3. Choose a **model** and **resolution** in the left panel.
4. Type a description in the **Prompt** field.
5. Optionally click **✦ Enhance** to have Ollama rewrite it for Flux.
6. Click **⚡ Generate**.  
   Images are saved to `generated_images/` next to `main.py`.
7. To inpaint: click **✏ Improve Image**, paint a mask, describe the change, click **⚡ Generate Improvement**.

---

## License

MIT

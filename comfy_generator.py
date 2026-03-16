#!/usr/bin/env python3
"""
ComfyUI Image Generator — PyQt6
Generates images via ComfyUI/Flux with Ollama prompt enhancement.
Supports inpainting: click "Improve Image", draw a mask, describe changes.

Usage:
    pip install PyQt6 requests
    python comfy_generator.py
"""

import math
import os
import random
import sys
import threading
import time
from pathlib import Path

import requests
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QBuffer, QIODevice, QRect
from PyQt6.QtGui import (QColor, QFont, QIcon, QPalette, QPixmap, QImage,
                          QPainter, QBrush, QPen)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QHBoxLayout, QLabel, QTextEdit, QPushButton,
                              QButtonGroup, QRadioButton, QFrame, QScrollArea,
                              QSizePolicy, QProgressBar, QGridLayout,
                              QSplitter, QSlider, QFileDialog)


# ── Config ─────────────────────────────────────────────────────────────────────

COMFYUI_URL  = "http://127.0.0.1:8188"
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma3:27b"

RESOLUTIONS = [
    ("512×512",    512,   512,  "1:1"),
    ("768×512",    768,   512,  "3:2"),
    ("832×512",    832,   512,  "16:9"),
    ("1024×576",   1024,  576,  "16:9"),
    ("1024×1024",  1024,  1024, "1:1"),
    ("1216×768",   1216,  768,  "16:10"),
    ("1344×768",   1344,  768,  "16:9"),
    ("1536×640",   1536,  640,  "21:9"),
    ("1920×1088",  1920,  1088, "16:9 XL"),
]

MODELS = {
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

T5_LIMIT   = 512
CLIP_LIMIT = 77

ENHANCE_PROMPT = """You are an expert at writing image generation prompts for Flux, a state-of-the-art text-to-image model.

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

# ── Palette ────────────────────────────────────────────────────────────────────

BG          = "#0a0b12"
SURFACE     = "#10121c"
SURFACE2    = "#171928"
BORDER      = "#252840"
BORDER_HI   = "#3a3f6e"
TEXT        = "#c8c4d8"
TEXT_DIM    = "#5a5870"
TEXT_BRIGHT = "#eae6f8"
ACCENT      = "#6c63ff"
ACCENT_HI   = "#8880ff"
ACCENT2     = "#3fb68b"
OK          = "#3fb68b"
WARN        = "#e8a840"
DANGER      = "#e85040"
IMPROVE_CLR = "#e8a840"   # amber — improve / inpaint actions

STYLESHEET = f"""
QWidget {{
    background-color: {BG};
    color: {TEXT};
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 13px;
}}
QLabel {{ background: transparent; }}

QFrame#header {{
    background-color: {SURFACE};
    border-bottom: 1px solid {BORDER};
}}
QFrame#panel {{
    background-color: {SURFACE2};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}
QFrame#divider {{ background-color: {BORDER}; }}

QSplitter::handle {{ background-color: {BORDER}; width: 1px; }}

QRadioButton {{
    color: {TEXT};
    font-size: 13px;
    spacing: 8px;
}}
QRadioButton::indicator {{
    width: 14px; height: 14px;
    border-radius: 7px;
    border: 2px solid {BORDER_HI};
    background: {SURFACE2};
}}
QRadioButton::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QRadioButton:hover {{ color: {TEXT_BRIGHT}; }}

QPushButton#res_btn {{
    background-color: {SURFACE};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 7px 4px;
    font-size: 11px;
    font-weight: 600;
    min-width: 72px;
}}
QPushButton#res_btn:hover {{
    background-color: {SURFACE2};
    border-color: {BORDER_HI};
    color: {TEXT_BRIGHT};
}}
QPushButton#res_btn[selected="true"] {{
    background-color: {SURFACE2};
    border-color: {ACCENT};
    color: {TEXT_BRIGHT};
}}

QTextEdit#prompt_field {{
    background-color: {SURFACE2};
    color: {TEXT_BRIGHT};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 10px;
    font-size: 13px;
    selection-background-color: {ACCENT};
}}
QTextEdit#prompt_field:focus {{ border-color: {ACCENT}; }}

QPushButton#generate_btn {{
    background-color: {ACCENT};
    color: white;
    border: none;
    border-radius: 5px;
    padding: 11px 22px;
    font-size: 13px;
    font-weight: bold;
}}
QPushButton#generate_btn:hover {{ background-color: {ACCENT_HI}; }}
QPushButton#generate_btn:disabled {{
    background-color: {BORDER};
    color: {TEXT_DIM};
}}

QPushButton#enhance_btn {{
    background-color: transparent;
    color: {ACCENT2};
    border: 1px solid {ACCENT2};
    border-radius: 5px;
    padding: 11px 16px;
    font-size: 12px;
    font-weight: bold;
}}
QPushButton#enhance_btn:hover {{
    background-color: rgba(63,182,139,0.12);
}}
QPushButton#enhance_btn:disabled {{
    color: {TEXT_DIM};
    border-color: {BORDER};
}}

QPushButton#save_btn {{
    background-color: transparent;
    color: {ACCENT};
    border: 1px solid {BORDER_HI};
    border-radius: 5px;
    padding: 11px 16px;
    font-size: 12px;
}}
QPushButton#save_btn:hover {{
    background-color: rgba(108,99,255,0.1);
    border-color: {ACCENT};
}}
QPushButton#save_btn:disabled {{
    color: {TEXT_DIM};
    border-color: {BORDER};
}}

QPushButton#count_btn {{
    background-color: {SURFACE};
    color: {TEXT_DIM};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 2px;
    font-size: 12px;
    font-weight: bold;
    min-width: 32px;
}}
QPushButton#count_btn:hover {{
    border-color: {BORDER_HI};
    color: {TEXT};
}}
QPushButton#count_btn[selected="true"] {{
    background-color: {SURFACE2};
    border-color: {ACCENT};
    color: {TEXT_BRIGHT};
}}

/* ── Improve / inpaint buttons ─────────────────────────────────────────── */

QPushButton#open_image_btn {{
    background-color: transparent;
    color: {TEXT};
    border: 1px solid {BORDER_HI};
    border-radius: 5px;
    padding: 8px 18px;
    font-size: 12px;
}}
QPushButton#open_image_btn:hover {{
    background-color: rgba(255,255,255,0.05);
    border-color: {TEXT_DIM};
    color: {TEXT_BRIGHT};
}}

QPushButton#improve_btn {{
    background-color: transparent;
    color: {IMPROVE_CLR};
    border: 1px solid {IMPROVE_CLR};
    border-radius: 5px;
    padding: 8px 18px;
    font-size: 12px;
    font-weight: bold;
}}
QPushButton#improve_btn:hover {{
    background-color: rgba(232,168,64,0.12);
}}
QPushButton#improve_btn:disabled {{
    color: {TEXT_DIM};
    border-color: {BORDER};
}}

QPushButton#back_btn {{
    background-color: transparent;
    color: {TEXT};
    border: 1px solid {BORDER_HI};
    border-radius: 5px;
    padding: 8px 14px;
    font-size: 12px;
}}
QPushButton#back_btn:hover {{
    background-color: rgba(255,255,255,0.05);
    border-color: {TEXT_DIM};
}}

QPushButton#reset_mask_btn {{
    background-color: transparent;
    color: {TEXT_DIM};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 8px 14px;
    font-size: 12px;
}}
QPushButton#reset_mask_btn:hover {{
    color: {TEXT};
    border-color: {BORDER_HI};
}}

QProgressBar {{
    background-color: {BORDER};
    border: none;
    border-radius: 2px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 2px;
}}

QSlider::groove:horizontal {{
    background: {BORDER};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 2px;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 6px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER_HI};
    border-radius: 3px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent;
    height: 6px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER_HI};
    border-radius: 3px;
    min-width: 20px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    return max(0, math.ceil(len(text) / 4))

def token_color(used: int, limit: int) -> str:
    r = used / limit
    if r < 0.7: return OK
    if r < 0.9: return WARN
    return DANGER

def build_workflow(prompt, model_key, width, height):
    m    = MODELS[model_key]
    seed = random.randint(0, 2**32 - 1)
    return {
        "1": {"class_type": "UNETLoader",      "inputs": {"unet_name": m["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "VAELoader",        "inputs": {"vae_name": "ae.safetensors"}},
        "3": {"class_type": "DualCLIPLoader",   "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp16.safetensors", "type": "flux"}},
        "4": {"class_type": "CLIPTextEncode",   "inputs": {"text": prompt, "clip": ["3", 0]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "FluxGuidance",     "inputs": {"guidance": m["guidance"], "conditioning": ["4", 0]}},
        "7": {"class_type": "KSampler",         "inputs": {
                "model": ["1",0], "positive": ["6",0], "negative": ["4",0],
                "latent_image": ["5",0], "seed": seed, "steps": m["steps"],
                "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode",        "inputs": {"samples": ["7",0], "vae": ["2",0]}},
        "9": {"class_type": "SaveImage",        "inputs": {"filename_prefix": "comfy_gen", "images": ["8",0]}},
    }


def build_inpaint_workflow(prompt, model_key, image_name, mask_name):
    """Flux inpainting workflow using SetLatentNoiseMask.

    Key choices:
    - denoise=1.0   : mask=1.0 pixels get 100% noise → full regeneration in the
                      painted area; mask=0.0 pixels get 0% noise → original kept.
                      Any value < 1.0 leaves residual original signal in the mask
                      and the model barely changes anything.
    - steps=max(12) : schnell's 4 steps is far too few for inpainting quality;
                      12 gives clean results without being slow.
    - guidance=3.5  : schnell's guidance=1.0 is tuned for full-image speed;
                      inpainting needs stronger conditioning to actually follow
                      the prompt within a small masked region.
    """
    m    = MODELS[model_key]
    seed = random.randint(0, 2**32 - 1)
    inpaint_steps    = max(12, m["steps"])   # never fewer than 12 for inpainting
    inpaint_guidance = max(3.5, m["guidance"])
    return {
        "1":  {"class_type": "UNETLoader",        "inputs": {"unet_name": m["unet"], "weight_dtype": "default"}},
        "2":  {"class_type": "VAELoader",          "inputs": {"vae_name": "ae.safetensors"}},
        "3":  {"class_type": "DualCLIPLoader",     "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp16.safetensors", "type": "flux"}},
        "4":  {"class_type": "CLIPTextEncode",     "inputs": {"text": prompt, "clip": ["3", 0]}},
        "5":  {"class_type": "FluxGuidance",       "inputs": {"guidance": inpaint_guidance, "conditioning": ["4", 0]}},
        # Load source image and mask
        "10": {"class_type": "LoadImage",          "inputs": {"image": image_name}},
        "11": {"class_type": "LoadImageMask",      "inputs": {"image": mask_name, "channel": "red"}},
        # Encode source image, attach noise mask
        "12": {"class_type": "VAEEncode",          "inputs": {"pixels": ["10", 0], "vae": ["2", 0]}},
        "13": {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["12", 0], "mask": ["11", 0]}},
        # denoise=1.0: masked pixels start from pure noise, unmasked pixels stay put
        "7":  {"class_type": "KSampler",           "inputs": {
                "model": ["1", 0], "positive": ["5", 0], "negative": ["4", 0],
                "latent_image": ["13", 0], "seed": seed, "steps": inpaint_steps,
                "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "8":  {"class_type": "VAEDecode",          "inputs": {"samples": ["7", 0], "vae": ["2", 0]}},
        "9":  {"class_type": "SaveImage",          "inputs": {"filename_prefix": "comfy_inpaint", "images": ["8", 0]}},
    }


# ── Workers ────────────────────────────────────────────────────────────────────

class EnhanceWorker(QObject):
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, description):
        super().__init__()
        self.description = description

    def run(self):
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user",
                               "content": ENHANCE_PROMPT.format(description=self.description)}],
                "stream": False,
                "options": {"temperature": 0.5, "num_predict": 400},
            }
            r = requests.post(OLLAMA_URL, json=payload, timeout=120)
            r.raise_for_status()
            result = r.json()["message"]["content"].strip().strip('"\'`')
            self.finished.emit(result)
        except requests.exceptions.ConnectionError:
            self.error.emit("Cannot connect to Ollama — is it running?")
        except Exception as ex:
            self.error.emit(str(ex))


class GeneratorWorker(QObject):
    status      = pyqtSignal(str)
    tick        = pyqtSignal(int, int, int)
    image_ready = pyqtSignal(bytes, str, int, int)
    error       = pyqtSignal(str)

    def __init__(self, prompt, model_key, width, height, count=1):
        super().__init__()
        self.prompt    = prompt
        self.model_key = model_key
        self.width     = width
        self.height    = height
        self.count     = count
        self._active   = True

    def _generate_one(self) -> tuple[bytes, str]:
        wf  = build_workflow(self.prompt, self.model_key, self.width, self.height)
        r   = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": wf}, timeout=15)
        r.raise_for_status()
        pid = r.json()["prompt_id"]

        filename = subfolder = img_type = None
        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                h = requests.get(f"{COMFYUI_URL}/history/{pid}", timeout=10).json()
                if pid in h:
                    for node in h[pid].get("outputs", {}).values():
                        if "images" in node:
                            img       = node["images"][0]
                            filename  = img["filename"]
                            subfolder = img.get("subfolder", "")
                            img_type  = img.get("type", "output")
                            break
            except Exception:
                pass
            if filename:
                break
            time.sleep(2)

        if not filename:
            raise RuntimeError("Timed out waiting for image")

        url         = f"{COMFYUI_URL}/view?filename={filename}&subfolder={subfolder}&type={img_type}"
        image_bytes = requests.get(url, timeout=30).content

        out_dir = Path("generated_images")
        out_dir.mkdir(exist_ok=True)
        ts   = time.strftime("%Y%m%d_%H%M%S")
        slug = self.prompt[:40].replace(" ", "_").replace("/", "")
        path = out_dir / f"{ts}_{self.model_key}_{self.width}x{self.height}_{slug}.png"
        path.write_bytes(image_bytes)
        return image_bytes, str(path)

    def run(self):
        t0 = time.time()

        def ticker():
            while self._active:
                self.tick.emit(int(time.time() - t0), self._current, self.count)
                time.sleep(1)
        self._current = 0
        threading.Thread(target=ticker, daemon=True).start()

        try:
            for i in range(1, self.count + 1):
                self._current = i
                self.status.emit(f"Image {i}/{self.count} — Submitting")
                image_bytes, path = self._generate_one()
                self.status.emit(f"Image {i}/{self.count} — Done")
                self.image_ready.emit(image_bytes, path, i, self.count)
        except requests.exceptions.ConnectionError:
            self.error.emit("Cannot connect to ComfyUI — is it running?")
        except Exception as ex:
            self.error.emit(str(ex))
        finally:
            self._active = False


class InpaintWorker(QObject):
    """Uploads image + mask to ComfyUI and runs an inpainting workflow."""
    status      = pyqtSignal(str)
    tick        = pyqtSignal(int, int, int)
    image_ready = pyqtSignal(bytes, str)
    error       = pyqtSignal(str)

    def __init__(self, image_bytes: bytes, mask_bytes: bytes, prompt: str, model_key: str):
        super().__init__()
        self.image_bytes = image_bytes
        self.mask_bytes  = mask_bytes
        self.prompt      = prompt
        self.model_key   = model_key
        self._active     = True

    def _upload(self, data: bytes, filename: str) -> str:
        """Upload a PNG to ComfyUI's input folder and return its server name."""
        r = requests.post(
            f"{COMFYUI_URL}/upload/image",
            files={"image": (filename, data, "image/png")},
            data={"type": "input", "overwrite": "true"},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["name"]

    def run(self):
        t0 = time.time()

        def ticker():
            while self._active:
                self.tick.emit(int(time.time() - t0), 1, 1)
                time.sleep(1)
        threading.Thread(target=ticker, daemon=True).start()

        try:
            self.status.emit("Uploading source image")
            ts         = time.strftime("%Y%m%d_%H%M%S")
            image_name = self._upload(self.image_bytes, f"src_{ts}.png")

            self.status.emit("Uploading mask")
            mask_name  = self._upload(self.mask_bytes, f"mask_{ts}.png")

            self.status.emit("Generating inpainted image")
            wf  = build_inpaint_workflow(self.prompt, self.model_key, image_name, mask_name)
            r   = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": wf}, timeout=15)
            r.raise_for_status()
            pid = r.json()["prompt_id"]

            filename = subfolder = img_type = None
            deadline = time.time() + 600
            while time.time() < deadline:
                try:
                    h = requests.get(f"{COMFYUI_URL}/history/{pid}", timeout=10).json()
                    if pid in h:
                        for node in h[pid].get("outputs", {}).values():
                            if "images" in node:
                                img       = node["images"][0]
                                filename  = img["filename"]
                                subfolder = img.get("subfolder", "")
                                img_type  = img.get("type", "output")
                                break
                except Exception:
                    pass
                if filename:
                    break
                time.sleep(2)

            if not filename:
                raise RuntimeError("Timed out waiting for inpainted image")

            url         = f"{COMFYUI_URL}/view?filename={filename}&subfolder={subfolder}&type={img_type}"
            result_bytes = requests.get(url, timeout=30).content

            out_dir = Path("generated_images")
            out_dir.mkdir(exist_ok=True)
            slug = self.prompt[:30].replace(" ", "_").replace("/", "")
            path = out_dir / f"{ts}_inpaint_{slug}.png"
            path.write_bytes(result_bytes)

            self.image_ready.emit(result_bytes, str(path))
        except requests.exceptions.ConnectionError:
            self.error.emit("Cannot connect to ComfyUI — is it running?")
        except Exception as ex:
            self.error.emit(str(ex))
        finally:
            self._active = False


# ── Custom widgets ─────────────────────────────────────────────────────────────

class TokenBar(QWidget):
    def __init__(self):
        super().__init__()
        self.used = 0
        self.setFixedHeight(6)

    def set_used(self, used):
        self.used = used
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.setBrush(QBrush(QColor(BORDER)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0, 0, w, h, 3, 3)
        fw = int(w * min(self.used / T5_LIMIT, 1.0))
        if fw > 0:
            p.setBrush(QBrush(QColor(token_color(self.used, T5_LIMIT))))
            p.drawRoundedRect(0, 0, fw, h, 3, 3)


class ImageDisplay(QLabel):
    """QLabel-based image display — word wrap for placeholder, scrollable via parent QScrollArea."""
    def __init__(self):
        super().__init__("image will appear here")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"color: {BORDER_HI}; font-size: 12px; background-color: {SURFACE};")
        self.setMinimumSize(400, 300)
        self._pixmap_orig = None

    def set_image(self, data: bytes):
        self._pixmap_orig = QPixmap.fromImage(QImage.fromData(data))
        self.setText("")
        self._refresh()

    def _refresh(self):
        if self._pixmap_orig:
            scaled = self._pixmap_orig.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)

    def resizeEvent(self, event):
        self._refresh()
        super().resizeEvent(event)


class MaskCanvas(QWidget):
    """
    Displays an image and lets the user paint a mask over it.

    Drawing behaviour
    -----------------
    - Left-click + drag  → paint (red overlay = region to inpaint)
    - Brush size         → controlled via set_brush_size()
    - Cursor             → circle showing current brush footprint

    Internals
    ---------
    Two QImages are kept in sync at the *original image resolution*:
      _mask_bw    — Grayscale8, white = inpaint, black = keep  (exported to ComfyUI)
      _mask_color — ARGB32, semi-transparent red overlay        (displayed on screen)
    """

    def __init__(self):
        super().__init__()
        self._source_pixmap = None
        self._mask_bw       = None
        self._mask_color    = None
        self._brush_size    = 40      # widget-space pixels
        self._drawing       = False
        self._cursor_pos    = None
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setStyleSheet(f"background-color: {SURFACE}; border-radius: 8px;")

    # ── Public API ────────────────────────────────────────────────────────────

    def set_image(self, data: bytes):
        self._source_pixmap = QPixmap.fromImage(QImage.fromData(data))
        w, h = self._source_pixmap.width(), self._source_pixmap.height()
        self._mask_bw = QImage(w, h, QImage.Format.Format_Grayscale8)
        self._mask_bw.fill(0)
        self._mask_color = QImage(w, h, QImage.Format.Format_ARGB32)
        self._mask_color.fill(QColor(0, 0, 0, 0))
        self.update()

    def set_brush_size(self, size: int):
        self._brush_size = size

    def reset_mask(self):
        if self._mask_bw:
            self._mask_bw.fill(0)
            self._mask_color.fill(QColor(0, 0, 0, 0))
            self.update()

    def get_mask_bytes(self) -> bytes:
        """Return the mask as a grayscale PNG (white = region to inpaint)."""
        if not self._mask_bw:
            return b""
        buf = QBuffer()
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        self._mask_bw.save(buf, "PNG")
        return bytes(buf.data())

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _layout(self):
        """Return (QRect, scale) describing how the image fits the widget, or None."""
        if not self._source_pixmap:
            return None
        pw, ph = self._source_pixmap.width(), self._source_pixmap.height()
        ww, wh = self.width(), self.height()
        if pw == 0 or ph == 0 or ww == 0 or wh == 0:
            return None
        scale = min(ww / pw, wh / ph)
        sw, sh = int(pw * scale), int(ph * scale)
        x = (ww - sw) // 2
        y = (wh - sh) // 2
        return QRect(x, y, sw, sh), scale

    def _to_image_coords(self, pos):
        """Map a widget QPoint to image-space (ix, iy), or None if outside image."""
        info = self._layout()
        if not info or not self._mask_bw:
            return None
        rect, scale = info
        ix = int((pos.x() - rect.x()) / scale)
        iy = int((pos.y() - rect.y()) / scale)
        w, h = self._mask_bw.width(), self._mask_bw.height()
        if 0 <= ix < w and 0 <= iy < h:
            return ix, iy
        return None

    # ── Drawing ────────────────────────────────────────────────────────────────

    def _paint_stroke(self, pos):
        coords = self._to_image_coords(pos)
        info   = self._layout()
        if coords is None or info is None or self._mask_bw is None:
            return
        ix, iy = coords
        _, scale = info
        # Convert widget-space brush diameter to image-space radius.
        # _brush_size is the on-screen diameter; dividing by scale converts to
        # image pixels; halving gives radius.  (the old code used //, which also
        # floor-divides by 2 but in the wrong order, producing 2× too large a mask)
        r = max(3, int(self._brush_size / scale / 2))
        diameter = r * 2

        # Export mask: white strokes
        p1 = QPainter(self._mask_bw)
        p1.setBrush(QBrush(QColor(255, 255, 255)))
        p1.setPen(Qt.PenStyle.NoPen)
        p1.drawEllipse(ix - r, iy - r, diameter, diameter)
        p1.end()

        # Display overlay: semi-transparent red strokes
        p2 = QPainter(self._mask_color)
        p2.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        p2.setBrush(QBrush(QColor(230, 75, 55, 165)))
        p2.setPen(Qt.PenStyle.NoPen)
        p2.drawEllipse(ix - r, iy - r, diameter, diameter)
        p2.end()

    # ── Mouse events ──────────────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drawing = True
            self._paint_stroke(event.pos())
            self.update()

    def mouseMoveEvent(self, event):
        self._cursor_pos = event.pos()
        if self._drawing:
            self._paint_stroke(event.pos())
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drawing = False

    def leaveEvent(self, event):
        self._cursor_pos = None
        self.update()

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not self._source_pixmap:
            p.setPen(QColor(BORDER_HI))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded")
            return

        info = self._layout()
        if not info:
            return
        rect, _ = info

        # 1 · Source image
        p.drawPixmap(rect, self._source_pixmap)

        # 2 · Red mask overlay (scaled from image-space to widget-space)
        if self._mask_color:
            overlay_px = QPixmap.fromImage(self._mask_color).scaled(
                rect.size(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            p.drawPixmap(rect.topLeft(), overlay_px)

        # 3 · Brush cursor circle (widget-space)
        if self._cursor_pos:
            r = self._brush_size // 2
            pen = QPen(QColor(255, 255, 255, 200))
            pen.setWidth(2)
            p.setPen(pen)
            p.setBrush(QBrush(QColor(230, 75, 55, 50)))
            p.drawEllipse(
                self._cursor_pos.x() - r,
                self._cursor_pos.y() - r,
                self._brush_size,
                self._brush_size,
            )


# ── Main window ────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ComfyUI Generator")
        self.resize(1100, 820)
        self.setMinimumSize(900, 660)
        self._model               = "flux-schnell"
        self._res_index           = 6
        self._count               = 1
        self._thread              = None
        self._worker              = None
        self._ethread             = None
        self._eworker             = None
        self._ithread             = None
        self._iworker             = None
        self._saved_path          = None
        self._status_msg          = ""
        self._thumbnails          = []
        self._current_image_bytes = None   # bytes of the image currently shown
        self._build_ui()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _section_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 10px; font-weight: bold;"
            "margin-top: 4px; margin-bottom: 2px;"
        )
        return lbl

    def _set_busy(self, busy, label=""):
        """Disable/enable action buttons during any background operation."""
        self.generate_btn.setEnabled(not busy)
        self.enhance_btn.setEnabled(not busy)
        self.open_image_btn.setEnabled(not busy)
        self.improve_btn.setEnabled(not busy and self._current_image_bytes is not None)
        self.inpaint_generate_btn.setEnabled(not busy)
        self.progress_bar.setVisible(busy)
        if not busy:
            self.status_dot.setText("● READY")
            self.status_dot.setStyleSheet(
                f"color: {OK}; font-size: 10px; font-weight: bold;")
        if label:
            self.elapsed_label.setText(label)
            self.elapsed_label.setStyleSheet(
                f"color: {TEXT_DIM}; font-size: 10px;")

    # ── Build UI ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)
        vbox.addWidget(self._make_header())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.addWidget(self._make_left())
        splitter.addWidget(self._make_right())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([370, 730])
        vbox.addWidget(splitter, stretch=1)

    def _make_header(self):
        frame = QFrame()
        frame.setObjectName("header")
        frame.setFixedHeight(70)
        h = QHBoxLayout(frame)
        h.setContentsMargins(28, 0, 28, 0)

        col = QVBoxLayout()
        col.setSpacing(2)
        sub = QLabel("IMAGE GENERATOR")
        sub.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px; font-weight: bold;")
        title = QLabel("ComfyUI / Flux")
        title.setStyleSheet(f"color: {TEXT_BRIGHT}; font-size: 20px; font-weight: 300;")
        col.addWidget(sub)
        col.addWidget(title)

        self.status_dot = QLabel("● READY")
        self.status_dot.setStyleSheet(
            f"color: {OK}; font-size: 10px; font-weight: bold;")

        h.addLayout(col)
        h.addStretch()
        h.addWidget(self.status_dot)
        return frame

    def _make_left(self):
        w = QWidget()
        w.setFixedWidth(370)
        v = QVBoxLayout(w)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(6)

        v.addWidget(self._section_label("MODEL"))
        v.addWidget(self._make_model_panel())
        v.addSpacing(10)

        v.addWidget(self._section_label("RESOLUTION"))
        v.addWidget(self._make_res_grid())
        self.pixel_label = QLabel()
        self.pixel_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px;")
        self._update_pixel_label()
        v.addWidget(self.pixel_label)
        v.addSpacing(10)

        v.addWidget(self._section_label("PROMPT TOKENS  (T5-XXL / 512 max)"))
        v.addWidget(self._make_token_panel())
        v.addSpacing(10)

        v.addWidget(self._section_label("NUMBER OF IMAGES"))
        v.addWidget(self._make_count_selector())
        v.addSpacing(10)

        v.addWidget(self._section_label("PROMPT"))
        self.prompt_field = QTextEdit()
        self.prompt_field.setObjectName("prompt_field")
        self.prompt_field.setPlaceholderText(
            "Describe the image… then hit ✦ Enhance to let Ollama rewrite it for Flux.")
        self.prompt_field.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.prompt_field.setMinimumHeight(100)
        self.prompt_field.setMaximumHeight(160)
        self.prompt_field.textChanged.connect(self._on_prompt_changed)
        v.addWidget(self.prompt_field)
        v.addSpacing(8)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.generate_btn = QPushButton("⚡  Generate")
        self.generate_btn.setObjectName("generate_btn")
        self.generate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.generate_btn.clicked.connect(self._on_generate)

        self.enhance_btn = QPushButton("✦  Enhance")
        self.enhance_btn.setObjectName("enhance_btn")
        self.enhance_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.enhance_btn.setToolTip(
            "Send your description to Ollama and rewrite it as an optimised Flux prompt")
        self.enhance_btn.clicked.connect(self._on_enhance)

        self.save_btn = QPushButton("📂")
        self.save_btn.setObjectName("save_btn")
        self.save_btn.setFixedWidth(44)
        self.save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_btn.setToolTip("Reveal in Finder")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._reveal_in_finder)

        btn_row.addWidget(self.generate_btn, stretch=2)
        btn_row.addWidget(self.enhance_btn,  stretch=2)
        btn_row.addWidget(self.save_btn,     stretch=0)
        v.addLayout(btn_row)
        v.addSpacing(6)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setVisible(False)
        v.addWidget(self.progress_bar)

        self.elapsed_label = QLabel("")
        self.elapsed_label.setWordWrap(True)
        self.elapsed_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px;")
        v.addWidget(self.elapsed_label)

        v.addStretch()
        return w

    def _make_count_selector(self):
        w   = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(5)
        self.count_buttons = []
        for n in range(1, 7):
            btn = QPushButton(str(n))
            btn.setObjectName("count_btn")
            btn.setProperty("selected", n == self._count)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(32)
            btn.clicked.connect(lambda _, v=n: self._on_count_selected(v))
            self.count_buttons.append(btn)
            row.addWidget(btn)
        row.addStretch()
        return w

    def _make_model_panel(self):
        frame = QFrame()
        frame.setObjectName("panel")
        v = QVBoxLayout(frame)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(6)

        self.model_group = QButtonGroup(self)
        for key, info in MODELS.items():
            rb = QRadioButton(info["label"])
            rb.setProperty("model_key", key)
            if key == self._model:
                rb.setChecked(True)
            self.model_group.addButton(rb)
            v.addWidget(rb)

        self.model_desc = QLabel(MODELS[self._model]["desc"])
        self.model_desc.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 11px; font-style: italic;")
        v.addWidget(self.model_desc)
        self.model_group.buttonClicked.connect(self._on_model_changed)
        return frame

    def _make_res_grid(self):
        w    = QWidget()
        grid = QGridLayout(w)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(5)
        self.res_buttons = []
        for i, (label, pw, ph, ratio) in enumerate(RESOLUTIONS):
            btn = QPushButton(f"{pw}×{ph}\n{ratio}")
            btn.setObjectName("res_btn")
            btn.setProperty("selected", i == self._res_index)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(44)
            btn.clicked.connect(lambda _, idx=i: self._on_res_selected(idx))
            self.res_buttons.append(btn)
            grid.addWidget(btn, i // 3, i % 3)
        return w

    def _make_token_panel(self):
        frame = QFrame()
        frame.setObjectName("panel")
        v = QVBoxLayout(frame)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(5)

        row = QHBoxLayout()
        self.token_used = QLabel("0")
        self.token_used.setStyleSheet(
            f"color: {OK}; font-size: 20px; font-weight: bold;")
        self.token_remaining = QLabel(f"/ {T5_LIMIT}  (512 remaining)")
        self.token_remaining.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        row.addWidget(self.token_used)
        row.addSpacing(4)
        row.addWidget(self.token_remaining)
        row.addStretch()
        v.addLayout(row)

        self.token_bar = TokenBar()
        v.addWidget(self.token_bar)

        self.token_warn = QLabel("")
        self.token_warn.setStyleSheet(f"color: {WARN}; font-size: 10px;")
        v.addWidget(self.token_warn)
        return frame

    def _make_right(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(8)

        # ── Normal view ───────────────────────────────────────────────────────
        self.normal_view = QWidget()
        nv = QVBoxLayout(self.normal_view)
        nv.setContentsMargins(0, 0, 0, 0)
        nv.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(f"background-color: {SURFACE}; border-radius: 8px;")
        self.image_display = ImageDisplay()
        scroll.setWidget(self.image_display)
        nv.addWidget(scroll, stretch=1)

        self.image_info = QLabel("")
        self.image_info.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px;")
        self.image_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_info.setWordWrap(True)
        nv.addWidget(self.image_info)

        # Image action row: open from disk · improve generated image
        improve_row = QHBoxLayout()
        improve_row.setContentsMargins(0, 0, 0, 0)
        improve_row.setSpacing(6)

        self.open_image_btn = QPushButton("📁  Open Image…")
        self.open_image_btn.setObjectName("open_image_btn")
        self.open_image_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_image_btn.setToolTip(
            "Load an image from disk and use the Improve tool on it")
        self.open_image_btn.clicked.connect(self._on_open_image)

        self.improve_btn = QPushButton("✏  Improve Image")
        self.improve_btn.setObjectName("improve_btn")
        self.improve_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.improve_btn.setEnabled(False)
        self.improve_btn.setToolTip(
            "Draw a mask over the region you want to change, then describe the improvement")
        self.improve_btn.clicked.connect(self._on_improve_clicked)

        improve_row.addStretch()
        improve_row.addWidget(self.open_image_btn)
        improve_row.addWidget(self.improve_btn)
        nv.addLayout(improve_row)

        v.addWidget(self.normal_view, stretch=1)

        # ── Improve / inpaint view ────────────────────────────────────────────
        self.improve_view = QWidget()
        self.improve_view.setVisible(False)
        iv = QVBoxLayout(self.improve_view)
        iv.setContentsMargins(0, 0, 0, 0)
        iv.setSpacing(8)

        # Mask canvas (drawing area)
        self.mask_canvas = MaskCanvas()
        iv.addWidget(self.mask_canvas, stretch=1)

        # Instruction strip
        instr = QLabel(
            "🖌  Paint over the region to improve  ·  use the slider to resize the brush")
        instr.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 10px; padding: 2px 0;")
        instr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        iv.addWidget(instr)

        # Controls row: back · reset · brush size
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(8)

        back_btn = QPushButton("← Back")
        back_btn.setObjectName("back_btn")
        back_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        back_btn.clicked.connect(self._on_back_from_improve)

        reset_btn = QPushButton("⟳  Reset Mask")
        reset_btn.setObjectName("reset_mask_btn")
        reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        reset_btn.clicked.connect(self._on_reset_mask)

        brush_label = QLabel("Brush size:")
        brush_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")

        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(10, 150)
        self.brush_slider.setValue(40)
        self.brush_slider.setFixedWidth(110)
        self.brush_slider.valueChanged.connect(self.mask_canvas.set_brush_size)

        ctrl_row.addWidget(back_btn)
        ctrl_row.addWidget(reset_btn)
        ctrl_row.addStretch()
        ctrl_row.addWidget(brush_label)
        ctrl_row.addWidget(self.brush_slider)
        iv.addLayout(ctrl_row)

        # Divider
        div = QFrame()
        div.setObjectName("divider")
        div.setFixedHeight(1)
        iv.addWidget(div)

        # Inpaint prompt label
        iv.addWidget(self._section_label("WHAT TO CHANGE IN THE SELECTED REGION"))

        self.inpaint_prompt = QTextEdit()
        self.inpaint_prompt.setObjectName("prompt_field")
        self.inpaint_prompt.setPlaceholderText(
            "Describe what should appear in the painted area…  "
            "e.g. \"a bright red door with a brass knocker\"")
        self.inpaint_prompt.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.inpaint_prompt.setFixedHeight(72)
        iv.addWidget(self.inpaint_prompt)

        # Generate improvement button
        inpaint_btn_row = QHBoxLayout()
        self.inpaint_generate_btn = QPushButton("⚡  Generate Improvement")
        self.inpaint_generate_btn.setObjectName("generate_btn")
        self.inpaint_generate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.inpaint_generate_btn.clicked.connect(self._on_inpaint_generate)
        inpaint_btn_row.addStretch()
        inpaint_btn_row.addWidget(self.inpaint_generate_btn)
        iv.addLayout(inpaint_btn_row)

        v.addWidget(self.improve_view, stretch=1)

        # ── Thumbnail strip (normal view only) ────────────────────────────────
        self.thumb_scroll = QScrollArea()
        self.thumb_scroll.setWidgetResizable(True)
        self.thumb_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.thumb_scroll.setFixedHeight(90)
        self.thumb_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.thumb_scroll.setStyleSheet("background-color: transparent;")
        self.thumb_container = QWidget()
        self.thumb_layout = QHBoxLayout(self.thumb_container)
        self.thumb_layout.setContentsMargins(0, 0, 0, 0)
        self.thumb_layout.setSpacing(6)
        self.thumb_layout.addStretch()
        self.thumb_scroll.setWidget(self.thumb_container)
        self.thumb_scroll.setVisible(False)
        v.addWidget(self.thumb_scroll)

        return w

    # ── Event handlers ─────────────────────────────────────────────────────────

    def _on_count_selected(self, n):
        self._count = n
        for i, btn in enumerate(self.count_buttons):
            btn.setProperty("selected", (i + 1) == n)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _on_model_changed(self, btn):
        self._model = btn.property("model_key")
        self.model_desc.setText(MODELS[self._model]["desc"])

    def _on_res_selected(self, idx):
        self._res_index = idx
        for i, btn in enumerate(self.res_buttons):
            btn.setProperty("selected", i == idx)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        self._update_pixel_label()

    def _update_pixel_label(self):
        _, w, h, _ = RESOLUTIONS[self._res_index]
        self.pixel_label.setText(
            f"{w * h:,} pixels  ·  {w * h / 1_000_000:.1f} MP")

    def _on_prompt_changed(self):
        text = self.prompt_field.toPlainText()
        used = estimate_tokens(text)
        rem  = T5_LIMIT - used
        col  = token_color(used, T5_LIMIT)
        self.token_used.setText(str(used))
        self.token_used.setStyleSheet(
            f"color: {col}; font-size: 20px; font-weight: bold;")
        self.token_remaining.setText(f"/ {T5_LIMIT}  ({max(0, rem)} remaining)")
        self.token_bar.set_used(used)
        self.token_warn.setText(
            f"⚠  exceeds CLIP-L limit ({CLIP_LIMIT} tokens) — T5 only"
            if used > CLIP_LIMIT else "")

    # ── Enhance ────────────────────────────────────────────────────────────────

    def _on_enhance(self):
        description = self.prompt_field.toPlainText().strip()
        if not description:
            self.elapsed_label.setText("Enter a description first.")
            return

        self._set_busy(True)
        self.status_dot.setText("● ENHANCING")
        self.status_dot.setStyleSheet(
            f"color: {ACCENT2}; font-size: 10px; font-weight: bold;")
        self.elapsed_label.setText("Sending to Ollama…")

        self._ethread = QThread()
        self._eworker = EnhanceWorker(description)
        self._eworker.moveToThread(self._ethread)
        self._ethread.started.connect(self._eworker.run)
        self._eworker.finished.connect(self._on_enhance_done)
        self._eworker.error.connect(self._on_enhance_error)
        self._eworker.finished.connect(self._ethread.quit)
        self._eworker.error.connect(self._ethread.quit)
        self._ethread.start()

    def _on_enhance_done(self, improved_prompt):
        self.prompt_field.setPlainText(improved_prompt)
        cursor = self.prompt_field.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.prompt_field.setTextCursor(cursor)
        self._set_busy(False, "✦ Prompt enhanced by Ollama")
        self.elapsed_label.setStyleSheet(f"color: {ACCENT2}; font-size: 10px;")

    def _on_enhance_error(self, msg):
        self._set_busy(False, f"Enhance error: {msg}")
        self.elapsed_label.setStyleSheet(f"color: {DANGER}; font-size: 10px;")

    # ── Generate ───────────────────────────────────────────────────────────────

    def _on_generate(self):
        prompt = self.prompt_field.toPlainText().strip()
        if not prompt:
            self.elapsed_label.setText("Please enter a prompt first.")
            return

        _, width, height, _ = RESOLUTIONS[self._res_index]
        self._status_msg = ""
        self._thumbnails  = []
        self._saved_path  = None
        self.save_btn.setEnabled(False)

        while self.thumb_layout.count() > 1:
            item = self.thumb_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.thumb_scroll.setVisible(False)

        self._set_busy(True)

        self._thread = QThread()
        self._worker = GeneratorWorker(prompt, self._model, width, height, self._count)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.status.connect(self._on_worker_status)
        self._worker.tick.connect(self._on_worker_tick)
        self._worker.image_ready.connect(self._on_image_ready)
        self._worker.error.connect(self._on_error)
        self._worker.error.connect(self._thread.quit)
        self._worker.image_ready.connect(self._check_all_done)
        self._thread.start()

    def _on_worker_status(self, msg):
        self._status_msg = msg
        self.status_dot.setText(f"● {msg.upper()}")
        self.status_dot.setStyleSheet(
            f"color: {WARN}; font-size: 10px; font-weight: bold;")
        self.elapsed_label.setText(f"{msg}…")

    def _on_worker_tick(self, secs, current, total):
        self.elapsed_label.setText(f"{self._status_msg}…  {secs}s")

    def _on_image_ready(self, image_bytes, saved_path, index, total):
        self._thumbnails.append((image_bytes, saved_path))
        self._saved_path          = saved_path
        self._current_image_bytes = image_bytes

        self.image_display.set_image(image_bytes)
        _, w, h, ratio = RESOLUTIONS[self._res_index]
        self.image_info.setText(
            f"{w}×{h}  ·  {ratio}  ·  {MODELS[self._model]['label']}  ·  "
            f"Image {index}/{total}  ·  {Path(saved_path).name}")

        if total > 1:
            self.thumb_scroll.setVisible(True)
            thumb_btn = QPushButton()
            thumb_btn.setFixedSize(120, 76)
            thumb_btn.setStyleSheet(
                f"border: 2px solid {ACCENT}; border-radius: 4px; padding: 0;"
                if index == len(self._thumbnails)
                else f"border: 1px solid {BORDER}; border-radius: 4px; padding: 0;"
            )
            px = QPixmap.fromImage(QImage.fromData(image_bytes)).scaled(
                118, 72,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            thumb_btn.setIcon(QIcon(px))
            from PyQt6.QtCore import QSize
            thumb_btn.setIconSize(QSize(118, 72))
            idx = index - 1
            thumb_btn.clicked.connect(lambda _, i=idx: self._select_thumbnail(i))
            self.thumb_layout.insertWidget(self.thumb_layout.count() - 1, thumb_btn)

        self.save_btn.setEnabled(True)
        self.improve_btn.setEnabled(True)

    def _check_all_done(self, image_bytes, saved_path, index, total):
        if index == total:
            self._thread.quit()
            self._set_busy(
                False,
                f"{total} image{'s' if total > 1 else ''} saved to generated_images/")

    def _select_thumbnail(self, idx):
        if 0 <= idx < len(self._thumbnails):
            image_bytes, path         = self._thumbnails[idx]
            self._current_image_bytes = image_bytes
            self.image_display.set_image(image_bytes)
            self._saved_path = path
            _, w, h, ratio = RESOLUTIONS[self._res_index]
            self.image_info.setText(
                f"{w}×{h}  ·  {ratio}  ·  {MODELS[self._model]['label']}  ·  "
                f"Image {idx+1}/{len(self._thumbnails)}  ·  {Path(path).name}")
            for i in range(self.thumb_layout.count() - 1):
                item = self.thumb_layout.itemAt(i)
                if item and item.widget():
                    item.widget().setStyleSheet(
                        f"border: 2px solid {ACCENT}; border-radius: 4px; padding: 0;"
                        if i == idx
                        else f"border: 1px solid {BORDER}; border-radius: 4px; padding: 0;"
                    )

    def _on_error(self, msg):
        self.elapsed_label.setText(f"Error: {msg}")
        self.elapsed_label.setStyleSheet(f"color: {DANGER}; font-size: 10px;")
        self.status_dot.setText("● ERROR")
        self.status_dot.setStyleSheet(
            f"color: {DANGER}; font-size: 10px; font-weight: bold;")
        self._set_busy(False)

    def _reveal_in_finder(self):
        if self._saved_path:
            os.system(f'open -R "{self._saved_path}"')

    # ── Open image from disk ───────────────────────────────────────────────────

    def _on_open_image(self):
        """Open a file dialog, load an image from disk into the preview pane."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.tif)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if not path:
            return   # user cancelled

        # Load via QImage so any format (JPEG, WebP, …) is normalised to PNG bytes
        qimg = QImage(path)
        if qimg.isNull():
            self.elapsed_label.setText(f"Could not load image: {Path(path).name}")
            self.elapsed_label.setStyleSheet(f"color: {DANGER}; font-size: 10px;")
            return

        buf = QBuffer()
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        qimg.save(buf, "PNG")
        image_bytes = bytes(buf.data())

        # Store and display exactly like a generated image, but skip thumbnails
        self._current_image_bytes = image_bytes
        self._saved_path          = path
        self.image_display.set_image(image_bytes)
        self.image_info.setText(
            f"{qimg.width()}×{qimg.height()}  ·  {Path(path).name}  ·  loaded from disk")
        self.improve_btn.setEnabled(True)
        self.save_btn.setEnabled(True)

        self.elapsed_label.setText(f"Loaded: {Path(path).name}")
        self.elapsed_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px;")
        self.status_dot.setText("● READY")
        self.status_dot.setStyleSheet(f"color: {OK}; font-size: 10px; font-weight: bold;")

    # ── Improve / inpaint ──────────────────────────────────────────────────────

    def _on_improve_clicked(self):
        """Switch to improve mode: load image into mask canvas, show inpaint panel."""
        if not self._current_image_bytes:
            return
        self.mask_canvas.set_image(self._current_image_bytes)
        self.mask_canvas.set_brush_size(self.brush_slider.value())
        self.normal_view.setVisible(False)
        self.thumb_scroll.setVisible(False)
        self.improve_view.setVisible(True)

    def _on_back_from_improve(self):
        """Return to the normal image view."""
        self.improve_view.setVisible(False)
        self.normal_view.setVisible(True)
        if len(self._thumbnails) > 1:
            self.thumb_scroll.setVisible(True)

    def _on_reset_mask(self):
        self.mask_canvas.reset_mask()

    def _on_inpaint_generate(self):
        """Validate inputs, then kick off InpaintWorker."""
        prompt = self.inpaint_prompt.toPlainText().strip()
        if not prompt:
            self.elapsed_label.setText("Describe what to change in the selected region first.")
            self.elapsed_label.setStyleSheet(f"color: {WARN}; font-size: 10px;")
            return

        mask_bytes = self.mask_canvas.get_mask_bytes()
        if not mask_bytes:
            self.elapsed_label.setText("No image loaded in the mask canvas.")
            return

        self._set_busy(True)
        self.status_dot.setText("● INPAINTING")
        self.status_dot.setStyleSheet(
            f"color: {IMPROVE_CLR}; font-size: 10px; font-weight: bold;")
        self.elapsed_label.setText("Uploading to ComfyUI…")

        self._ithread = QThread()
        self._iworker = InpaintWorker(
            self._current_image_bytes, mask_bytes, prompt, self._model)
        self._iworker.moveToThread(self._ithread)
        self._ithread.started.connect(self._iworker.run)
        self._iworker.status.connect(self._on_worker_status)
        self._iworker.tick.connect(self._on_worker_tick)
        self._iworker.image_ready.connect(self._on_inpaint_ready)
        self._iworker.error.connect(self._on_error)
        self._iworker.error.connect(self._ithread.quit)
        self._ithread.start()

    def _on_inpaint_ready(self, image_bytes, saved_path):
        """Inpainting finished — return to normal view and display result."""
        self._ithread.quit()

        # Switch back to normal view
        self._on_back_from_improve()

        # Update display with the improved image
        self._current_image_bytes = image_bytes
        self._saved_path          = saved_path
        self._thumbnails.append((image_bytes, saved_path))

        self.image_display.set_image(image_bytes)
        self.image_info.setText(
            f"Improved  ·  {Path(saved_path).name}")

        self.save_btn.setEnabled(True)
        self._set_busy(False, "✏ Improvement saved to generated_images/")
        self.elapsed_label.setStyleSheet(
            f"color: {IMPROVE_CLR}; font-size: 10px;")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(BG))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base,            QColor(SURFACE2))
    palette.setColor(QPalette.ColorRole.Text,            QColor(TEXT_BRIGHT))
    palette.setColor(QPalette.ColorRole.Button,          QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    w = MainWindow()
    w.show()
    sys.exit(app.exec())

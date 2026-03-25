"""
comfy_generator.workers
=======================
QObject subclasses that perform network I/O on background QThreads.

Each worker emits Qt signals; the main window connects to them to update the UI.
None of these classes import any widget — they are pure data/network workers.
"""

import threading
import time
from pathlib import Path

import requests
from PyQt6.QtCore import QObject, pyqtSignal

from .config import COMFYUI_URL, OLLAMA_URL, OLLAMA_MODEL, ENHANCE_PROMPT
from .workflows import build_workflow, build_inpaint_workflow


# ── Helpers ────────────────────────────────────────────────────────────────────

def _poll_comfy_history(pid: str, deadline: float) -> tuple[str, str, str]:
    """
    Poll ComfyUI's /history endpoint until *pid* appears in the output.

    Returns ``(filename, subfolder, img_type)`` or raises ``RuntimeError``
    if the deadline is exceeded.
    """
    filename = subfolder = img_type = None
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
            return filename, subfolder, img_type
        time.sleep(2)
    raise RuntimeError("Timed out waiting for image from ComfyUI")


def _fetch_comfy_image(filename: str, subfolder: str, img_type: str) -> bytes:
    """Download a finished image from ComfyUI's /view endpoint."""
    url = (
        f"{COMFYUI_URL}/view"
        f"?filename={filename}&subfolder={subfolder}&type={img_type}"
    )
    return requests.get(url, timeout=30).content


# ── Workers ────────────────────────────────────────────────────────────────────

class EnhanceWorker(QObject):
    """Sends a description to Ollama and emits the enhanced prompt text."""

    finished = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, description: str):
        super().__init__()
        self.description = description

    def run(self):
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "messages": [{
                    "role":    "user",
                    "content": ENHANCE_PROMPT.format(description=self.description),
                }],
                "stream":  False,
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
    """Submits one or more text-to-image jobs to ComfyUI."""

    status      = pyqtSignal(str)
    tick        = pyqtSignal(int, int, int)   # elapsed_secs, current, total
    image_ready = pyqtSignal(bytes, str, int, int)  # data, path, index, total
    error       = pyqtSignal(str)

    def __init__(
        self,
        prompt: str,
        model_key: str,
        width: int,
        height: int,
        count: int = 1,
    ):
        super().__init__()
        self.prompt    = prompt
        self.model_key = model_key
        self.width     = width
        self.height    = height
        self.count     = count
        self._active   = True
        self._current  = 0

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _generate_one(self) -> tuple[bytes, str]:
        wf  = build_workflow(self.prompt, self.model_key, self.width, self.height)
        r   = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": wf}, timeout=15)
        r.raise_for_status()
        pid = r.json()["prompt_id"]

        filename, subfolder, img_type = _poll_comfy_history(pid, time.time() + 600)
        image_bytes = _fetch_comfy_image(filename, subfolder, img_type)

        out_dir = Path("generated_images")
        out_dir.mkdir(exist_ok=True)
        ts   = time.strftime("%Y%m%d_%H%M%S")
        slug = self.prompt[:40].replace(" ", "_").replace("/", "")
        path = out_dir / f"{ts}_{self.model_key}_{self.width}x{self.height}_{slug}.png"
        path.write_bytes(image_bytes)
        return image_bytes, str(path)

    # ── Main entry ─────────────────────────────────────────────────────────────

    def run(self):
        t0 = time.time()

        def _ticker():
            while self._active:
                self.tick.emit(int(time.time() - t0), self._current, self.count)
                time.sleep(1)

        threading.Thread(target=_ticker, daemon=True).start()

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

    def __init__(
        self,
        image_bytes: bytes,
        mask_bytes: bytes,
        prompt: str,
        model_key: str,
    ):
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

        def _ticker():
            while self._active:
                self.tick.emit(int(time.time() - t0), 1, 1)
                time.sleep(1)

        threading.Thread(target=_ticker, daemon=True).start()

        try:
            ts = time.strftime("%Y%m%d_%H%M%S")

            self.status.emit("Uploading source image")
            image_name = self._upload(self.image_bytes, f"src_{ts}.png")

            self.status.emit("Uploading mask")
            mask_name = self._upload(self.mask_bytes, f"mask_{ts}.png")

            self.status.emit("Generating inpainted image")
            wf  = build_inpaint_workflow(
                self.prompt, self.model_key, image_name, mask_name
            )
            r   = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": wf}, timeout=15)
            r.raise_for_status()
            pid = r.json()["prompt_id"]

            filename, subfolder, img_type = _poll_comfy_history(pid, time.time() + 600)
            result_bytes = _fetch_comfy_image(filename, subfolder, img_type)

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

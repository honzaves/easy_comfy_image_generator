"""
comfy_generator.widgets
=======================
Reusable custom Qt widgets:

- ``TokenBar``     — thin coloured progress bar showing T5 token usage.
- ``ImageDisplay`` — QLabel that auto-scales a pixmap on resize.
- ``MaskCanvas``   — interactive paint surface for drawing inpaint masks.
"""

import math

from PyQt6.QtCore import Qt, QRect, QBuffer, QIODevice
from PyQt6.QtGui import (
    QColor, QImage, QPixmap, QPainter, QBrush, QPen,
)
from PyQt6.QtWidgets import QLabel, QWidget, QSizePolicy

from .config import T5_LIMIT
from .styles import BORDER, BORDER_HI, SURFACE, OK, WARN, DANGER, TEXT_DIM


# ── Helpers ────────────────────────────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """Rough token count: one token ≈ 4 characters."""
    return max(0, math.ceil(len(text) / 4))


def token_color(used: int, limit: int) -> str:
    """Return a colour string reflecting how close *used* is to *limit*."""
    ratio = used / limit
    if ratio < 0.7:
        return OK
    if ratio < 0.9:
        return WARN
    return DANGER


# ── TokenBar ───────────────────────────────────────────────────────────────────

class TokenBar(QWidget):
    """Thin horizontal bar that fills proportionally to T5 token usage."""

    def __init__(self):
        super().__init__()
        self.used = 0
        self.setFixedHeight(6)

    def set_used(self, used: int):
        self.used = used
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background track
        p.setBrush(QBrush(QColor(BORDER)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0, 0, w, h, 3, 3)

        # Filled portion
        fw = int(w * min(self.used / T5_LIMIT, 1.0))
        if fw > 0:
            p.setBrush(QBrush(QColor(token_color(self.used, T5_LIMIT))))
            p.drawRoundedRect(0, 0, fw, h, 3, 3)


# ── ImageDisplay ───────────────────────────────────────────────────────────────

class ImageDisplay(QLabel):
    """
    QLabel-based image viewer.

    Displays a placeholder message when no image is loaded, and keeps the
    pixmap scaled to fit the widget whenever it is resized.
    """

    def __init__(self):
        super().__init__("image will appear here")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            f"color: {BORDER_HI}; font-size: 12px; background-color: {SURFACE};"
        )
        self.setMinimumSize(400, 300)
        self._pixmap_orig: QPixmap | None = None

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


# ── MaskCanvas ─────────────────────────────────────────────────────────────────

class MaskCanvas(QWidget):
    """
    Interactive paint widget for creating inpaint masks.

    Drawing behaviour
    -----------------
    - Left-click + drag  → paint (red overlay = region to inpaint).
    - Brush size         → controlled via :meth:`set_brush_size`.
    - Cursor             → circle showing the current brush footprint.

    Internals
    ---------
    Two ``QImage`` objects are kept at the *original image resolution*:

    ``_mask_bw``
        ``Grayscale8`` — white = inpaint region, black = keep.
        Exported as PNG and uploaded to ComfyUI.

    ``_mask_color``
        ``ARGB32`` — semi-transparent red overlay painted on screen.
    """

    def __init__(self):
        super().__init__()
        self._source_pixmap: QPixmap | None = None
        self._mask_bw:       QImage  | None = None
        self._mask_color:    QImage  | None = None
        self._brush_size  = 40      # diameter in widget-space pixels
        self._drawing     = False
        self._cursor_pos  = None

        self.setMinimumSize(400, 300)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.setMouseTracking(True)
        self.setStyleSheet(f"background-color: {SURFACE}; border-radius: 8px;")

    # ── Public API ─────────────────────────────────────────────────────────────

    def set_image(self, data: bytes):
        """Load image bytes and reset any existing mask."""
        self._source_pixmap = QPixmap.fromImage(QImage.fromData(data))
        w = self._source_pixmap.width()
        h = self._source_pixmap.height()
        self._mask_bw    = QImage(w, h, QImage.Format.Format_Grayscale8)
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

    # ── Geometry helpers ───────────────────────────────────────────────────────

    def _layout(self) -> tuple[QRect, float] | None:
        """Return ``(draw_rect, scale)`` or ``None`` if no image is loaded."""
        if not self._source_pixmap:
            return None
        pw, ph = self._source_pixmap.width(), self._source_pixmap.height()
        ww, wh = self.width(), self.height()
        if not (pw and ph and ww and wh):
            return None
        scale = min(ww / pw, wh / ph)
        sw, sh = int(pw * scale), int(ph * scale)
        x = (ww - sw) // 2
        y = (wh - sh) // 2
        return QRect(x, y, sw, sh), scale

    def _to_image_coords(self, pos) -> tuple[int, int] | None:
        """Map a widget ``QPoint`` to image-space ``(ix, iy)``, or ``None``."""
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

    # ── Stroke painting ────────────────────────────────────────────────────────

    def _paint_stroke(self, pos):
        coords = self._to_image_coords(pos)
        info   = self._layout()
        if coords is None or info is None or self._mask_bw is None:
            return

        ix, iy = coords
        _, scale = info

        # Convert widget-space brush diameter → image-space radius.
        # _brush_size is the on-screen diameter; /scale converts to image pixels;
        # /2 gives radius.
        r        = max(3, int(self._brush_size / scale / 2))
        diameter = r * 2

        # Export mask: white strokes
        p1 = QPainter(self._mask_bw)
        p1.setBrush(QBrush(QColor(255, 255, 255)))
        p1.setPen(Qt.PenStyle.NoPen)
        p1.drawEllipse(ix - r, iy - r, diameter, diameter)
        p1.end()

        # Display overlay: semi-transparent red strokes
        p2 = QPainter(self._mask_color)
        p2.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_SourceOver
        )
        p2.setBrush(QBrush(QColor(230, 75, 55, 165)))
        p2.setPen(Qt.PenStyle.NoPen)
        p2.drawEllipse(ix - r, iy - r, diameter, diameter)
        p2.end()

    # ── Mouse events ───────────────────────────────────────────────────────────

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

    # ── Paint ──────────────────────────────────────────────────────────────────

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        if not self._source_pixmap:
            p.setPen(QColor(BORDER_HI))
            p.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "No image loaded"
            )
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
            r   = self._brush_size // 2
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

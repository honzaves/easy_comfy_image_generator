"""
comfy_generator.main_window
===========================
The application's main window.

Responsibilities
----------------
- Build and lay out all UI panels (left controls, right image view, inpaint view).
- Connect worker threads and relay their signals to UI updates.
- Delegate generation/inpainting logic entirely to the worker classes.
"""

import os
from pathlib import Path

from PyQt6.QtCore import Qt, QSize, QThread, QBuffer, QIODevice
from PyQt6.QtGui import QColor, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton,
    QButtonGroup, QRadioButton, QFrame, QScrollArea,
    QSizePolicy, QProgressBar, QGridLayout, QSplitter, QSlider,
    QFileDialog,
)

from .config import RESOLUTIONS, MODELS, T5_LIMIT, CLIP_LIMIT
from .styles import (
    BG, SURFACE, SURFACE2, BORDER, BORDER_HI,
    TEXT, TEXT_DIM, TEXT_BRIGHT,
    ACCENT, ACCENT2, OK, WARN, DANGER, IMPROVE_CLR,
)
from .widgets import TokenBar, ImageDisplay, MaskCanvas, estimate_tokens, token_color
from .workers import EnhanceWorker, GeneratorWorker, InpaintWorker


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ComfyUI Generator")
        self.resize(1100, 820)
        self.setMinimumSize(900, 660)

        # State
        self._model               = "flux-schnell"
        self._res_index           = 6
        self._count               = 1
        self._thread              = None
        self._worker              = None
        self._ethread             = None
        self._eworker             = None
        self._ithread             = None
        self._iworker             = None
        self._saved_path: str | None          = None
        self._status_msg          = ""
        self._thumbnails: list[tuple[bytes, str]] = []
        self._current_image_bytes: bytes | None   = None

        self._build_ui()

    # ── UI helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {TEXT_DIM}; font-size: 10px; font-weight: bold;"
            "margin-top: 4px; margin-bottom: 2px;"
        )
        return lbl

    def _set_busy(self, busy: bool, label: str = ""):
        """Disable / enable action buttons during any background operation."""
        self.generate_btn.setEnabled(not busy)
        self.enhance_btn.setEnabled(not busy)
        self.open_image_btn.setEnabled(not busy)
        self.improve_btn.setEnabled(not busy and self._current_image_bytes is not None)
        self.inpaint_generate_btn.setEnabled(not busy)
        self.progress_bar.setVisible(busy)
        if not busy:
            self.status_dot.setText("● READY")
            self.status_dot.setStyleSheet(
                f"color: {OK}; font-size: 10px; font-weight: bold;"
            )
        if label:
            self.elapsed_label.setText(label)
            self.elapsed_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px;")

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

    # ── Header ─────────────────────────────────────────────────────────────────

    def _make_header(self) -> QFrame:
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
        title.setStyleSheet(
            f"color: {TEXT_BRIGHT}; font-size: 20px; font-weight: 300;"
        )
        col.addWidget(sub)
        col.addWidget(title)

        self.status_dot = QLabel("● READY")
        self.status_dot.setStyleSheet(
            f"color: {OK}; font-size: 10px; font-weight: bold;"
        )

        h.addLayout(col)
        h.addStretch()
        h.addWidget(self.status_dot)
        return frame

    # ── Left panel ─────────────────────────────────────────────────────────────

    def _make_left(self) -> QWidget:
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
            "Describe the image… then hit ✦ Enhance to let Ollama rewrite it for Flux."
        )
        self.prompt_field.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.prompt_field.setMinimumHeight(100)
        self.prompt_field.setMaximumHeight(160)
        self.prompt_field.textChanged.connect(self._on_prompt_changed)
        v.addWidget(self.prompt_field)
        v.addSpacing(8)

        v.addLayout(self._make_action_buttons())
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

    def _make_action_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)

        self.generate_btn = QPushButton("⚡  Generate")
        self.generate_btn.setObjectName("generate_btn")
        self.generate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.generate_btn.clicked.connect(self._on_generate)

        self.enhance_btn = QPushButton("✦  Enhance")
        self.enhance_btn.setObjectName("enhance_btn")
        self.enhance_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.enhance_btn.setToolTip(
            "Send your description to Ollama and rewrite it as an optimised Flux prompt"
        )
        self.enhance_btn.clicked.connect(self._on_enhance)

        self.save_btn = QPushButton("📂")
        self.save_btn.setObjectName("save_btn")
        self.save_btn.setFixedWidth(44)
        self.save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_btn.setToolTip("Reveal in Finder")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._reveal_in_finder)

        row.addWidget(self.generate_btn, stretch=2)
        row.addWidget(self.enhance_btn,  stretch=2)
        row.addWidget(self.save_btn,     stretch=0)
        return row

    def _make_model_panel(self) -> QFrame:
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
            f"color: {TEXT_DIM}; font-size: 11px; font-style: italic;"
        )
        v.addWidget(self.model_desc)
        self.model_group.buttonClicked.connect(self._on_model_changed)
        return frame

    def _make_res_grid(self) -> QWidget:
        w    = QWidget()
        grid = QGridLayout(w)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(5)
        self.res_buttons: list[QPushButton] = []
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

    def _make_token_panel(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("panel")
        v = QVBoxLayout(frame)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(5)

        row = QHBoxLayout()
        self.token_used = QLabel("0")
        self.token_used.setStyleSheet(
            f"color: {OK}; font-size: 20px; font-weight: bold;"
        )
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

    def _make_count_selector(self) -> QWidget:
        w   = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(5)
        self.count_buttons: list[QPushButton] = []
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

    # ── Right panel ────────────────────────────────────────────────────────────

    def _make_right(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(20, 20, 20, 20)
        v.setSpacing(8)

        v.addWidget(self._make_normal_view(), stretch=1)
        v.addWidget(self._make_improve_view(), stretch=1)
        v.addWidget(self._make_thumb_strip())
        return w

    def _make_normal_view(self) -> QWidget:
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

        improve_row = QHBoxLayout()
        improve_row.setContentsMargins(0, 0, 0, 0)
        improve_row.setSpacing(6)

        self.open_image_btn = QPushButton("📁  Open Image…")
        self.open_image_btn.setObjectName("open_image_btn")
        self.open_image_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_image_btn.setToolTip("Load an image from disk and use the Improve tool on it")
        self.open_image_btn.clicked.connect(self._on_open_image)

        self.improve_btn = QPushButton("✏  Improve Image")
        self.improve_btn.setObjectName("improve_btn")
        self.improve_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.improve_btn.setEnabled(False)
        self.improve_btn.setToolTip(
            "Draw a mask over the region you want to change, then describe the improvement"
        )
        self.improve_btn.clicked.connect(self._on_improve_clicked)

        improve_row.addStretch()
        improve_row.addWidget(self.open_image_btn)
        improve_row.addWidget(self.improve_btn)
        nv.addLayout(improve_row)

        return self.normal_view

    def _make_improve_view(self) -> QWidget:
        self.improve_view = QWidget()
        self.improve_view.setVisible(False)
        iv = QVBoxLayout(self.improve_view)
        iv.setContentsMargins(0, 0, 0, 0)
        iv.setSpacing(8)

        self.mask_canvas = MaskCanvas()
        iv.addWidget(self.mask_canvas, stretch=1)

        instr = QLabel(
            "🖌  Paint over the region to improve  ·  use the slider to resize the brush"
        )
        instr.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px; padding: 2px 0;")
        instr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        iv.addWidget(instr)

        iv.addLayout(self._make_inpaint_controls())

        div = QFrame()
        div.setObjectName("divider")
        div.setFixedHeight(1)
        iv.addWidget(div)

        iv.addWidget(self._section_label("WHAT TO CHANGE IN THE SELECTED REGION"))

        self.inpaint_prompt = QTextEdit()
        self.inpaint_prompt.setObjectName("prompt_field")
        self.inpaint_prompt.setPlaceholderText(
            'Describe what should appear in the painted area…  '
            'e.g. "a bright red door with a brass knocker"'
        )
        self.inpaint_prompt.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.inpaint_prompt.setFixedHeight(72)
        iv.addWidget(self.inpaint_prompt)

        inpaint_btn_row = QHBoxLayout()
        self.inpaint_generate_btn = QPushButton("⚡  Generate Improvement")
        self.inpaint_generate_btn.setObjectName("generate_btn")
        self.inpaint_generate_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.inpaint_generate_btn.clicked.connect(self._on_inpaint_generate)
        inpaint_btn_row.addStretch()
        inpaint_btn_row.addWidget(self.inpaint_generate_btn)
        iv.addLayout(inpaint_btn_row)

        return self.improve_view

    def _make_inpaint_controls(self) -> QHBoxLayout:
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
        return ctrl_row

    def _make_thumb_strip(self) -> QScrollArea:
        self.thumb_scroll = QScrollArea()
        self.thumb_scroll.setWidgetResizable(True)
        self.thumb_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.thumb_scroll.setFixedHeight(90)
        self.thumb_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.thumb_scroll.setStyleSheet("background-color: transparent;")
        self.thumb_container = QWidget()
        self.thumb_layout = QHBoxLayout(self.thumb_container)
        self.thumb_layout.setContentsMargins(0, 0, 0, 0)
        self.thumb_layout.setSpacing(6)
        self.thumb_layout.addStretch()
        self.thumb_scroll.setWidget(self.thumb_container)
        self.thumb_scroll.setVisible(False)
        return self.thumb_scroll

    # ── Event handlers ─────────────────────────────────────────────────────────

    def _on_count_selected(self, n: int):
        self._count = n
        for i, btn in enumerate(self.count_buttons):
            btn.setProperty("selected", (i + 1) == n)
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _on_model_changed(self, btn):
        self._model = btn.property("model_key")
        self.model_desc.setText(MODELS[self._model]["desc"])

    def _on_res_selected(self, idx: int):
        self._res_index = idx
        for i, btn in enumerate(self.res_buttons):
            btn.setProperty("selected", i == idx)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        self._update_pixel_label()

    def _update_pixel_label(self):
        _, w, h, _ = RESOLUTIONS[self._res_index]
        self.pixel_label.setText(f"{w * h:,} pixels  ·  {w * h / 1_000_000:.1f} MP")

    def _on_prompt_changed(self):
        text = self.prompt_field.toPlainText()
        used = estimate_tokens(text)
        rem  = T5_LIMIT - used
        col  = token_color(used, T5_LIMIT)
        self.token_used.setText(str(used))
        self.token_used.setStyleSheet(
            f"color: {col}; font-size: 20px; font-weight: bold;"
        )
        self.token_remaining.setText(f"/ {T5_LIMIT}  ({max(0, rem)} remaining)")
        self.token_bar.set_used(used)
        self.token_warn.setText(
            f"⚠  exceeds CLIP-L limit ({CLIP_LIMIT} tokens) — T5 only"
            if used > CLIP_LIMIT else ""
        )

    # ── Enhance ────────────────────────────────────────────────────────────────

    def _on_enhance(self):
        description = self.prompt_field.toPlainText().strip()
        if not description:
            self.elapsed_label.setText("Enter a description first.")
            return

        self._set_busy(True)
        self.status_dot.setText("● ENHANCING")
        self.status_dot.setStyleSheet(
            f"color: {ACCENT2}; font-size: 10px; font-weight: bold;"
        )
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

    def _on_enhance_done(self, improved_prompt: str):
        self.prompt_field.setPlainText(improved_prompt)
        cursor = self.prompt_field.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.prompt_field.setTextCursor(cursor)
        self._set_busy(False, "✦ Prompt enhanced by Ollama")
        self.elapsed_label.setStyleSheet(f"color: {ACCENT2}; font-size: 10px;")

    def _on_enhance_error(self, msg: str):
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

        # Clear thumbnail strip
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

    def _on_worker_status(self, msg: str):
        self._status_msg = msg
        self.status_dot.setText(f"● {msg.upper()}")
        self.status_dot.setStyleSheet(
            f"color: {WARN}; font-size: 10px; font-weight: bold;"
        )
        self.elapsed_label.setText(f"{msg}…")

    def _on_worker_tick(self, secs: int, current: int, total: int):
        self.elapsed_label.setText(f"{self._status_msg}…  {secs}s")

    def _on_image_ready(self, image_bytes: bytes, saved_path: str, index: int, total: int):
        self._thumbnails.append((image_bytes, saved_path))
        self._saved_path          = saved_path
        self._current_image_bytes = image_bytes

        self.image_display.set_image(image_bytes)
        _, w, h, ratio = RESOLUTIONS[self._res_index]
        self.image_info.setText(
            f"{w}×{h}  ·  {ratio}  ·  {MODELS[self._model]['label']}  ·  "
            f"Image {index}/{total}  ·  {Path(saved_path).name}"
        )

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
            thumb_btn.setIconSize(QSize(118, 72))
            idx = index - 1
            thumb_btn.clicked.connect(lambda _, i=idx: self._select_thumbnail(i))
            self.thumb_layout.insertWidget(self.thumb_layout.count() - 1, thumb_btn)

        self.save_btn.setEnabled(True)
        self.improve_btn.setEnabled(True)

    def _check_all_done(self, image_bytes: bytes, saved_path: str, index: int, total: int):
        if index == total:
            self._thread.quit()
            self._set_busy(
                False,
                f"{total} image{'s' if total > 1 else ''} saved to generated_images/",
            )

    def _select_thumbnail(self, idx: int):
        if 0 <= idx < len(self._thumbnails):
            image_bytes, path         = self._thumbnails[idx]
            self._current_image_bytes = image_bytes
            self.image_display.set_image(image_bytes)
            self._saved_path = path
            _, w, h, ratio = RESOLUTIONS[self._res_index]
            self.image_info.setText(
                f"{w}×{h}  ·  {ratio}  ·  {MODELS[self._model]['label']}  ·  "
                f"Image {idx + 1}/{len(self._thumbnails)}  ·  {Path(path).name}"
            )
            for i in range(self.thumb_layout.count() - 1):
                item = self.thumb_layout.itemAt(i)
                if item and item.widget():
                    item.widget().setStyleSheet(
                        f"border: 2px solid {ACCENT}; border-radius: 4px; padding: 0;"
                        if i == idx
                        else f"border: 1px solid {BORDER}; border-radius: 4px; padding: 0;"
                    )

    def _on_error(self, msg: str):
        self.elapsed_label.setText(f"Error: {msg}")
        self.elapsed_label.setStyleSheet(f"color: {DANGER}; font-size: 10px;")
        self.status_dot.setText("● ERROR")
        self.status_dot.setStyleSheet(
            f"color: {DANGER}; font-size: 10px; font-weight: bold;"
        )
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
            return

        qimg = QImage(path)
        if qimg.isNull():
            self.elapsed_label.setText(f"Could not load image: {Path(path).name}")
            self.elapsed_label.setStyleSheet(f"color: {DANGER}; font-size: 10px;")
            return

        buf = QBuffer()
        buf.open(QIODevice.OpenModeFlag.WriteOnly)
        qimg.save(buf, "PNG")
        image_bytes = bytes(buf.data())

        self._current_image_bytes = image_bytes
        self._saved_path          = path
        self.image_display.set_image(image_bytes)
        self.image_info.setText(
            f"{qimg.width()}×{qimg.height()}  ·  {Path(path).name}  ·  loaded from disk"
        )
        self.improve_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.elapsed_label.setText(f"Loaded: {Path(path).name}")
        self.elapsed_label.setStyleSheet(f"color: {TEXT_DIM}; font-size: 10px;")
        self.status_dot.setText("● READY")
        self.status_dot.setStyleSheet(
            f"color: {OK}; font-size: 10px; font-weight: bold;"
        )

    # ── Improve / inpaint ──────────────────────────────────────────────────────

    def _on_improve_clicked(self):
        if not self._current_image_bytes:
            return
        self.mask_canvas.set_image(self._current_image_bytes)
        self.mask_canvas.set_brush_size(self.brush_slider.value())
        self.normal_view.setVisible(False)
        self.thumb_scroll.setVisible(False)
        self.improve_view.setVisible(True)

    def _on_back_from_improve(self):
        self.improve_view.setVisible(False)
        self.normal_view.setVisible(True)
        if len(self._thumbnails) > 1:
            self.thumb_scroll.setVisible(True)

    def _on_reset_mask(self):
        self.mask_canvas.reset_mask()

    def _on_inpaint_generate(self):
        """Validate inputs then kick off InpaintWorker."""
        prompt = self.inpaint_prompt.toPlainText().strip()
        if not prompt:
            self.elapsed_label.setText(
                "Describe what to change in the selected region first."
            )
            self.elapsed_label.setStyleSheet(f"color: {WARN}; font-size: 10px;")
            return

        mask_bytes = self.mask_canvas.get_mask_bytes()
        if not mask_bytes:
            self.elapsed_label.setText("No image loaded in the mask canvas.")
            return

        self._set_busy(True)
        self.status_dot.setText("● INPAINTING")
        self.status_dot.setStyleSheet(
            f"color: {IMPROVE_CLR}; font-size: 10px; font-weight: bold;"
        )
        self.elapsed_label.setText("Uploading to ComfyUI…")

        self._ithread = QThread()
        self._iworker = InpaintWorker(
            self._current_image_bytes, mask_bytes, prompt, self._model
        )
        self._iworker.moveToThread(self._ithread)
        self._ithread.started.connect(self._iworker.run)
        self._iworker.status.connect(self._on_worker_status)
        self._iworker.tick.connect(self._on_worker_tick)
        self._iworker.image_ready.connect(self._on_inpaint_ready)
        self._iworker.error.connect(self._on_error)
        self._iworker.error.connect(self._ithread.quit)
        self._ithread.start()

    def _on_inpaint_ready(self, image_bytes: bytes, saved_path: str):
        """Inpainting finished — return to normal view and display result."""
        self._ithread.quit()
        self._on_back_from_improve()

        self._current_image_bytes = image_bytes
        self._saved_path          = saved_path
        self._thumbnails.append((image_bytes, saved_path))

        self.image_display.set_image(image_bytes)
        self.image_info.setText(f"Improved  ·  {Path(saved_path).name}")
        self.save_btn.setEnabled(True)
        self._set_busy(False, "✏ Improvement saved to generated_images/")
        self.elapsed_label.setStyleSheet(f"color: {IMPROVE_CLR}; font-size: 10px;")

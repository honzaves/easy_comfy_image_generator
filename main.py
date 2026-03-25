#!/usr/bin/env python3
"""
ComfyUI Image Generator
=======================
Entry point.  Run with:

    python main.py
    # or, after installing with pip install -e .:
    comfy-generator
"""

import sys

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

from comfy_generator.main_window import MainWindow
from comfy_generator.styles import (
    STYLESHEET,
    BG, SURFACE2, TEXT, TEXT_BRIGHT, ACCENT,
)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(BG))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base,            QColor(SURFACE2))
    palette.setColor(QPalette.ColorRole.Text,            QColor(TEXT_BRIGHT))
    palette.setColor(QPalette.ColorRole.Button,          QColor(BG))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

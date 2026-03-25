"""
comfy_generator.styles
======================
Colour palette and the application-wide Qt stylesheet.
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

# ── Stylesheet ─────────────────────────────────────────────────────────────────

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

/* ── Improve / inpaint buttons ─────────────────────────────────────────────── */

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

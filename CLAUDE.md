# CLAUDE.md — Notes for AI-assisted development

This file documents conventions, known pitfalls, and context that should be
included at the start of any new Claude session working on this project.

---

## Project summary

**ComfyUI Generator** is a PyQt6 desktop application (macOS-primary) that:

- Sends text-to-image jobs to a local ComfyUI instance running Flux.
- Optionally enhances prompts via a local Ollama model.
- Supports inpainting: the user paints a mask, describes the change, and a new image is generated with only the masked region replaced.

---

## Module map

| File | Responsibility |
|---|---|
| `config.py` | All constants — URLs, model defs, resolutions, T5/CLIP limits, Ollama prompt |
| `styles.py` | Dark colour palette + full Qt stylesheet string |
| `workflows.py` | Pure functions returning ComfyUI workflow dicts (no Qt, no network) |
| `workers.py` | `QObject` subclasses run on `QThread`s — all network I/O lives here |
| `widgets.py` | `TokenBar`, `ImageDisplay`, `MaskCanvas` — custom Qt widgets |
| `main_window.py` | `MainWindow` — builds UI, connects signals, delegates to workers |
| `main.py` | Entry point — creates `QApplication`, sets palette, shows window |

---

## Key conventions

### Qt threading
All network calls (ComfyUI, Ollama) run in `QObject` workers moved to `QThread`s.
Workers emit Qt signals; `MainWindow` connects them to UI update slots.
**Never** call UI methods directly from a worker thread.

### Stylesheet
Colours are defined as constants in `styles.py` and interpolated into the f-string `STYLESHEET`.
To add a new button style: add a `QPushButton#new_name_btn` block in `STYLESHEET`
and call `btn.setObjectName("new_name_btn")` in the widget builder.

### Dynamic property styling
Count and resolution buttons use `setProperty("selected", bool)` + `unpolish/polish`
to trigger Qt stylesheet re-evaluation.  The same pattern must be used for any
new button group that needs a selected state.

### ComfyUI polling
`_poll_comfy_history()` in `workers.py` polls `/history/{prompt_id}` every 2 s
until the output appears or a 600 s deadline is hit.  The helper is shared by
both `GeneratorWorker` and `InpaintWorker`.

### Inpainting parameters
See docstring on `build_inpaint_workflow()` in `workflows.py`.
`denoise=1.0` and `steps=max(12, model_steps)` are intentional — do **not**
lower these without testing; lower values cause the model to barely change the
masked region.

---

## Known pitfalls

### Class / function declaration lines silently dropped by some editors
When asking an AI to edit this file, confirm that class and `def` lines are
preserved exactly.  A common failure mode is the opening line of a class or
method being omitted, causing an `IndentationError` or `NameError` at runtime.

### `QThread` lifecycle
Threads must be kept alive as instance attributes (`self._thread`, `self._ithread`,
`self._ethread`).  If they go out of scope the thread is garbage-collected while
still running, causing a crash.

### `QBuffer` / `QIODevice`
When converting a `QImage` to bytes, always `buf.open(...)` before `img.save(buf, ...)`,
and wrap with `bytes(buf.data())` — **not** `buf.data()` alone (returns `QByteArray`).

### macOS "Reveal in Finder"
`os.system(f'open -R "{path}"')` is macOS-specific.  Do not replace it with a
cross-platform call unless explicitly asked; it would need a platform check.

---

## Adding a new model

1. Add an entry to `MODELS` in `config.py`:
   ```python
   "my-model": {
       "unet":     "my-model.safetensors",
       "steps":    8,
       "guidance": 2.0,
       "label":    "My Model  ·  custom",
       "desc":     "8 steps  ·  custom quality",
   }
   ```
2. Place the UNET checkpoint in ComfyUI's `models/unet/` folder.
3. No other code changes needed.

---

## Adding a new resolution

Append a tuple to `RESOLUTIONS` in `config.py`:

```python
("2048×1152", 2048, 1152, "16:9 2K"),
```

The resolution grid in `MainWindow` is built dynamically from this list.

---

## Session start checklist

When beginning a new coding session on this project, share this file plus a
brief description of what you want to change.  No need to paste the full source.

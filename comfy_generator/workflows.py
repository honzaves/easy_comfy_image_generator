"""
comfy_generator.workflows
=========================
Pure functions that return ComfyUI workflow dicts.
No Qt, no network calls — trivially unit-testable.
"""

import random

from .config import MODELS


def build_workflow(
    prompt: str,
    model_key: str,
    width: int,
    height: int,
    *,
    hires: bool = False,
) -> dict:
    """Return a text-to-image workflow dict ready to POST to /prompt.

    Dispatches on the model's ``arch``:
      - ``flux``            → the split UNET/VAE/DualCLIP graph (``hires`` ignored;
                              Flux is tolerant up to large sizes already).
      - ``sd15`` / ``sdxl`` → an all-in-one CheckpointLoaderSimple graph.  When
                              ``hires`` is set, the first pass renders at the
                              model's native resolution and a LatentUpscale +
                              second low-denoise KSampler pass enlarges it to
                              ``width × height`` — clean large images without the
                              duplication artifacts of generating big directly.
    """
    if MODELS[model_key]["arch"] == "flux":
        return _build_flux_workflow(prompt, model_key, width, height)
    return _build_checkpoint_workflow(prompt, model_key, width, height, hires=hires)


def _build_flux_workflow(prompt: str, model_key: str, width: int, height: int) -> dict:
    """Return a text-to-image Flux workflow dict ready to POST to /prompt."""
    m    = MODELS[model_key]
    seed = random.randint(0, 2**32 - 1)
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": m["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "VAELoader",
              "inputs": {"vae_name": "ae.safetensors"}},
        "3": {"class_type": "DualCLIPLoader",
              "inputs": {"clip_name1": "clip_l.safetensors",
                         "clip_name2": "t5xxl_fp16.safetensors",
                         "type": "flux"}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["3", 0]}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "FluxGuidance",
              "inputs": {"guidance": m["guidance"], "conditioning": ["4", 0]}},
        "7": {"class_type": "KSampler",
              "inputs": {
                  "model": ["1", 0], "positive": ["6", 0], "negative": ["4", 0],
                  "latent_image": ["5", 0], "seed": seed, "steps": m["steps"],
                  "cfg": 1.0, "sampler_name": "euler",
                  "scheduler": "simple", "denoise": 1.0,
              }},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["7", 0], "vae": ["2", 0]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "comfy_gen", "images": ["8", 0]}},
    }


def _build_checkpoint_workflow(
    prompt: str,
    model_key: str,
    width: int,
    height: int,
    *,
    hires: bool,
) -> dict:
    """Return a text-to-image workflow for an all-in-one SD/SDXL checkpoint.

    Uses ``CheckpointLoaderSimple`` (outputs MODEL/CLIP/VAE from one file) and a
    *separate* empty/negative ``CLIPTextEncode`` node — unlike Flux, these models
    run at cfg > 1, so the negative conditioning is active and must NOT reuse the
    positive prompt.
    """
    m    = MODELS[model_key]
    seed = random.randint(0, 2**32 - 1)
    nw, nh = m["native"]

    # With hi-res fix, render the first pass at native size for correct
    # composition, then upscale.  Otherwise render straight at the target size.
    upscale = hires and (width, height) != (nw, nh)
    first_w, first_h = (nw, nh) if upscale else (width, height)

    wf: dict = {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": m["ckpt"]}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": m.get("negative", ""), "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": first_w, "height": first_h, "batch_size": 1}},
        "5": {"class_type": "KSampler",
              "inputs": {
                  "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                  "latent_image": ["4", 0], "seed": seed, "steps": m["steps"],
                  "cfg": m["cfg"], "sampler_name": m["sampler"],
                  "scheduler": m["scheduler"], "denoise": 1.0,
              }},
    }

    latent = ["5", 0]
    if upscale:
        wf["6"] = {"class_type": "LatentUpscale",
                   "inputs": {"samples": ["5", 0], "upscale_method": "bilinear",
                              "width": width, "height": height, "crop": "disabled"}}
        # Second pass: low denoise so composition is preserved but detail is
        # added at the higher resolution.
        wf["7"] = {"class_type": "KSampler",
                   "inputs": {
                       "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                       "latent_image": ["6", 0], "seed": seed, "steps": m["steps"],
                       "cfg": m["cfg"], "sampler_name": m["sampler"],
                       "scheduler": m["scheduler"], "denoise": 0.5,
                   }}
        latent = ["7", 0]

    wf["8"] = {"class_type": "VAEDecode",
               "inputs": {"samples": latent, "vae": ["1", 2]}}
    wf["9"] = {"class_type": "SaveImage",
               "inputs": {"filename_prefix": "comfy_gen", "images": ["8", 0]}}
    return wf


def build_inpaint_workflow(
    prompt: str,
    model_key: str,
    image_name: str,
    mask_name: str,
) -> dict:
    """Return an inpainting workflow dict, dispatched on the model's arch."""
    if MODELS[model_key]["arch"] == "flux":
        return _build_flux_inpaint_workflow(prompt, model_key, image_name, mask_name)
    return _build_checkpoint_inpaint_workflow(prompt, model_key, image_name, mask_name)


def _build_flux_inpaint_workflow(
    prompt: str,
    model_key: str,
    image_name: str,
    mask_name: str,
) -> dict:
    """Return a Flux inpainting workflow dict using SetLatentNoiseMask.

    Key choices
    -----------
    - ``denoise=1.0``  — mask=1.0 pixels get 100% noise → full regeneration in
                         the painted area; mask=0.0 pixels stay untouched.
                         Any value < 1.0 leaves residual original signal and
                         the model barely changes the masked region.
    - ``steps=max(12)``— Schnell's 4 steps is far too few for inpainting;
                         12 gives clean results without being slow.
    - ``guidance=3.5`` — Schnell's guidance=1.0 is tuned for full-image speed;
                         inpainting needs stronger conditioning to follow the
                         prompt within a small masked region.
    """
    m = MODELS[model_key]
    seed             = random.randint(0, 2**32 - 1)
    inpaint_steps    = max(12, m["steps"])
    inpaint_guidance = max(3.5, m["guidance"])

    return {
        "1":  {"class_type": "UNETLoader",
               "inputs": {"unet_name": m["unet"], "weight_dtype": "default"}},
        "2":  {"class_type": "VAELoader",
               "inputs": {"vae_name": "ae.safetensors"}},
        "3":  {"class_type": "DualCLIPLoader",
               "inputs": {"clip_name1": "clip_l.safetensors",
                          "clip_name2": "t5xxl_fp16.safetensors",
                          "type": "flux"}},
        "4":  {"class_type": "CLIPTextEncode",
               "inputs": {"text": prompt, "clip": ["3", 0]}},
        "5":  {"class_type": "FluxGuidance",
               "inputs": {"guidance": inpaint_guidance, "conditioning": ["4", 0]}},
        # Load source image and mask
        "10": {"class_type": "LoadImage",
               "inputs": {"image": image_name}},
        "11": {"class_type": "LoadImageMask",
               "inputs": {"image": mask_name, "channel": "red"}},
        # Encode source image, attach noise mask
        "12": {"class_type": "VAEEncode",
               "inputs": {"pixels": ["10", 0], "vae": ["2", 0]}},
        "13": {"class_type": "SetLatentNoiseMask",
               "inputs": {"samples": ["12", 0], "mask": ["11", 0]}},
        # denoise=1.0: masked pixels start from pure noise, unmasked stay put
        "7":  {"class_type": "KSampler",
               "inputs": {
                   "model": ["1", 0], "positive": ["5", 0], "negative": ["4", 0],
                   "latent_image": ["13", 0], "seed": seed, "steps": inpaint_steps,
                   "cfg": 1.0, "sampler_name": "euler",
                   "scheduler": "simple", "denoise": 1.0,
               }},
        "8":  {"class_type": "VAEDecode",
               "inputs": {"samples": ["7", 0], "vae": ["2", 0]}},
        "9":  {"class_type": "SaveImage",
               "inputs": {"filename_prefix": "comfy_inpaint", "images": ["8", 0]}},
    }


def _build_checkpoint_inpaint_workflow(
    prompt: str,
    model_key: str,
    image_name: str,
    mask_name: str,
) -> dict:
    """Return an SD/SDXL inpainting workflow using SetLatentNoiseMask.

    Same masked-noise approach as the Flux variant, but built around
    ``CheckpointLoaderSimple`` with a separate active negative prompt and the
    model's own cfg/sampler (Flux's cfg=1.0 / euler would ignore the negative
    and under-condition these models).  ``denoise=1.0`` so the masked region is
    fully regenerated; the model's configured step count is sufficient.
    """
    m    = MODELS[model_key]
    seed = random.randint(0, 2**32 - 1)
    return {
        "1":  {"class_type": "CheckpointLoaderSimple",
               "inputs": {"ckpt_name": m["ckpt"]}},
        "2":  {"class_type": "CLIPTextEncode",
               "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3":  {"class_type": "CLIPTextEncode",
               "inputs": {"text": m.get("negative", ""), "clip": ["1", 1]}},
        "10": {"class_type": "LoadImage",
               "inputs": {"image": image_name}},
        "11": {"class_type": "LoadImageMask",
               "inputs": {"image": mask_name, "channel": "red"}},
        "12": {"class_type": "VAEEncode",
               "inputs": {"pixels": ["10", 0], "vae": ["1", 2]}},
        "13": {"class_type": "SetLatentNoiseMask",
               "inputs": {"samples": ["12", 0], "mask": ["11", 0]}},
        "7":  {"class_type": "KSampler",
               "inputs": {
                   "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                   "latent_image": ["13", 0], "seed": seed, "steps": m["steps"],
                   "cfg": m["cfg"], "sampler_name": m["sampler"],
                   "scheduler": m["scheduler"], "denoise": 1.0,
               }},
        "8":  {"class_type": "VAEDecode",
               "inputs": {"samples": ["7", 0], "vae": ["1", 2]}},
        "9":  {"class_type": "SaveImage",
               "inputs": {"filename_prefix": "comfy_inpaint", "images": ["8", 0]}},
    }

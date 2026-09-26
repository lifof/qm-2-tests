"""Image generation backends for Qwen-Image.

All backends implement `generate(prompt, negative, width, height, seed, references) -> PIL.Image`.

- comfyui   : a ComfyUI server (auto-started locally) running Qwen-Image 2.1 from split
              checkpoint files (.gguf / .safetensors). See comfyui.py.
- diffusers : run the model locally from a Hugging Face repo id or a local folder.
- openai    : any server exposing an OpenAI-style /v1/images/generations endpoint
              (vLLM-Omni, LocalAI, a custom FastAPI wrapper, ...).
- dashscope : Alibaba Cloud Model Studio (hosted Qwen-Image models).
- mock      : draws placeholder panels; for testing the pipeline without a GPU.
"""

from __future__ import annotations

import base64
import inspect
import io
import os
import textwrap
from pathlib import Path
from typing import List, Optional, Sequence

from PIL import Image, ImageDraw

from .models import ImageSettings

# Aspect ratio (width / height) per camera shot. Panels are generated at about
# ImageSettings.megapixels and shrunk to the strip width (800px) afterwards, so ~1 MP
# (Qwen-Image 2.1's native default) is plenty; raise it for print-quality art.
SHOT_ASPECTS = {
    "establishing": 16 / 9,
    "wide": 3 / 2,
    "medium": 1.0,
    "close-up": 1.0,
    "extreme-close-up": 16 / 9,
    "over-the-shoulder": 4 / 5,
    "action": 2 / 3,
}
PORTRAIT_ASPECT = 4 / 5


def size_for_aspect(aspect: float, megapixels: float = 1.0) -> tuple[int, int]:
    """Width/height near the requested pixel count, both multiples of 32."""
    pixels = max(0.25, megapixels) * 1024 * 1024
    width = round((pixels * aspect) ** 0.5 / 32) * 32
    height = round((pixels / aspect) ** 0.5 / 32) * 32
    return max(256, width), max(256, height)


def portrait_size(megapixels: float = 1.0) -> tuple[int, int]:
    return size_for_aspect(PORTRAIT_ASPECT, megapixels)


class ImageBackend:
    supports_references = False

    def generate(
        self,
        prompt: str,
        negative: str,
        width: int,
        height: int,
        seed: int,
        references: Sequence[Path] = (),
    ) -> Image.Image:
        raise NotImplementedError


def _to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------


class DiffusersBackend(ImageBackend):
    def __init__(self, settings: ImageSettings):
        import torch
        from diffusers import DiffusionPipeline

        self.torch = torch
        self.settings = settings
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = Path(settings.model).expanduser()
        if model.is_file() or (model.is_dir() and not (model / "model_index.json").exists()):
            raise ValueError(
                f"{settings.model} is not a diffusers model folder (no model_index.json). Single-file checkpoints "
                "(.gguf / .safetensors, e.g. Qwen-Image 2.1) need the 'comfyui' backend."
            )
        import diffusers.utils as dutils

        # newer diffusers renamed torch_dtype -> dtype (and warn on the old name)
        new_api = hasattr(getattr(dutils, "deprecation_utils", None), "_resolve_dtype")
        dtype_kwarg = {"dtype": dtype} if new_api else {"torch_dtype": dtype}
        self.pipe = DiffusionPipeline.from_pretrained(settings.model, **dtype_kwarg)
        if device == "cuda" and os.environ.get("WEBTOON_CPU_OFFLOAD"):
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)
        self.device = device
        self.params = set(inspect.signature(self.pipe.__call__).parameters)
        # Edit-capable Qwen pipelines take `image=`; plain text-to-image ones don't.
        self.supports_references = "image" in self.params

    def generate(self, prompt, negative, width, height, seed, references=()):
        kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": self.settings.steps,
            "generator": self.torch.Generator(device="cpu").manual_seed(seed),
        }
        if "negative_prompt" in self.params:
            kwargs["negative_prompt"] = negative
        if "true_cfg_scale" in self.params:  # Qwen-Image pipelines
            kwargs["true_cfg_scale"] = self.settings.cfg
        elif "guidance_scale" in self.params:
            kwargs["guidance_scale"] = self.settings.cfg
        if references and self.supports_references:
            images = [Image.open(p).convert("RGB") for p in references]
            kwargs["image"] = images if len(images) > 1 else images[0]
        return self.pipe(**kwargs).images[0]


# ---------------------------------------------------------------------------


class OpenAICompatibleBackend(ImageBackend):
    """POST {base_url}/images/generations, and /images/edits when references are given."""

    supports_references = True

    def __init__(self, settings: ImageSettings):
        import requests

        if not settings.base_url:
            raise ValueError("The 'openai' image backend needs --image-base-url, e.g. http://localhost:8000/v1")
        self.requests = requests
        self.settings = settings
        self.base = settings.base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {os.environ.get('IMAGE_API_KEY', 'not-needed')}"}

    def _decode(self, payload: dict) -> Image.Image:
        item = payload["data"][0]
        if item.get("b64_json"):
            return Image.open(io.BytesIO(base64.b64decode(item["b64_json"]))).convert("RGB")
        resp = self.requests.get(item["url"], timeout=300)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")

    def generate(self, prompt, negative, width, height, seed, references=()):
        extra = {
            "negative_prompt": negative,
            "seed": seed,
            "num_inference_steps": self.settings.steps,
            "true_cfg_scale": self.settings.cfg,
        }
        common = {"model": self.settings.model, "prompt": prompt, "size": f"{width}x{height}", "n": 1,
                  "response_format": "b64_json"}
        if references and self.settings.use_references:
            files = [("image[]", (Path(p).name, Path(p).read_bytes(), "image/png")) for p in references]
            data = {**common, **{k: str(v) for k, v in extra.items()}}
            resp = self.requests.post(f"{self.base}/images/edits", headers=self.headers, data=data, files=files,
                                      timeout=1800)
        else:
            resp = self.requests.post(f"{self.base}/images/generations", headers=self.headers,
                                      json={**common, **extra}, timeout=1800)
        resp.raise_for_status()
        return self._decode(resp.json())


# ---------------------------------------------------------------------------


class DashScopeBackend(ImageBackend):
    """Alibaba Cloud Model Studio multimodal-generation endpoint (DASHSCOPE_API_KEY)."""

    supports_references = True
    DEFAULT_URL = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    def __init__(self, settings: ImageSettings):
        import requests

        self.requests = requests
        self.settings = settings
        self.url = settings.base_url or self.DEFAULT_URL
        key = os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            raise ValueError("Set DASHSCOPE_API_KEY to use the dashscope backend")
        self.headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def generate(self, prompt, negative, width, height, seed, references=()):
        content: List[dict] = []
        if references and self.settings.use_references:
            for p in references:
                content.append({"image": "data:image/png;base64," + base64.b64encode(Path(p).read_bytes()).decode()})
        content.append({"text": prompt})
        body = {
            "model": self.settings.model,
            "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": {
                "negative_prompt": negative,
                "size": f"{width}*{height}",
                "seed": seed % 2147483647,
                "prompt_extend": False,
                "watermark": False,
            },
        }
        resp = self.requests.post(self.url, headers=self.headers, json=body, timeout=1800)
        if resp.status_code >= 400:
            raise RuntimeError(f"DashScope error {resp.status_code}: {resp.text[:1000]}")
        url = resp.json()["output"]["choices"][0]["message"]["content"][0]["image"]
        img = self.requests.get(url, timeout=300)
        img.raise_for_status()
        return Image.open(io.BytesIO(img.content)).convert("RGB")


# ---------------------------------------------------------------------------


class MockBackend(ImageBackend):
    supports_references = True

    def generate(self, prompt, negative, width, height, seed, references=()):
        rng = seed & 0xFFFFFF
        color = ((rng >> 16) % 156 + 60, (rng >> 8 & 0xFF) % 156 + 60, (rng & 0xFF) % 156 + 60)
        img = Image.new("RGB", (width, height), color)
        draw = ImageDraw.Draw(img)
        body = textwrap.fill(prompt, width=max(20, width // 14))
        draw.multiline_text((24, 24), f"[mock seed={seed} refs={len(references)}]\n{body}", fill=(0, 0, 0))
        return img


def _comfyui_backend(settings: ImageSettings) -> ImageBackend:
    from .comfyui import ComfyUIBackend

    return ComfyUIBackend(settings)


def make_backend(settings: ImageSettings) -> ImageBackend:
    backends = {
        "comfyui": _comfyui_backend,
        "diffusers": DiffusersBackend,
        "openai": OpenAICompatibleBackend,
        "dashscope": DashScopeBackend,
        "mock": lambda s: MockBackend(),
    }
    if settings.backend not in backends:
        raise ValueError(f"Unknown image backend {settings.backend!r}; choose from {sorted(backends)}")
    return backends[settings.backend](settings)


def size_for_shot(shot: Optional[str], megapixels: float = 1.0) -> tuple[int, int]:
    return size_for_aspect(SHOT_ASPECTS.get(shot or "medium", 1.0), megapixels)

"""ComfyUI backend: runs Qwen-Image 2.1 from split checkpoint files.

Qwen-Image 2.1 ships as three files (diffusion model, Qwen3-VL-8B text encoder, VAE),
often with the diffusion model as a .gguf. diffusers can't load these, but ComfyUI
supports the model natively. This backend:

1. starts ComfyUI if it isn't running (when you give the install folder),
2. links your model files into ComfyUI's models/ folders,
3. sends a Qwen-Image 2.1 text-to-image graph per panel (the same nodes and defaults as
   ComfyUI's official template), passing character sheets as reference images, and
4. downloads the result.

Loading a .gguf diffusion model needs the ComfyUI-GGUF custom node.
"""

from __future__ import annotations

import atexit
import io
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image

from .image_backends import ImageBackend
from .models import ImageSettings

DEFAULT_URL = "http://127.0.0.1:8188"
GGUF_REPO = "https://github.com/city96/ComfyUI-GGUF"
MODEL_SUBDIRS = {"diffusion_model": "diffusion_models", "text_encoder": "text_encoders", "vae": "vae"}
MAX_REFERENCES = 16  # TextEncodeQwenImage21 accepts images.image_1 .. image_16


class ComfyUIError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Model files
# ---------------------------------------------------------------------------


def detect_model_files(folder: Path) -> Dict[str, Optional[Path]]:
    """Guess which file in a folder is the diffusion model, the text encoder and the VAE."""
    files = sorted(p for p in Path(folder).expanduser().iterdir()
                   if p.is_file() and p.suffix.lower() in (".gguf", ".safetensors", ".sft", ".ckpt", ".pt"))
    found: Dict[str, Optional[Path]] = {"diffusion_model": None, "text_encoder": None, "vae": None}
    for f in files:
        name = f.name.lower()
        if "vae" in name:
            found["vae"] = found["vae"] or f
        elif re.search(r"qwen[-_]?[23](\.5)?[-_]?vl|text[-_]?enc|clip|umt5|t5xxl|qwen[-_]?[23](\.5)?[-_]?\d+b", name):
            found["text_encoder"] = found["text_encoder"] or f
    for f in files:
        if f not in found.values():
            found["diffusion_model"] = found["diffusion_model"] or f
    return found


def link_into_comfy(comfy_dir: Path, kind: str, source: Path) -> str:
    """Symlink a model file into ComfyUI/models/<subdir>/ and return the name ComfyUI knows it by."""
    target_dir = Path(comfy_dir).expanduser() / "models" / MODEL_SUBDIRS[kind]
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / source.name
    if target.exists() or target.is_symlink():
        if target.resolve() != source.resolve():
            if target.is_symlink():
                target.unlink()
            else:
                return source.name  # a real file with that name is already there: use it
        else:
            return source.name
    try:
        target.symlink_to(source.resolve())
    except OSError as exc:  # e.g. Windows without symlink rights
        raise ComfyUIError(f"Could not link {source} into {target_dir} ({exc}). Copy the file there by hand.") from exc
    return source.name


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


def build_workflow(names: Dict[str, str], prompt: str, negative: str, width: int, height: int, seed: int,
                   settings: ImageSettings, reference_names: Sequence[str] = ()) -> dict:
    """API-format graph mirroring ComfyUI's official Qwen-Image 2.1 template."""
    unet = names["diffusion_model"]
    text_encoder = names["text_encoder"]
    wf: dict = {
        "1": ({"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet}} if unet.lower().endswith(".gguf")
              else {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}}),
        "2": ({"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": text_encoder, "type": "qwen_image"}}
              if text_encoder.lower().endswith(".gguf")
              else {"class_type": "CLIPLoader", "inputs": {"clip_name": text_encoder, "type": "qwen_image",
                                                           "device": "default"}}),
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": names["vae"]}},
        "4": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "clip": ["2", 0], "prompt": prompt, "negative_prompt": negative, "vae": ["3", 0], "resolution": 1024}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "seed": seed, "steps": settings.steps, "cfg": settings.cfg,
            "sampler_name": settings.sampler, "scheduler": settings.scheduler,
            "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["5", 0], "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "PreviewImage", "inputs": {"images": ["7", 0]}},
    }
    for i, ref in enumerate(reference_names[:MAX_REFERENCES], start=1):
        node = str(100 + i)
        wf[node] = {"class_type": "LoadImage", "inputs": {"image": ref}}
        wf["4"]["inputs"][f"images.image_{i}"] = [node, 0]
    return wf


def fill_custom_workflow(template: dict, values: Dict[str, object], reference_names: Sequence[str]) -> dict:
    """Fill a user workflow (ComfyUI "Export (API)") that uses {{placeholders}} in its inputs.

    Placeholders: {{prompt}} {{negative}} {{seed}} {{width}} {{height}} {{steps}} {{cfg}}
    and {{ref_1}} .. {{ref_16}} for LoadImage file names. A string that is exactly one
    placeholder gets the typed value; unused {{ref_N}} LoadImage nodes are removed.
    """
    refs = {f"ref_{i}": name for i, name in enumerate(reference_names, start=1)}
    wf = json.loads(json.dumps(template))
    unused_ref_nodes = []
    for node_id, node in wf.items():
        for key, value in list(node.get("inputs", {}).items()):
            if not isinstance(value, str):
                continue
            m = re.fullmatch(r"\{\{(\w+)\}\}", value.strip())
            if m and m.group(1) in values:
                node["inputs"][key] = values[m.group(1)]
            elif m and m.group(1).startswith("ref_"):
                if m.group(1) in refs:
                    node["inputs"][key] = refs[m.group(1)]
                else:
                    unused_ref_nodes.append(node_id)
            else:
                node["inputs"][key] = re.sub(r"\{\{(\w+)\}\}", lambda mm: str(values.get(mm.group(1), mm.group(0))), value)
    for node_id in unused_ref_nodes:  # drop missing references and every input that pointed at them
        wf.pop(node_id, None)
        for node in wf.values():
            for key, value in list(node.get("inputs", {}).items()):
                if isinstance(value, list) and value and value[0] == node_id:
                    del node["inputs"][key]
    return wf


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class ComfyUIBackend(ImageBackend):
    supports_references = True

    def __init__(self, settings: ImageSettings, log=print):
        import requests

        self.requests = requests
        self.settings = settings
        self.log = log
        self.url = (settings.base_url or DEFAULT_URL).rstrip("/")
        self.client_id = uuid.uuid4().hex
        self.process: Optional[subprocess.Popen] = None
        self.template = None
        if settings.workflow_file:
            self.template = json.loads(Path(settings.workflow_file).expanduser().read_text(encoding="utf-8"))
            if "nodes" in self.template:
                raise ComfyUIError("That workflow is in UI format. In ComfyUI use Workflow > Export (API).")
        self.ensure_server()
        self.names = self.resolve_models()

    # -- server --------------------------------------------------------------
    def alive(self) -> bool:
        try:
            return self.requests.get(f"{self.url}/system_stats", timeout=3).ok
        except self.requests.RequestException:
            return False

    def ensure_server(self) -> None:
        if self.alive():
            return
        comfy_dir = Path(self.settings.comfy_dir).expanduser() if self.settings.comfy_dir else None
        if comfy_dir and not (comfy_dir / "main.py").exists():
            raise ComfyUIError(
                f"ComfyUI is not reachable at {self.url}. {comfy_dir} looks like a ComfyUI Desktop folder: open the "
                "ComfyUI app first, and check that the address matches the one in its settings "
                "(ComfyUI Desktop usually serves on http://127.0.0.1:8000).")
        if not (self.settings.auto_start and comfy_dir):
            raise ComfyUIError(
                f"ComfyUI is not reachable at {self.url}. Start it (python main.py in your ComfyUI folder), "
                "or set the ComfyUI folder in the settings so it can be started automatically.")
        python = comfy_python(comfy_dir)
        port = re.search(r":(\d+)$", self.url)
        log_path = Path.home() / ".cache" / "webtoon" / "comfyui.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log(f"Starting ComfyUI ({python} main.py), log: {log_path}")
        log_file = open(log_path, "ab")
        cmd = [python, "main.py", "--listen", "127.0.0.1", "--port", port.group(1) if port else "8188"]
        env = dict(os.environ)
        if sys.platform == "darwin":  # Apple Silicon: run ops MPS lacks on the CPU instead of crashing
            env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self.process = subprocess.Popen(cmd + shlex.split(self.settings.comfy_args or ""),
                                        cwd=comfy_dir, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        atexit.register(self.shutdown)
        deadline = time.time() + 600
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise ComfyUIError(f"ComfyUI exited during startup (code {self.process.returncode}):\n"
                                   f"{_tail(log_path)}\n(full log: {log_path})")
            if self.alive():
                self.log("ComfyUI is up.")
                return
            time.sleep(2)
        raise ComfyUIError(f"ComfyUI did not start within 10 minutes; see {log_path}")

    def shutdown(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()

    # -- models --------------------------------------------------------------
    def node_options(self, node_class: str, input_name: str) -> Optional[List[str]]:
        """Choices ComfyUI offers for a combo input, or None if the node isn't installed."""
        resp = self.requests.get(f"{self.url}/object_info/{node_class}", timeout=30)
        info = resp.json().get(node_class) if resp.ok else None
        if not info:
            return None
        spec = {**info["input"].get("required", {}), **info["input"].get("optional", {})}.get(input_name)
        if not spec:
            return []
        if isinstance(spec[0], list):  # classic combo: [[choices...], {...}]
            return spec[0]
        return (spec[1] or {}).get("options", []) if len(spec) > 1 else []  # ["COMBO", {"options": [...]}]

    def resolve_models(self) -> Dict[str, str]:
        s = self.settings
        missing = [k for k in MODEL_SUBDIRS if not getattr(s, k)]
        if self.template is None and missing:
            raise ComfyUIError(f"Set the model files first (missing: {', '.join(missing)}).")
        names: Dict[str, str] = {}
        for kind in MODEL_SUBDIRS:
            value = getattr(s, kind)
            if not value:
                continue
            path = Path(value).expanduser()
            if s.comfy_dir and path.is_file():
                names[kind] = link_into_comfy(Path(s.comfy_dir), kind, path)
            else:
                names[kind] = path.name
        if s.workflow_file:
            return names

        checks = [
            ("TextEncodeQwenImage21", "clip", None,
             "This ComfyUI has no Qwen-Image 2.1 support. Update ComfyUI (git pull; pip install -r requirements.txt)."),
        ]
        unet = names["diffusion_model"]
        if unet.lower().endswith(".gguf"):
            checks.append(("UnetLoaderGGUF", "unet_name", "diffusion_model", _gguf_help("diffusion model")))
        else:
            checks.append(("UNETLoader", "unet_name", "diffusion_model", None))
        if names["text_encoder"].lower().endswith(".gguf"):
            checks.append(("CLIPLoaderGGUF", "clip_name", "text_encoder", _gguf_help("text encoder")))
        else:
            checks.append(("CLIPLoader", "clip_name", "text_encoder", None))
        checks.append(("VAELoader", "vae_name", "vae", None))

        for node, field, kind, not_installed in checks:
            options = self.node_options(node, field)
            if options is None:
                raise ComfyUIError(not_installed or f"ComfyUI has no {node} node.")
            if kind is None:
                continue
            wanted = names[kind]
            if wanted in options:
                continue
            match = next((o for o in options if o.replace("\\", "/").split("/")[-1] == wanted), None)
            if match is None:
                folder = MODEL_SUBDIRS[kind]
                hint = ("Set the ComfyUI folder in the settings so the file can be linked automatically, or put it in "
                        f"ComfyUI/models/{folder}/." if not s.comfy_dir else
                        f"It was linked into {Path(s.comfy_dir).expanduser() / 'models' / folder}; is that the same "
                        f"ComfyUI that runs at {self.url}?")
                raise ComfyUIError(f"ComfyUI can't see the {kind.replace('_', ' ')} '{wanted}'. {hint}")
            names[kind] = match
        return names

    # -- generation ----------------------------------------------------------
    def upload(self, path: Path) -> str:
        with open(path, "rb") as fh:
            resp = self.requests.post(f"{self.url}/upload/image", files={"image": (path.name, fh, "image/png")},
                                      data={"overwrite": "true", "subfolder": "webtoon"}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return f"{data['subfolder']}/{data['name']}" if data.get("subfolder") else data["name"]

    def generate(self, prompt, negative, width, height, seed, references=()):
        width, height = _round32(width), _round32(height)
        refs = [self.upload(Path(p)) for p in references] if self.settings.use_references else []
        if self.template is not None:
            values = {"prompt": prompt, "negative": negative, "seed": seed, "width": width, "height": height,
                      "steps": self.settings.steps, "cfg": self.settings.cfg}
            workflow = fill_custom_workflow(self.template, values, refs)
        else:
            workflow = build_workflow(self.names, prompt, negative, width, height, seed, self.settings, refs)
        return self.run(workflow)

    def run(self, workflow: dict) -> Image.Image:
        resp = self.requests.post(f"{self.url}/prompt", json={"prompt": workflow, "client_id": self.client_id},
                                  timeout=120)
        if resp.status_code >= 400:
            raise ComfyUIError(_format_validation_error(resp))
        prompt_id = resp.json()["prompt_id"]
        while True:
            if self.process is not None and self.process.poll() is not None:
                log_path = Path.home() / ".cache" / "webtoon" / "comfyui.log"
                raise ComfyUIError(f"ComfyUI stopped while generating (out of memory?):\n{_tail(log_path)}")
            hist = self.requests.get(f"{self.url}/history/{prompt_id}", timeout=30)
            entry = hist.json().get(prompt_id) if hist.ok else None
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise ComfyUIError(_format_execution_error(status))
                if status.get("completed", True):
                    break
            time.sleep(1)
        for output in entry.get("outputs", {}).values():
            for img in output.get("images", []):
                view = self.requests.get(f"{self.url}/view", params={
                    "filename": img["filename"], "subfolder": img.get("subfolder", ""), "type": img.get("type", "output")},
                    timeout=120)
                view.raise_for_status()
                return Image.open(io.BytesIO(view.content)).convert("RGB")
        raise ComfyUIError("The workflow finished without producing an image (does it end in a Save/Preview Image node?)")


# ---------------------------------------------------------------------------


def system_memory_gb() -> Optional[float]:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3
    except (ValueError, OSError, AttributeError):
        pass
    try:  # macOS fallback
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)) / 1024 ** 3
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None


def memory_report(files: Dict[str, Optional[Path]], ram_gb: Optional[float] = None,
                  unified: Optional[bool] = None) -> tuple[List[str], List[str]]:
    """(info lines, warnings) about whether the model files fit in memory.

    On Apple Silicon the GPU shares system memory but may only use about 75% of it by
    default, and ComfyUI needs the text encoder and the diffusion model loaded together
    to avoid reloading one of them for every panel.
    """
    ram_gb = system_memory_gb() if ram_gb is None else ram_gb
    unified = (sys.platform == "darwin") if unified is None else unified
    sizes = {k: (p.stat().st_size / 1024 ** 3 if p and p.exists() else 0.0) for k, p in files.items()}
    info = [f"{k.replace('_', ' ')}: {v:.1f} GB" for k, v in sizes.items()]
    warnings: List[str] = []
    if not (unified and ram_gb):
        return info, warnings
    budget = ram_gb * 0.75
    dit, te = sizes.get("diffusion_model", 0.0), sizes.get("text_encoder", 0.0)
    together = dit + te + sizes.get("vae", 0.0) + 3  # + ~3 GB for activations at ~1 MP
    info.append(f"this Mac: {ram_gb:.0f} GB unified memory, of which the GPU can use about {budget:.0f} GB")
    if dit + 3 > budget:
        warnings.append(f"The diffusion model alone ({dit:.1f} GB) is more than the GPU can use: generation will be "
                        "extremely slow or fail. Use a quantized GGUF (Q8_0 is about half the size of BF16).")
    elif together > budget:
        warnings.append(
            f"Text encoder + diffusion model need about {together:.0f} GB, more than the ~{budget:.0f} GB the GPU can "
            "use, so ComfyUI will reload one of them for every panel (slow). Fixes: a Q8_0 GGUF of the diffusion model "
            "(about half of BF16, near-identical quality) and/or a GGUF text encoder; or raise the macOS GPU limit "
            "until reboot with `sudo sysctl iogpu.wired_limit_mb=<MB>` (leave several GB for macOS).")
    return info, warnings


def comfy_python(comfy_dir: Path) -> str:
    """The Python ComfyUI should run with: its own venv if it has one."""
    for rel in (".venv/bin/python", "venv/bin/python", ".venv/Scripts/python.exe", "venv/Scripts/python.exe",
                "../python_embeded/python.exe"):
        candidate = comfy_dir / rel
        if candidate.exists():
            return str(candidate)
    return os.environ.get("COMFYUI_PYTHON", sys.executable)


def _tail(path: Path, lines: int = 12) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def _round32(v: int) -> int:
    return max(256, int(round(v / 32)) * 32)


def _gguf_help(what: str) -> str:
    return (f"Your {what} is a .gguf file, which needs the ComfyUI-GGUF custom node:\n"
            f"  cd <ComfyUI>/custom_nodes && git clone {GGUF_REPO}\n"
            "  <ComfyUI python> -m pip install -r ComfyUI-GGUF/requirements.txt\n"
            "then restart ComfyUI.")


def _format_validation_error(resp) -> str:
    try:
        data = resp.json()
    except ValueError:
        return f"ComfyUI rejected the workflow ({resp.status_code}): {resp.text[:500]}"
    lines = [f"ComfyUI rejected the workflow: {data.get('error', {}).get('message', data.get('error'))}"]
    for node_id, err in (data.get("node_errors") or {}).items():
        for e in err.get("errors", []):
            lines.append(f"  node {node_id} ({err.get('class_type')}): {e.get('message')} - {e.get('details', '')}")
    return "\n".join(lines)


def _format_execution_error(status: dict) -> str:
    for event, data in status.get("messages", []):
        if event == "execution_error":
            return (f"ComfyUI failed in {data.get('node_type')} (node {data.get('node_id')}): "
                    f"{data.get('exception_type')}: {data.get('exception_message', '').strip()}")
    return "ComfyUI reported an error while generating."

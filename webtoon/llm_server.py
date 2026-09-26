"""Running the local storyboarding LLM (llama.cpp's llama-server) only while it's needed.

A Mac can't hold a large LLM and Qwen-Image at the same time, so storyboarding and
drawing take turns: PlannerSession frees ComfyUI's memory, starts llama-server with
your GGUF on the first real planner call, and stops it again when planning is done.
"""

from __future__ import annotations

import atexit
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional

from .models import Project

DEFAULT_LLAMA_URL = "http://127.0.0.1:8080/v1"
LOG_PATH = Path.home() / ".cache" / "webtoon" / "llama-server.log"


class LLMServerError(RuntimeError):
    pass


def _tail(path: Path, lines: int = 15) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def find_llama_server(configured: str = "llama-server") -> Optional[str]:
    path = Path(configured).expanduser()
    if path.is_file():
        return str(path)
    found = shutil.which(configured)
    if found:
        return found
    for candidate in ("/opt/homebrew/bin/llama-server", "/usr/local/bin/llama-server",
                      str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-server")):
        if Path(candidate).is_file():
            return candidate
    return None


class LlamaServer:
    def __init__(self, project: Project, log=print):
        import requests

        self.requests = requests
        self.settings = project.planner
        self.log = log
        base = (self.settings.base_url or DEFAULT_LLAMA_URL).rstrip("/")
        self.root = re.sub(r"/v1$", "", base)
        port = re.search(r":(\d+)$", self.root)
        self.port = port.group(1) if port else "8080"
        self.process: Optional[subprocess.Popen] = None

    def healthy(self) -> bool:
        try:
            return self.requests.get(f"{self.root}/health", timeout=3).status_code == 200
        except self.requests.RequestException:
            return False

    def command(self, with_extra_args: bool = True) -> List[str]:
        s = self.settings
        binary = find_llama_server(s.llama_server)
        if not binary:
            raise LLMServerError(f"Can't find llama-server ('{s.llama_server}'). Install llama.cpp (brew install llama.cpp) "
                                 "or set the path to the llama-server binary in Settings > Planner.")
        model = Path(s.llm_model or "").expanduser()
        if not model.is_file():
            raise LLMServerError(f"LLM model file not found: {s.llm_model or '(not set)'}")
        cmd = [binary, "-m", str(model), "--host", "127.0.0.1", "--port", self.port, "-c", str(s.llm_context), "--jinja"]
        return cmd + (shlex.split(s.llm_args or "") if with_extra_args else [])

    def start(self) -> None:
        if self.healthy():
            return
        for with_extra in (True, False):
            cmd = self.command(with_extra)
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.log(f"Starting the storyboard LLM: {Path(cmd[2]).name} (log: {LOG_PATH})")
            with open(LOG_PATH, "ab") as log_file:
                self.process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            atexit.register(self.stop)
            deadline = time.time() + 1800  # big models can take a while to load from disk
            while time.time() < deadline:
                if self.process.poll() is not None:
                    tail = _tail(LOG_PATH)
                    bad_arg = re.search(r"(invalid|unknown) argument|error while handling argument", tail, re.I)
                    if with_extra and self.settings.llm_args and bad_arg:
                        self.log(f"  llama-server didn't accept the extra arguments '{self.settings.llm_args}' "
                                 "(older llama.cpp?) - retrying without them")
                        break
                    raise LLMServerError(f"llama-server exited while loading (code {self.process.returncode}):\n{tail}")
                if self.healthy():
                    self.log("  LLM loaded.")
                    return
                time.sleep(2)
            else:
                self.stop()
                raise LLMServerError(f"llama-server didn't become ready within 30 minutes; see {LOG_PATH}")

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.log("Stopping the storyboard LLM to free memory for drawing.")
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None


def free_comfyui(project: Project, log=print) -> None:
    """Ask a running ComfyUI to unload its models (no-op if it isn't running)."""
    if project.image.backend != "comfyui":
        return
    import requests

    url = (project.image.base_url or "http://127.0.0.1:8188").rstrip("/")
    try:
        if requests.post(f"{url}/free", json={"unload_models": True, "free_memory": True}, timeout=5).ok:
            log("Asked ComfyUI to unload its models to make room for the LLM.")
    except requests.RequestException:
        pass


def is_local_url(url: Optional[str]) -> bool:
    return bool(url) and re.search(r"//(localhost|127\.|0\.0\.0\.0|\[::1\])", url or "") is not None


class PlannerSession:
    """Context manager around one or more chapters of planning.

    The LLM is only started when a planner call actually happens (cached segments
    don't need it), and stopped on exit if this session started it.
    """

    def __init__(self, project: Project, log=print):
        self.project = project
        self.log = log
        self.server: Optional[LlamaServer] = None
        self.ready = False

    def ensure_ready(self) -> None:
        if self.ready:
            return
        provider = self.project.planner.provider
        local = provider == "llamacpp" or (provider == "openai" and is_local_url(self.project.planner.base_url))
        if local:
            free_comfyui(self.project, self.log)
        if provider == "llamacpp":
            self.server = LlamaServer(self.project, self.log)
            self.server.start()
        self.ready = True

    def close(self) -> None:
        if self.server is not None:
            self.server.stop()
            self.server = None
        self.ready = False

    def __enter__(self) -> "PlannerSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def llm_file_warning(path: Optional[str], ram_gb: Optional[float]) -> Optional[str]:
    """Warn when a GGUF alone won't fit in the ~75% of unified memory the GPU may use."""
    if not path or not ram_gb or not Path(path).expanduser().is_file():
        return None
    size = Path(path).expanduser().stat().st_size / 1024 ** 3
    budget = ram_gb * 0.75
    if size + 3 > budget:  # + KV cache for a 32k context
        return (f"This model file is {size:.1f} GB; with its context it needs more than the ~{budget:.0f} GB the GPU can "
                "use. Use a smaller quantization (e.g. Q4_K_M / Q5_K_M) or lower the context size.")
    return None

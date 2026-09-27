"""Interactive terminal app: `python -m webtoon` (or `webtoon` after `pip install -e .`).

Menus to create/open a project, point it at your Qwen-Image model and planner,
add chapters, redraw panels, review storyboards and edit characters.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Any, Callable, List, Optional

import questionary
from prompt_toolkit.completion import PathCompleter
from questionary import Choice, Separator
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from .comfyui import (GGUF_REPO, MODEL_SUBDIRS, comfy_python, detect_model_files, memory_report,
                      system_memory_gb)
from .llm_server import PlannerSession, find_llama_server, llm_file_warning
from .image_backends import make_backend, portrait_size
from .models import Project
from .pipeline import assemble_step, ensure_reference_sheets, plan_step, render_step
from .planner import plan_chapter
from .project import ProjectDir
from .prompts import reference_sheet_prompt
from .segment import split_segments, target_panels
from .textio import ChapterFileError, encoding_label, read_chapter

CONFIG_PATH = Path(os.environ.get("WEBTOON_CONFIG", Path.home() / ".config" / "webtoon" / "app.json"))

IMAGE_BACKENDS = {
    "comfyui": "Qwen-Image 2.1 model files (.gguf / .safetensors) through ComfyUI - recommended",
    "diffusers": "A diffusers model folder (Qwen-Image 1.x; runs in-process)",
    "openai": "My own server with an OpenAI-style /v1/images API (vLLM-Omni, LocalAI, custom...)",
    "dashscope": "Alibaba Cloud Model Studio (DashScope API)",
    "mock": "Placeholder images (test the pipeline without a GPU)",
}
PLANNERS = {
    "anthropic": "Claude (Anthropic API)",
    "llamacpp": "A local GGUF model with llama.cpp - loaded only while storyboarding, unloaded before drawing",
    "openai": "An OpenAI-compatible chat server you run yourself (Ollama, LM Studio, vLLM...)",
    "mock": "No LLM: one caption panel per paragraph (for testing)",
}
API_KEYS = {
    "ANTHROPIC_API_KEY": "Claude planner",
    "OPENAI_API_KEY": "OpenAI-compatible planner server (if it needs one)",
    "DASHSCOPE_API_KEY": "DashScope image backend",
    "IMAGE_API_KEY": "OpenAI-style image server (if it needs one)",
}


class Back(Exception):
    """Raised to leave the current submenu."""


def _needs_comfyui(model: str) -> bool:
    """True for a local file, or a folder that isn't in diffusers format."""
    path = Path(model).expanduser()
    return path.is_file() or (path.is_dir() and not (path / "model_index.json").exists())


def _valid_comfy_dir(value: str):
    if not value.strip():
        return True
    folder = Path(value).expanduser()
    return (folder / "main.py").exists() or (folder / "models").is_dir() or "Not a ComfyUI folder (no main.py or models/)"


def parse_panel_list(text: str) -> set[int]:
    """'3, 7-9' -> {3, 7, 8, 9}"""
    result: set[int] = set()
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            result.update(range(int(a), int(b) + 1))
        else:
            result.add(int(part))
    return result


class App:
    def __init__(self, project_path: Optional[str] = None, console: Optional[Console] = None, **prompt_kwargs: Any):
        self.console = console or Console()
        self.prompt_kwargs = prompt_kwargs  # lets tests drive prompts with a pipe input
        self.config = self._load_config()
        self.pdir: Optional[ProjectDir] = None
        self.project: Optional[Project] = None
        self._backend = None
        self._backend_key = ""
        self._checked_setup: Optional[Path] = None
        if project_path:
            self.open_project(Path(project_path), create_if_missing=True)

    # ------------------------------------------------------------------ prompts
    def _ask(self, question):
        answer = question.unsafe_ask()
        return answer

    def select(self, message: str, choices: List, default=None):
        return self._ask(questionary.select(message, choices=choices, default=default, **self.prompt_kwargs))

    def text(self, message: str, default: str = "", validate: Optional[Callable] = None, multiline: bool = False):
        return self._ask(questionary.text(message, default=default or "", validate=validate, multiline=multiline,
                                          **self.prompt_kwargs))

    def path(self, message: str, default: str = "", only_directories: bool = False, validate=None):
        # A text prompt with a path completer: Tab completes, Enter always submits.
        # (questionary.path opens its menu on every "/" and then needs a second Enter.)
        answer = self._ask(questionary.text(
            message, default=default or "", validate=validate,
            completer=PathCompleter(expanduser=True, only_directories=only_directories),
            complete_while_typing=False, **self.prompt_kwargs))
        return answer.rstrip("/\\") if len(answer) > 1 else answer

    def confirm(self, message: str, default: bool = True) -> bool:
        return self._ask(questionary.confirm(message, default=default, **self.prompt_kwargs))

    def number(self, message: str, default, kind=int):
        def valid(v):
            try:
                return kind(v) > 0 or "Must be positive"
            except ValueError:
                return f"Enter a {'whole ' if kind is int else ''}number"
        return kind(self.text(message, str(default), validate=valid))

    # ------------------------------------------------------------------ config
    def _load_config(self) -> dict:
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"recent_projects": [], "defaults": {}}

    def _save_config(self) -> None:
        try:
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            CONFIG_PATH.write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        except OSError as exc:
            self.console.print(f"[yellow]Could not save app settings to {CONFIG_PATH}: {exc}[/]")

    def _remember(self, path: Path) -> None:
        recent = [p for p in self.config.get("recent_projects", []) if p != str(path.resolve())]
        self.config["recent_projects"] = [str(path.resolve())] + recent[:9]
        self._save_config()

    def save(self) -> None:
        assert self.pdir and self.project
        self.pdir.save(self.project)

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        self.console.print(Panel.fit("[bold]Webtoon studio[/] - story chapters to webtoon strips with Qwen-Image",
                                     border_style="magenta"))
        try:
            if self.project is None:
                self.choose_project()
            while True:
                self.check_setup()
                self.show_header()
                try:
                    if not self.main_menu():
                        break
                except (Back, KeyboardInterrupt):
                    self.console.print("[dim]cancelled[/]")
                except Exception as exc:  # keep the app alive on errors in one action
                    self.console.print(Panel(escape(f"{type(exc).__name__}: {exc}"), title="Error", border_style="red"))
        except (Back, KeyboardInterrupt, EOFError):
            pass
        self.console.print("Bye!")

    def main_menu(self) -> bool:
        p = self.project
        next_n = max((c.number for c in p.chapters), default=0) + 1
        choices = [Choice(f"Add chapter {next_n}", "add"),
                   Choice("Add several chapters (storyboard all, then draw all)", "batch")]
        if p.chapters:
            choices += [Choice("Draw / redraw a chapter", "render"), Choice("Review a chapter's storyboard", "review"),
                        Choice("Open a chapter in the browser", "open")]
        choices += [
            Choice(f"Characters ({len(p.characters)})", "chars"),
            Separator(),
            Choice("Settings: models, paths, pacing, style", "settings"),
            Choice("Test the image model", "test_image"),
            Choice("Test the planner", "test_planner"),
            Separator(),
            Choice("Switch project", "switch"),
            Choice("Quit", "quit"),
        ]
        action = self.select("What do you want to do?", choices)
        if action == "quit":
            return False
        {
            "add": self.add_chapter, "batch": self.add_chapters, "render": self.render_chapter, "review": self.review_chapter,
            "open": self.open_reader, "chars": self.characters_menu, "settings": self.settings_menu,
            "test_image": self.test_image, "test_planner": self.test_planner, "switch": self.choose_project,
        }[action]()
        return True

    def show_header(self) -> None:
        p = self.project
        img, pl = p.image, p.planner
        rows = Table.grid(padding=(0, 2))
        rows.add_column(style="bold cyan")
        rows.add_column()
        rows.add_row("Project", f"{p.title}  [dim]{self.pdir.root}[/]")
        if img.backend == "comfyui":
            where = img.comfy_dir or img.base_url or "http://127.0.0.1:8188"
            rows.add_row("Image model", f"Qwen-Image 2.1 via ComfyUI: {Path(img.diffusion_model or '-').name}  [dim]{where}[/]")
        else:
            rows.add_row("Image model", f"{img.backend}: {img.model}" + (f"  @ {img.base_url}" if img.base_url else ""))
        if pl.provider == "llamacpp":
            rows.add_row("Planner", f"llama.cpp: {escape(pl.model)}  [dim]loaded only while storyboarding[/]")
        else:
            rows.add_row("Planner", f"{pl.provider}: {pl.model}" + (f"  @ {pl.base_url}" if pl.base_url else ""))
        rows.add_row("Chapters", ", ".join(f"{c.number}. {c.title}" for c in p.chapters) or "none yet")
        warnings = self.config_warnings()
        if warnings:
            rows.add_row("[yellow]Check[/]", "[yellow]" + "\n".join(warnings) + "[/]")
        self.console.print(Panel(rows, border_style="magenta"))

    def config_warnings(self) -> List[str]:
        p, w = self.project, []
        if p.image.backend == "diffusers" and not Path(p.image.model).expanduser().exists() and "/" not in p.image.model:
            w.append(f"image model path '{p.image.model}' does not exist")
        elif p.image.backend == "diffusers" and not Path(p.image.model).expanduser().exists():
            w.append(f"'{p.image.model}' is not a local folder - it will be downloaded from Hugging Face")
        if p.image.backend == "openai" and not p.image.base_url:
            w.append("image server URL is not set")
        if p.image.backend == "comfyui" and not p.image.workflow_file:
            for kind in MODEL_SUBDIRS:
                value = getattr(p.image, kind)
                if not value:
                    w.append(f"{kind.replace('_', ' ')} file is not set (Settings > Image model)")
                elif os.sep in value and not Path(value).expanduser().exists():
                    w.append(f"{kind.replace('_', ' ')} file not found: {value}")
            comfy = Path(p.image.comfy_dir).expanduser() if p.image.comfy_dir else None
            if comfy and _valid_comfy_dir(str(comfy)) is not True:
                w.append(f"no ComfyUI at {comfy}")
            files = {k: Path(getattr(p.image, k)).expanduser() if getattr(p.image, k) else None for k in MODEL_SUBDIRS}
            for warning in memory_report(files)[1]:
                w.append(warning.split(". ")[0] + " - see README 'Running on a Mac'")
            if comfy and str(p.image.diffusion_model or "").lower().endswith(".gguf") and \
                    not (comfy / "custom_nodes" / "ComfyUI-GGUF").exists():
                w.append("ComfyUI-GGUF custom node is not installed (needed for .gguf) - Settings > Image model")
        if p.image.backend == "dashscope" and not os.environ.get("DASHSCOPE_API_KEY"):
            w.append("DASHSCOPE_API_KEY is not set")
        if p.planner.provider == "anthropic" and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            w.append("ANTHROPIC_API_KEY is not set (fine if you used `ant auth login`)")
        if p.planner.provider == "openai" and not p.planner.base_url:
            w.append("planner server URL is not set")
        if p.planner.provider == "llamacpp":
            if not p.planner.llm_model or not Path(p.planner.llm_model).expanduser().is_file():
                w.append("storyboard LLM .gguf file is not set / not found (Settings > Planner)")
            elif not find_llama_server(p.planner.llama_server):
                w.append("llama-server not found (Settings > Planner)")
            else:
                warning = llm_file_warning(p.planner.llm_model, system_memory_gb())
                if warning:
                    w.append(warning)
        if p.planner.provider == "mock":
            w.append("planner is 'mock' (test mode): every paragraph just becomes a caption. Choose Claude or a "
                     "local LLM in Settings for a real storyboard")
        if p.image.backend == "diffusers" and _needs_comfyui(p.image.model):
            w.append("the diffusers backend can't load single-file / Qwen-Image 2.1 checkpoints - switch to ComfyUI "
                     "in Settings > Backend and model files")
        return w

    def check_setup(self) -> None:
        """Once per opened project: offer to move a project whose model files need ComfyUI onto it."""
        if self._checked_setup == self.pdir.root:
            return
        self._checked_setup = self.pdir.root
        self.offer_comfyui()

    def offer_comfyui(self) -> bool:
        """If the project points diffusers at single-file checkpoints, offer to switch it to ComfyUI.

        Returns True when the image settings are usable afterwards.
        """
        img = self.project.image
        if img.backend != "diffusers" or not _needs_comfyui(img.model):
            return True
        folder = Path(img.model).expanduser()
        folder = folder if folder.is_dir() else folder.parent
        self.console.print(Panel(
            f"This project uses the diffusers backend with {escape(str(folder))}, but that folder holds single-file "
            "checkpoints (like Qwen-Image 2.1's .gguf / .safetensors files), which diffusers can't load. "
            "They run through ComfyUI instead.", title="Image model setup", border_style="yellow"))
        if not self.confirm("Set up ComfyUI for this project now?", default=True):
            self.console.print("[dim]OK - you can do it any time in Settings > Backend and model files.[/]")
            return False
        img.backend = "comfyui"
        self.save()
        self.configure_comfyui(folder)
        if self.confirm("Also make these model settings the default for new projects?", default=True):
            self.save_defaults()
        return True

    # ------------------------------------------------------------------ projects
    def choose_project(self) -> None:
        recent = [r for r in self.config.get("recent_projects", []) if (Path(r) / "story.json").exists()]
        choices = [Choice(f"Open {r}", r) for r in recent]
        choices += [Choice("Create a new project", "new"), Choice("Open a project folder...", "open")]
        if self.project is not None:
            choices.append(Choice("Back", "back"))
        choice = self.select("Project", choices)
        if choice == "back":
            return
        if choice == "new":
            self.new_project()
        elif choice == "open":
            folder = self.path("Project folder (contains story.json):", only_directories=True,
                               validate=lambda v: (Path(v).expanduser() / "story.json").exists() or "No story.json there")
            self.open_project(Path(folder).expanduser())
        else:
            self.open_project(Path(choice))

    def open_project(self, path: Path, create_if_missing: bool = False) -> None:
        pdir = ProjectDir(path)
        if not pdir.exists():
            if not create_if_missing:
                raise Back()
            self.pdir, self.project = pdir, self._new_project_state(pdir, path.name.replace("_", " ").title())
            self.console.print(f"Created project at {path}")
        else:
            self.pdir, self.project = pdir, pdir.load()
        self._remember(path)

    def _new_project_state(self, pdir: ProjectDir, title: str) -> Project:
        project = pdir.create(title)
        defaults = self.config.get("defaults") or {}
        if defaults:
            project = Project.model_validate({**project.model_dump(), **defaults, "title": title})
            pdir.save(project)
        return project

    def new_project(self) -> None:
        title = self.text("Story title:", validate=lambda v: bool(v.strip()) or "Enter a title")
        default_dir = Path.cwd() / "".join(ch if ch.isalnum() else "_" for ch in title.lower()).strip("_")
        folder = Path(self.path("Where should the project live?", str(default_dir), only_directories=True)).expanduser()
        if (folder / "story.json").exists():
            self.console.print("[yellow]That folder already has a project; opening it.[/]")
            self.open_project(folder)
            return
        self.pdir = ProjectDir(folder)
        self.project = self._new_project_state(self.pdir, title.strip())
        self._remember(folder)
        if self.config.get("defaults"):
            self.console.print("Using your saved default settings.")
            if not self.confirm("Change the model settings for this project?", default=False):
                return
        self.console.print("\n[bold]Image model[/]")
        self.configure_image()
        self.console.print("\n[bold]Planner (the LLM that storyboards chapters)[/]")
        self.configure_planner()
        if self.confirm("Save these model settings as the defaults for new projects?", default=True):
            self.save_defaults()

    # ------------------------------------------------------------------ settings
    def configure_image(self) -> None:
        img = self.project.image
        img.backend = self.select("How do you run Qwen-Image?", [Choice(v, k) for k, v in IMAGE_BACKENDS.items()],
                                  default=img.backend)
        if img.backend == "comfyui":
            self.configure_comfyui()
            return
        if img.backend == "diffusers":
            model = self.path("Model folder (or a Hugging Face repo id):", img.model)
            path = Path(model).expanduser()
            if path.is_file() or (path.is_dir() and not (path / "model_index.json").exists()):
                self.console.print("[yellow]That is not a diffusers model folder (no model_index.json). Single-file "
                                   "checkpoints such as Qwen-Image 2.1 (.gguf / .safetensors) run through ComfyUI.[/]")
                if self.confirm("Set it up with ComfyUI instead?", default=True):
                    img.backend = "comfyui"
                    self.configure_comfyui(path if path.is_dir() else path.parent)
                    return
            img.model = str(path) if path.exists() else model.strip()
            img.base_url = None
        elif img.backend == "openai":
            img.base_url = self.text("Server base URL:", img.base_url or "http://localhost:8000/v1",
                                     validate=lambda v: v.startswith("http") or "Must start with http").strip()
            img.model = self.text("Model name on the server:", img.model or "qwen-image-2.1").strip()
        elif img.backend == "dashscope":
            img.model = self.text("DashScope model name:", img.model).strip()
            url = self.text("Endpoint (leave empty for the default international endpoint):", img.base_url or "")
            img.base_url = url.strip() or None
            self._ensure_key("DASHSCOPE_API_KEY")
        if img.backend != "mock":
            img.use_references = self.confirm(
                "Does this model accept reference images (Qwen-Image-Edit style)? If yes, character sheets are "
                "passed with every panel for stronger consistency.", default=img.use_references)
        self.save()

    def configure_comfyui(self, folder: Optional[Path] = None) -> None:
        img = self.project.image
        self.console.print("[dim]Qwen-Image 2.1 comes as three files: the diffusion model, the Qwen3-VL text encoder "
                           "and the VAE. They are run with ComfyUI, which supports the model natively.[/]")
        start = folder or (Path(img.diffusion_model).expanduser().parent if img.diffusion_model else None)
        folder = Path(self.path("Folder with your Qwen-Image 2.1 model files:", str(start or ""), only_directories=True,
                                validate=lambda v: Path(v).expanduser().is_dir() or "Folder not found")).expanduser()
        found = detect_model_files(folder)
        table = Table(show_header=False, box=None)
        for kind, path in found.items():
            table.add_row(kind.replace("_", " "), path.name if path else "[red]not found[/]")
        self.console.print(Panel(table, title="Detected model files"))
        if not all(found.values()) or not self.confirm("Use these files?", default=True):
            for kind in MODEL_SUBDIRS:
                default = found[kind] or (Path(getattr(img, kind)) if getattr(img, kind) else folder)
                chosen = self.path(f"{kind.replace('_', ' ').capitalize()} file:", str(default),
                                   validate=lambda v: Path(v).expanduser().is_file() or "File not found")
                found[kind] = Path(chosen).expanduser()
        for kind, path in found.items():
            setattr(img, kind, str(path.resolve()))
        img.model = found["diffusion_model"].name
        info, warnings = memory_report(found)
        self.console.print("[dim]" + " | ".join(info) + "[/]")
        for warning in warnings:
            self.console.print(Panel(escape(warning), title="Memory", border_style="yellow"))

        comfy = self.path("Your ComfyUI folder (with main.py, or the ComfyUI Desktop folder with models/). The files "
                          "get linked into it. Leave empty if ComfyUI runs on another machine:",
                          img.comfy_dir or "", only_directories=True, validate=_valid_comfy_dir)
        img.comfy_dir = str(Path(comfy).expanduser().resolve()) if comfy.strip() else None
        desktop = bool(img.comfy_dir) and not (Path(img.comfy_dir) / "main.py").exists()
        if desktop:
            self.console.print("[dim]That looks like ComfyUI Desktop: open the ComfyUI app before drawing. Its server "
                               "address is in the app's settings (usually port 8000).[/]")
        default_url = img.base_url or ("http://127.0.0.1:8000" if desktop else "http://127.0.0.1:8188")
        img.base_url = self.text("ComfyUI address:", default_url,
                                 validate=lambda v: v.startswith("http") or "Must start with http").strip()
        if not img.comfy_dir:
            self.console.print("[dim]Make sure ComfyUI can see the files: put (or link) them in ComfyUI/models/"
                               "diffusion_models, text_encoders and vae.[/]")
        uses_gguf = any(str(getattr(img, k) or "").lower().endswith(".gguf") for k in MODEL_SUBDIRS)
        if uses_gguf and img.comfy_dir and not (Path(img.comfy_dir) / "custom_nodes" / "ComfyUI-GGUF").exists():
            self.console.print("[yellow]Loading .gguf files needs the ComfyUI-GGUF custom node, which isn't installed.[/]")
            if self.confirm("Install it now (git clone + pip install into ComfyUI's Python)?", default=True):
                self.install_gguf_node(Path(img.comfy_dir))

        if self.confirm("Use the official Qwen-Image 2.1 sampling settings (25 steps, CFG 1.0, euler / simple)?",
                        default=True):
            img.steps, img.cfg, img.sampler, img.scheduler = 25, 1.0, "euler", "simple"
        img.use_references = self.confirm(
            "Pass each character's reference sheet to Qwen-Image 2.1 with every panel they appear in? "
            "(strongest consistency; slightly slower)", default=img.use_references)
        self.save()
        if self.confirm("Generate a test image now?", default=False):
            self.test_image()

    def install_gguf_node(self, comfy_dir: Path) -> None:
        target = comfy_dir / "custom_nodes" / "ComfyUI-GGUF"
        python = comfy_python(comfy_dir)
        for cmd in (["git", "clone", "--depth", "1", GGUF_REPO, str(target)],
                    [python, "-m", "pip", "install", "-r", str(target / "requirements.txt")]):
            self.console.print(f"$ {' '.join(cmd)}", markup=False)
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.console.print((result.stdout + result.stderr)[-2000:], markup=False)
                self.console.print("[red]Install failed - see the output above.[/]")
                return
        self.console.print(f"[green]ComfyUI-GGUF installed (using {python}).[/] Restart ComfyUI if it is already running.")

    def configure_planner(self) -> None:
        pl = self.project.planner
        provider = self.select("Which LLM should storyboard the chapters?",
                               [Choice(v, k) for k, v in PLANNERS.items()], default=pl.provider)
        if provider != pl.provider:
            pl.model = {"anthropic": "claude-opus-5", "mock": "mock"}.get(provider, "")
        pl.provider = provider
        if provider == "anthropic":
            pl.model = self.text("Claude model:", pl.model or "claude-opus-5").strip()
            pl.base_url = None
            if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
                self._ensure_key("ANTHROPIC_API_KEY", optional=True)
        elif provider == "llamacpp":
            self.configure_llamacpp()
        elif provider == "openai":
            self.console.print(
                "[dim]Local LLM tips (e.g. Ollama on a Mac): use a model that is good at long JSON, such as "
                "qwen3:14b or qwen3:30b (the 30B MoE runs fast on Apple Silicon). Ollama's default context window "
                "is far too small for storyboarding and it silently cuts the chapter off - start it with "
                "OLLAMA_CONTEXT_LENGTH=32768, and OLLAMA_KEEP_ALIVE=0 so it frees memory before ComfyUI draws. "
                "If the model struggles, lower 'Segment size' in Settings to ~600 words.[/]")
            pl.base_url = self.text("Chat server base URL:", pl.base_url or "http://localhost:11434/v1",
                                    validate=lambda v: v.startswith("http") or "Must start with http").strip()
            pl.model = self.text("Model name on the server:", pl.model or "qwen3:30b",
                                 validate=lambda v: bool(v.strip()) or "Required").strip()
        self.save()

    def configure_llamacpp(self) -> None:
        pl = self.project.planner
        self.console.print("[dim]The app starts llama-server with this model when it storyboards, and stops it before "
                           "drawing, so the LLM and Qwen-Image never need memory at the same time.[/]")
        model = self.path("Your LLM's .gguf file:", pl.llm_model or "",
                          validate=lambda v: (Path(v).expanduser().is_file() and v.lower().endswith(".gguf"))
                          or "Pick a .gguf file")
        pl.llm_model = str(Path(model).expanduser().resolve())
        pl.model = Path(pl.llm_model).name
        size = Path(pl.llm_model).stat().st_size / 1024 ** 3
        self.console.print(f"[dim]{escape(pl.model)}: {size:.1f} GB[/]")
        warning = llm_file_warning(pl.llm_model, system_memory_gb())
        if warning:
            self.console.print(Panel(escape(warning), title="Memory", border_style="yellow"))
        found = find_llama_server(pl.llama_server)
        if found:
            self.console.print(f"[dim]Using llama-server at {escape(found)}[/]")
            pl.llama_server = found
        else:
            self.console.print("[yellow]llama-server wasn't found on your PATH.[/]")
            server = self.path("Path to the llama-server binary (from your llama.cpp build or `brew install llama.cpp`):",
                               validate=lambda v: Path(v).expanduser().is_file() or "File not found")
            pl.llama_server = str(Path(server).expanduser().resolve())
        pl.llm_context = self.number("Context size in tokens (32768 fits a ~1200-word segment with room to spare):",
                                     pl.llm_context)
        pl.llm_args = self.text("Extra llama-server arguments (the default turns thinking off: faster, more reliable "
                                "JSON):", pl.llm_args).strip()
        current_port = re.search(r":(\d+)", pl.base_url or "")
        port = self.text("Port for llama-server:", current_port.group(1) if current_port else "8080",
                         validate=lambda v: v.isdigit() or "Enter a port number")
        pl.base_url = f"http://127.0.0.1:{port}/v1"
        self.save()

    def _ensure_key(self, name: str, optional: bool = False) -> None:
        if os.environ.get(name):
            return
        hint = " (leave empty to skip)" if optional else ""
        value = self._ask(questionary.password(f"{name} for this session{hint}:", **self.prompt_kwargs))
        if value:
            os.environ[name] = value.strip()
            self.console.print(f"[dim]{name} set for this session only - export it in your shell to keep it.[/]")

    def save_defaults(self) -> None:
        p = self.project
        self.config["defaults"] = {
            "image": p.image.model_dump(), "planner": p.planner.model_dump(), "style": p.style,
            "negative_prompt": p.negative_prompt, "width": p.width, "font": p.font,
        }
        self._save_config()
        self.console.print(f"[green]Saved defaults to {CONFIG_PATH}[/]")

    def settings_menu(self) -> None:
        while True:
            p, img, pl = self.project, self.project.image, self.project.planner
            short = lambda s, n=50: (s[:n] + "...") if s and len(s) > n else (s or "-")  # noqa: E731
            options = [
                Separator("-- Image model --"),
                Choice("Backend and model files: " + (f"ComfyUI: {Path(img.diffusion_model or '-').name}"
                       if img.backend == "comfyui" else f"{img.backend}: {short(img.model, 60)}"), "image"),
                Choice(f"Sampling steps: {img.steps}", "steps"),
                Choice(f"Panel resolution: {img.megapixels:g} MP", "megapixels"),
                Choice(f"CFG scale: {img.cfg}", "cfg"),
                Choice(f"Pass character sheets as references: {'yes' if img.use_references else 'no'}", "refs"),
            ] + ([
                Choice(f"ComfyUI: {img.comfy_dir or 'not managed'} @ {img.base_url or 'http://127.0.0.1:8188'}"
                       f"{'  args: ' + img.comfy_args if img.comfy_args else ''}", "comfy"),
                Choice(f"Sampler / scheduler: {img.sampler} / {img.scheduler}", "sampler"),
                Choice(f"Custom ComfyUI workflow: {img.workflow_file or 'none (built-in Qwen-Image 2.1 graph)'}",
                       "workflow"),
            ] if img.backend == "comfyui" else []) + [
                Separator("-- Planner --"),
                Choice(f"Planner: {pl.provider}: {pl.model}" + (f" @ {pl.base_url}" if pl.base_url else ""), "planner"),
                Choice(f"Segment size for long chapters: {pl.segment_words} words", "segment"),
                Choice(f"Pacing: {pl.panels_per_1000_words:g} panels per 1000 words", "density"),
                Choice(f"Re-plan attempts when text is skipped: {pl.max_retries}", "retries"),
                Separator("-- Look --"),
                Choice(f"Art style: {short(p.style)}", "style"),
                Choice(f"Negative prompt: {short(p.negative_prompt)}", "negative"),
                Choice(f"Strip width: {p.width}px", "width"),
                Choice("Text: " + ("drawn by Qwen-Image into the art" if p.lettering == "model" else
                                   "added on top by the app"), "lettering"),
                Choice(f"Lettering font: {p.font or 'auto'}", "font"),
                Separator("-- Other --"),
                Choice("API keys for this session", "keys"),
                Choice("Save these settings as defaults for new projects", "defaults"),
                Choice("Back", "back"),
            ]
            what = self.select("Settings", options)
            if what == "back":
                return
            if what == "image":
                self.configure_image()
            elif what == "planner":
                self.configure_planner()
            elif what == "comfy":
                comfy = self.path("ComfyUI folder (empty = ComfyUI runs on another machine):", img.comfy_dir or "",
                                  only_directories=True, validate=_valid_comfy_dir)
                img.comfy_dir = str(Path(comfy).expanduser().resolve()) if comfy.strip() else None
                img.base_url = self.text("ComfyUI address:", img.base_url or "http://127.0.0.1:8188").strip()
                img.comfy_args = self.text("Extra launch arguments (e.g. --lowvram; empty for none):",
                                           img.comfy_args).strip()
            elif what == "sampler":
                img.sampler = self.text("Sampler name (ComfyUI):", img.sampler).strip()
                img.scheduler = self.text("Scheduler (ComfyUI):", img.scheduler).strip()
            elif what == "workflow":
                self.console.print("[dim]Optional: build your own graph in ComfyUI (LoRAs, upscaling...), use "
                                   "{{prompt}} {{negative}} {{seed}} {{width}} {{height}} {{steps}} {{cfg}} and "
                                   "{{ref_1}}..{{ref_16}} (LoadImage) as values, then Workflow > Export (API).[/]")
                wf = self.path("Workflow JSON (empty = built-in):", img.workflow_file or "",
                               validate=lambda v: not v.strip() or Path(v).expanduser().is_file() or "File not found")
                img.workflow_file = str(Path(wf).expanduser().resolve()) if wf.strip() else None
            elif what == "megapixels":
                self.console.print("[dim]Panels are shrunk to the strip width afterwards, so 1 MP is plenty for "
                                   "screens. Higher = slower (roughly proportional to pixel count).[/]")
                img.megapixels = self.number("Megapixels per panel:", img.megapixels, float)
            elif what == "steps":
                img.steps = self.number("Sampling steps:", img.steps)
            elif what == "cfg":
                img.cfg = self.number("CFG scale (Qwen-Image 2.1 uses 1.0):", img.cfg, float)
            elif what == "refs":
                img.use_references = self.confirm("Pass character sheets to the model as reference images?",
                                                  img.use_references)
            elif what == "segment":
                self.console.print("[dim]Long chapters are storyboarded in parts of about this size. Smaller parts "
                                   "= more faithful, more planner calls. Use ~600-1000 for small local LLMs.[/]")
                pl.segment_words = self.number("Words per segment:", pl.segment_words)
            elif what == "density":
                self.console.print("[dim]Typical webtoon pacing is 8-15 panels per 1000 words.[/]")
                pl.panels_per_1000_words = self.number("Panels per 1000 words:", pl.panels_per_1000_words, float)
            elif what == "retries":
                pl.max_retries = int(self.text("Re-plan attempts:", str(pl.max_retries),
                                               validate=lambda v: v.isdigit() or "Enter 0 or more"))
            elif what == "style":
                p.style = self.text("Art style prompt (Esc then Enter to finish):", p.style, multiline=True).strip()
            elif what == "negative":
                p.negative_prompt = self.text("Negative prompt (Esc then Enter to finish):", p.negative_prompt,
                                              multiline=True).strip()
            elif what == "width":
                p.width = self.number("Strip width in pixels:", p.width)
            elif what == "lettering":
                self.console.print(
                    "[dim]Drawn by Qwen-Image: dialogue, captions, game windows and sound effects are part of the "
                    "image prompt, so the model letters them in its own style. It can occasionally misspell a word "
                    "(redraw that panel), and changing text means redrawing the art. Texts longer than the limit - "
                    "e.g. character sheets - are still added by the app.\n"
                    "Added by the app: exact text every time, instant re-lettering, simpler look.[/]")
                old = p.lettering
                p.lettering = self.select("Who draws the text?", [
                    Choice("Qwen-Image draws it into the art", "model"),
                    Choice("The app adds it on top of the art", "app")], default=p.lettering)
                if p.lettering == "model":
                    p.model_text_max_words = self.number(
                        "Longest text (in words) Qwen-Image should letter; longer ones are added by the app:",
                        p.model_text_max_words)
                if p.lettering != old and p.chapters:
                    self.console.print("[dim]Existing chapters keep their current art until redrawn: Draw / redraw a "
                                       "chapter > 'Missing panels, and panels whose prompt changed' redraws just the "
                                       "panels that have text.[/]")
            elif what == "font":
                font = self.path("Path to a .ttf font (empty = auto):", p.font or "",
                                 validate=lambda v: not v or Path(v).expanduser().is_file() or "File not found")
                p.font = str(Path(font).expanduser()) if font else None
            elif what == "keys":
                self.keys_menu()
            elif what == "defaults":
                self.save_defaults()
            self.save()

    def keys_menu(self) -> None:
        choices = [Choice(f"{k}: {'set' if os.environ.get(k) else 'not set'} - {v}", k) for k, v in API_KEYS.items()]
        key = self.select("Set which key? (kept for this session only)", choices + [Choice("Back", "back")])
        if key != "back":
            value = self._ask(questionary.password(f"{key}:", **self.prompt_kwargs))
            if value:
                os.environ[key] = value.strip()

    # ------------------------------------------------------------------ backend
    def backend(self):
        if not self.offer_comfyui():
            raise Back()
        key = self.project.image.model_dump_json()
        if self._backend is None or key != self._backend_key:
            with self.console.status(f"Loading {self.project.image.backend}: {self.project.image.model} ..."):
                self._backend = make_backend(self.project.image)
            self._backend_key = key
        return self._backend

    def log(self, message: str) -> None:
        self.console.print(message, markup=False, highlight=False)

    # ------------------------------------------------------------------ chapters
    def release_image_model(self) -> None:
        """Stop the ComfyUI this app started, so the storyboard LLM gets the memory."""
        backend, self._backend, self._backend_key = self._backend, None, ""
        if backend is not None and getattr(backend, "process", None) is not None:
            self.log("Stopping ComfyUI to free memory for storyboarding.")
            backend.shutdown()

    def planner_ready(self) -> bool:
        p = self.project
        if p.planner.provider == "mock":
            self.console.print(Panel(
                "The planner is set to 'mock', a test mode: it pastes each paragraph in as the image prompt, with no "
                "shots, no character designs and no story beats - the images won't follow the story and characters "
                "won't stay consistent. Use Claude or a local LLM to storyboard for real.",
                title="No real storyboarder", border_style="yellow"))
            choice = self.select("What now?", [Choice("Set up a real planner now (recommended)", "setup"),
                                               Choice("Continue with the mock planner anyway", "mock"),
                                               Choice("Cancel", "cancel")])
            if choice == "cancel":
                return False
            if choice == "setup":
                self.configure_planner()
                return p.planner.provider != "mock"
        return True

    def add_chapter(self) -> None:
        p = self.project
        if not self.planner_ready():
            return
        next_n = max((c.number for c in p.chapters), default=0) + 1
        file = self.path("Chapter text file:", validate=lambda v: Path(v).expanduser().is_file() or "File not found")
        file_path = Path(file).expanduser().resolve()
        try:
            text, encoding = read_chapter(file_path)
        except ChapterFileError as exc:
            self.console.print(Panel(escape(str(exc)), title="Can't read this chapter file", border_style="red"))
            return
        if encoding not in ("utf-8", "utf-8-sig"):
            self.console.print(f"[dim]{escape(file_path.name)} isn't UTF-8; read it as {encoding_label(encoding)}. "
                               f"First line: {escape(text.strip().splitlines()[0][:80]) if text.strip() else '(empty)'}[/]")
        number = self.number("Chapter number:", next_n)
        if any(c.number == number for c in p.chapters) and not self.confirm(
                f"Chapter {number} already exists. Redo it (storyboard and images)?", default=False):
            return

        segments = split_segments(text, p.planner.segment_words)
        words = sum(s.words for s in segments)
        est = sum(target_panels(s.words, p.planner.panels_per_1000_words) for s in segments)
        table = Table(show_header=False, box=None)
        table.add_row("Words", f"{words:,}")
        table.add_row("Segments", f"{len(segments)} (about {p.planner.segment_words} words each)")
        table.add_row("Estimated panels", f"~{est}")
        self.console.print(Panel(table, title=f"Chapter {number}: {file_path.name}"))

        how = self.select("Go?", [Choice("Storyboard and draw it", "all"),
                                  Choice("Storyboard only (review / edit plan.json before drawing)", "plan"),
                                  Choice("Cancel", "cancel")])
        if how == "cancel":
            return
        redo = any(c.number == number for c in p.chapters)
        started = time.time()
        self.release_image_model()
        plan_step(self.pdir, p, file_path, number, log=self.log)  # the LLM is unloaded again when this returns
        self.show_coverage(number)
        if how == "plan":
            self.console.print(f"Storyboard saved to [bold]{self.pdir.chapter_dir(number) / 'plan.json'}[/]. "
                               "Use 'Draw / redraw a chapter' when you're ready.")
            return
        render_step(self.pdir, p, number, force=redo, backend=self.backend(), log=self.log)
        self.console.print(f"[green]Done in {time.time() - started:.0f}s.[/]")
        self.show_result(number)

    def add_chapters(self) -> None:
        """Storyboard several chapters with one LLM session, then draw them all with one image-model session."""
        p = self.project
        if not self.planner_ready():
            return
        folder = Path(self.path("Folder with the chapter .txt files:", str(self.pdir.root), only_directories=True,
                                validate=lambda v: Path(v).expanduser().is_dir() or "Folder not found")).expanduser()
        files = sorted(folder.glob("*.txt"), key=lambda f: [int(t) if t.isdigit() else t.lower()
                                                            for t in re.split(r"(\d+)", f.name)])
        if not files:
            self.console.print("No .txt files in that folder.")
            return
        done = {Path(c.source_file).resolve() for c in p.chapters}
        picked = self._ask(questionary.checkbox(
            "Chapters to add, in order (space toggles, enter confirms):",
            choices=[Choice(f.name, f, checked=f.resolve() not in done) for f in files], **self.prompt_kwargs))
        if not picked:
            return
        first = self.number("Number of the first one:", max((c.number for c in p.chapters), default=0) + 1)
        numbered = list(zip(range(first, first + len(picked)), picked))
        table = Table("Chapter", "File", "Words", box=None)
        texts = {}
        for number, f in numbered:
            try:
                texts[f] = read_chapter(f)[0]
            except ChapterFileError as exc:
                self.console.print(Panel(escape(str(exc)), title="Can't read a chapter file", border_style="red"))
                return
            table.add_row(str(number), f.name, f"{len(texts[f].split()):,}")
        self.console.print(table)
        how = self.select("Go?", [Choice("Storyboard all, then draw all", "all"),
                                  Choice("Storyboard all only (review before drawing)", "plan"),
                                  Choice("Cancel", "cancel")])
        if how == "cancel":
            return
        redo = {n for n, _ in numbered if any(c.number == n for c in p.chapters)}
        started = time.time()
        self.release_image_model()
        self.console.rule("Phase 1: storyboarding")
        with PlannerSession(p, self.log) as session:  # the LLM stays loaded across all chapters
            for number, f in numbered:
                plan_step(self.pdir, p, f, number, log=self.log, session=session)
                self.show_coverage(number)
        if how == "plan":
            self.console.print(f"[green]Storyboarded {len(numbered)} chapter(s) in {time.time() - started:.0f}s.[/] "
                               "Review them, then use 'Draw / redraw a chapter'.")
            return
        self.console.rule("Phase 2: drawing")
        for number, _ in numbered:
            render_step(self.pdir, p, number, force=number in redo, backend=self.backend(), log=self.log)
        self.console.print(f"[green]Done: {len(numbered)} chapter(s) in {time.time() - started:.0f}s.[/]")
        for number, _ in numbered:
            self.show_result(number)

    def pick_chapter(self, message: str) -> int:
        return self.select(message, [Choice(f"{c.number}. {c.title}", c.number) for c in self.project.chapters]
                           + [Choice("Back", 0)]) or self._back()

    @staticmethod
    def _back():
        raise Back()

    def render_chapter(self) -> None:
        number = self.pick_chapter("Which chapter?")
        plan = self.pdir.load_plan(number)
        art = self.pdir.chapter_dir(number) / "art"
        drawn = sum(1 for i in range(len(plan.panels)) if (art / f"panel_{i + 1:03d}.png").exists())
        self.console.print(f"{drawn}/{len(plan.panels)} panels drawn.")
        mode = self.select("Draw what?", [
            Choice("Missing panels, and panels whose prompt changed (e.g. after switching lettering)", "missing"),
            Choice("Specific panels (e.g. 3, 7-9)", "some"),
            Choice("Everything again", "all"),
            Choice("Only redo lettering / strip (no image generation)", "letter"),
            Choice("Back", "back"),
        ])
        if mode == "back":
            return
        if mode == "letter":
            assemble_step(self.pdir, self.project, number, log=self.log)
        elif mode == "some":
            spec = self.text("Panel numbers:", validate=lambda v: self._valid_panels(v, len(plan.panels)))
            render_step(self.pdir, self.project, number, only=parse_panel_list(spec), force=True,
                        backend=self.backend(), log=self.log)
        else:
            render_step(self.pdir, self.project, number, force=mode == "all", backend=self.backend(), log=self.log,
                        redraw_changed=True)
        self.show_result(number)

    @staticmethod
    def _valid_panels(value: str, total: int):
        try:
            nums = parse_panel_list(value)
        except ValueError:
            return "Use numbers and ranges like 3, 7-9"
        if not nums:
            return "Enter at least one panel"
        bad = [n for n in nums if not 1 <= n <= total]
        return f"No such panel(s): {bad} (1-{total})" if bad else True

    def review_chapter(self) -> None:
        number = self.pick_chapter("Review which chapter?")
        plan = self.pdir.load_plan(number)
        table = Table(title=f"Chapter {number}: {plan.title}", show_lines=False, expand=True)
        for col, kw in (("#", {"justify": "right"}), ("Shot", {}), ("Characters", {}), ("Action", {"ratio": 3}),
                        ("Text", {"ratio": 3}), ("¶", {})):
            table.add_column(col, **kw)
        for i, panel in enumerate(plan.panels, 1):
            text = " / ".join([f"[i]{escape(panel.narration)}[/]"] * bool(panel.narration) +
                              [escape(f"{d.speaker}: {d.text}") for d in panel.dialogue])
            paras = ",".join(map(str, panel.source_paragraphs))
            table.add_row(str(i), panel.shot, escape(", ".join(panel.characters)), escape(panel.action[:140]),
                          text[:220], paras)
        self.console.print(table)
        self.show_coverage(number)
        self.console.print(f"Edit [bold]{self.pdir.chapter_dir(number) / 'plan.json'}[/] to change the storyboard, "
                           "then redraw the affected panels.")

    def show_coverage(self, number: int) -> None:
        path = self.pdir.chapter_dir(number) / "coverage.json"
        if not path.exists():
            return
        report = json.loads(path.read_text(encoding="utf-8"))
        segs = report.get("segments", [])
        retried = sum(1 for s in segs if s.get("attempts", 1) > 1)
        msg = f"Coverage: {len(segs)} segment(s), every paragraph mapped to a panel"
        if retried:
            msg += f"; {retried} segment(s) needed a re-plan"
        self.console.print(f"[green]{msg}.[/]")
        if report.get("patched"):
            self.console.print(f"[yellow]{report['patched']} passage(s) the planner kept skipping were added as "
                               f"caption panels - see {path}[/]")

    def open_reader(self) -> None:
        self.open_in_browser(self.pick_chapter("Open which chapter?"))

    def show_result(self, number: int) -> None:
        """Say where the chapter is; opening it is left to the 'Open a chapter in the browser' menu item."""
        cdir = self.pdir.chapter_dir(number)
        self.console.print(f"Chapter {number}: [bold]{escape(str(cdir / 'reader.html'))}[/]\n"
                           f"  [dim]panels/ = lettered panels · strip/ = stitched strip slices · "
                           f"art/ = raw images without text[/]")

    def open_in_browser(self, number: int) -> None:
        reader = self.pdir.chapter_dir(number) / "reader.html"
        self.console.print(f"Opening [bold]{escape(str(reader))}[/]")
        if reader.exists():
            webbrowser.open(reader.resolve().as_uri())

    # ------------------------------------------------------------------ characters
    def characters_menu(self) -> None:
        while True:
            chars = self.project.characters
            if not chars:
                self.console.print("No characters yet - they are created when you add a chapter.")
                return
            table = Table(expand=True)
            for col in ("Name", "Since", "Appearance", "Outfit", "Sheet"):
                table.add_column(col)
            for c in chars:
                table.add_row(escape(c.name) + (f"\n[dim]aka {escape(', '.join(c.aliases))}[/]" if c.aliases else ""),
                              f"ch.{c.first_chapter}", escape(c.appearance), escape(c.outfit), c.reference_image or "-")
            self.console.print(table)
            name = self.select("Edit a character?", [Choice(c.name, c.name) for c in chars] + [Choice("Back", "")])
            if not name:
                return
            self.edit_character(next(c for c in chars if c.name == name))

    def edit_character(self, c) -> None:
        what = self.select(f"{c.name}:", [
            Choice("Edit appearance (face, hair, body - locked across chapters)", "appearance"),
            Choice("Edit current outfit", "outfit"),
            Choice("Edit aliases / nicknames", "aliases"),
            Choice("Redraw character sheet", "sheet"),
            Choice("Back", "back"),
        ])
        if what == "back":
            return
        if what == "appearance":
            c.appearance = self.text("Appearance (Esc then Enter to finish):", c.appearance, multiline=True).strip()
            c.reference_image = None
        elif what == "outfit":
            c.outfit = self.text("Outfit:", c.outfit).strip()
        elif what == "aliases":
            c.aliases = [a.strip() for a in self.text("Aliases (comma separated):", ", ".join(c.aliases)).split(",")
                         if a.strip()]
        elif what == "sheet":
            c.reference_image = None
        self.save()
        if c.reference_image is None and what != "aliases" and self.confirm("Draw the new character sheet now?"):
            ensure_reference_sheets(self.pdir, self.project, [c.name], self.backend(), log=self.log)
            self.console.print(f"Sheet: {self.pdir.root / c.reference_image}")
        if what in ("appearance", "outfit"):
            self.console.print("[dim]Redraw the panels this character appears in to apply the change.[/]")

    # ------------------------------------------------------------------ tests
    def test_image(self) -> None:
        project = self.project
        out = self.pdir.root / "test_image.png"
        started = time.time()
        backend = self.backend()
        sample = project.characters[0] if project.characters else None
        prompt = (reference_sheet_prompt(project, sample) if sample else
                  f"{project.style}. A young woman standing in a rainy neon-lit street at night, holding a glowing lantern.")
        with self.console.status("Generating a test image ..."):
            w, h = portrait_size(project.image.megapixels)
            backend.generate(prompt, project.negative_prompt, w, h, 1234).save(out)
        self.console.print(f"[green]Image model works[/] ({time.time() - started:.0f}s): {out}")

    def test_planner(self) -> None:
        sample = ("Chapter 1 - Test\n\nAnna opened the door of the bakery. \"We're closed,\" said the old baker, "
                  "not looking up from the dough.\n\n\"I'm not here for bread,\" Anna said, and placed a key on the counter.")
        scratch = copy.deepcopy(self.project)
        scratch.characters, scratch.chapters = [], []
        started = time.time()
        self.release_image_model()
        with self.console.status(f"Asking {scratch.planner.provider}: {scratch.planner.model} ..."):
            plan, report = plan_chapter(scratch, sample, 1, log=self.log)
        self.console.print(f"[green]Planner works[/] ({time.time() - started:.0f}s): {len(plan.panels)} panels, "
                           f"characters: {', '.join(c.name for c in plan.new_characters) or '-'}")
        for i, panel in enumerate(plan.panels, 1):
            self.console.print(f"  {i}. [{panel.shot}] {panel.action[:100]}", markup=False, highlight=False)


def run(project_path: Optional[str] = None) -> None:
    App(project_path).run()

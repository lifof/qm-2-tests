"""Interactive terminal app: `python -m webtoon` (or `webtoon` after `pip install -e .`).

Menus to create/open a project, point it at your Qwen-Image model and planner,
add chapters, redraw panels, review storyboards and edit characters.
"""

from __future__ import annotations

import copy
import json
import os
import time
import webbrowser
from pathlib import Path
from typing import Any, Callable, List, Optional

import questionary
from questionary import Choice, Separator
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from .image_backends import PORTRAIT_SIZE, make_backend
from .models import Project
from .pipeline import assemble_step, ensure_reference_sheets, plan_step, render_step
from .planner import plan_chapter
from .project import ProjectDir
from .prompts import reference_sheet_prompt
from .segment import split_segments, target_panels

CONFIG_PATH = Path(os.environ.get("WEBTOON_CONFIG", Path.home() / ".config" / "webtoon" / "app.json"))

IMAGE_BACKENDS = {
    "diffusers": "Local weights (runs Qwen-Image on this machine's GPU with diffusers)",
    "openai": "My own server with an OpenAI-style /v1/images API (vLLM-Omni, LocalAI, custom...)",
    "dashscope": "Alibaba Cloud Model Studio (DashScope API)",
    "mock": "Placeholder images (test the pipeline without a GPU)",
}
PLANNERS = {
    "anthropic": "Claude (Anthropic API)",
    "openai": "OpenAI-compatible chat server (e.g. a local Qwen LLM on vLLM / Ollama / LM Studio)",
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
        return self._ask(questionary.path(message, default=default or "", only_directories=only_directories,
                                          validate=validate, **self.prompt_kwargs))

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
                self.show_header()
                try:
                    if not self.main_menu():
                        break
                except (Back, KeyboardInterrupt):
                    self.console.print("[dim]cancelled[/]")
                except Exception as exc:  # keep the app alive on errors in one action
                    self.console.print(Panel(f"{type(exc).__name__}: {exc}", title="Error", border_style="red"))
        except (Back, KeyboardInterrupt, EOFError):
            pass
        self.console.print("Bye!")

    def main_menu(self) -> bool:
        p = self.project
        next_n = max((c.number for c in p.chapters), default=0) + 1
        choices = [Choice(f"Add chapter {next_n}", "add")]
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
            "add": self.add_chapter, "render": self.render_chapter, "review": self.review_chapter,
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
        rows.add_row("Image model", f"{img.backend}: {img.model}" + (f"  @ {img.base_url}" if img.base_url else ""))
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
        if p.image.backend == "dashscope" and not os.environ.get("DASHSCOPE_API_KEY"):
            w.append("DASHSCOPE_API_KEY is not set")
        if p.planner.provider == "anthropic" and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            w.append("ANTHROPIC_API_KEY is not set (fine if you used `ant auth login`)")
        if p.planner.provider == "openai" and not p.planner.base_url:
            w.append("planner server URL is not set")
        return w

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
        if img.backend == "diffusers":
            model = self.path("Model folder (or a Hugging Face repo id):", img.model)
            img.model = str(Path(model).expanduser()) if Path(model).expanduser().exists() else model.strip()
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
        elif provider == "openai":
            pl.base_url = self.text("Chat server base URL:", pl.base_url or "http://localhost:11434/v1",
                                    validate=lambda v: v.startswith("http") or "Must start with http").strip()
            pl.model = self.text("Model name on the server:", pl.model or "qwen3:32b",
                                 validate=lambda v: bool(v.strip()) or "Required").strip()
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
                Choice(f"Backend, model path / URL: {img.backend}: {short(img.model, 60)}", "image"),
                Choice(f"Sampling steps: {img.steps}", "steps"),
                Choice(f"CFG scale: {img.cfg}", "cfg"),
                Choice(f"Pass character sheets as references: {'yes' if img.use_references else 'no'}", "refs"),
                Separator("-- Planner --"),
                Choice(f"Planner: {pl.provider}: {pl.model}" + (f" @ {pl.base_url}" if pl.base_url else ""), "planner"),
                Choice(f"Segment size for long chapters: {pl.segment_words} words", "segment"),
                Choice(f"Pacing: {pl.panels_per_1000_words:g} panels per 1000 words", "density"),
                Choice(f"Re-plan attempts when text is skipped: {pl.max_retries}", "retries"),
                Separator("-- Look --"),
                Choice(f"Art style: {short(p.style)}", "style"),
                Choice(f"Negative prompt: {short(p.negative_prompt)}", "negative"),
                Choice(f"Strip width: {p.width}px", "width"),
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
            elif what == "steps":
                img.steps = self.number("Sampling steps:", img.steps)
            elif what == "cfg":
                img.cfg = self.number("CFG scale (true_cfg_scale):", img.cfg, float)
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
        key = self.project.image.model_dump_json()
        if self._backend is None or key != self._backend_key:
            with self.console.status(f"Loading {self.project.image.backend}: {self.project.image.model} ..."):
                self._backend = make_backend(self.project.image)
            self._backend_key = key
        return self._backend

    def log(self, message: str) -> None:
        self.console.print(message, markup=False, highlight=False)

    # ------------------------------------------------------------------ chapters
    def add_chapter(self) -> None:
        p = self.project
        next_n = max((c.number for c in p.chapters), default=0) + 1
        file = self.path("Chapter text file:", validate=lambda v: Path(v).expanduser().is_file() or "File not found")
        file_path = Path(file).expanduser().resolve()
        text = file_path.read_text(encoding="utf-8")
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
        plan_step(self.pdir, p, file_path, number, log=self.log)
        self.show_coverage(number)
        if how == "plan":
            self.console.print(f"Storyboard saved to [bold]{self.pdir.chapter_dir(number) / 'plan.json'}[/]. "
                               "Use 'Draw / redraw a chapter' when you're ready.")
            return
        render_step(self.pdir, p, number, force=redo, backend=self.backend(), log=self.log)
        self.console.print(f"[green]Done in {time.time() - started:.0f}s.[/]")
        self.offer_open(number)

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
            Choice("Missing panels only", "missing"),
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
            render_step(self.pdir, self.project, number, force=mode == "all", backend=self.backend(), log=self.log)
        self.offer_open(number)

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
        self.offer_open(self.pick_chapter("Open which chapter?"), ask=False)

    def offer_open(self, number: int, ask: bool = True) -> None:
        reader = self.pdir.chapter_dir(number) / "reader.html"
        self.console.print(f"Reader: [bold]{reader}[/]")
        if reader.exists() and (not ask or self.confirm("Open it in your browser?", default=True)):
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
            w, h = PORTRAIT_SIZE
            backend.generate(prompt, project.negative_prompt, w, h, 1234).save(out)
        self.console.print(f"[green]Image model works[/] ({time.time() - started:.0f}s): {out}")
        if self.confirm("Open it?", default=True):
            webbrowser.open(out.resolve().as_uri())

    def test_planner(self) -> None:
        sample = ("Chapter 1 - Test\n\nAnna opened the door of the bakery. \"We're closed,\" said the old baker, "
                  "not looking up from the dough.\n\n\"I'm not here for bread,\" Anna said, and placed a key on the counter.")
        scratch = copy.deepcopy(self.project)
        scratch.characters, scratch.chapters = [], []
        started = time.time()
        with self.console.status(f"Asking {scratch.planner.provider}: {scratch.planner.model} ..."):
            plan, report = plan_chapter(scratch, sample, 1, log=lambda m: None)
        self.console.print(f"[green]Planner works[/] ({time.time() - started:.0f}s): {len(plan.panels)} panels, "
                           f"characters: {', '.join(c.name for c in plan.new_characters) or '-'}")
        for i, panel in enumerate(plan.panels, 1):
            self.console.print(f"  {i}. [{panel.shot}] {panel.action[:100]}", markup=False, highlight=False)


def run(project_path: Optional[str] = None) -> None:
    App(project_path).run()

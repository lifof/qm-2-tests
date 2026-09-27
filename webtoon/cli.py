"""Command-line interface.

    python -m webtoon                                     # interactive app
    python -m webtoon init   my_story --title "My Story"
    python -m webtoon chapter my_story chapter1.txt
    python -m webtoon chapter my_story chapter2.txt      # same characters carry over
    python -m webtoon render  my_story 2 --only 3,7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .models import ChapterPlan
from .llm_server import PlannerSession
from .pipeline import assemble_step, plan_step, render_step
from .project import ProjectDir


def _add_settings_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("image model (Qwen-Image)")
    g.add_argument("--image-backend", choices=["comfyui", "diffusers", "openai", "dashscope", "mock"])
    g.add_argument("--model-dir", help="Folder with the Qwen-Image 2.1 files (comfyui): the three files are detected")
    g.add_argument("--diffusion-model", help="comfyui: diffusion model file (.gguf / .safetensors)")
    g.add_argument("--text-encoder", help="comfyui: text encoder file (e.g. qwen3vl_8b_bf16.safetensors)")
    g.add_argument("--vae", help="comfyui: VAE file")
    g.add_argument("--comfy-dir", help="comfyui: local ComfyUI folder (files get linked in, server auto-started)")
    g.add_argument("--comfy-args", help='comfyui: extra ComfyUI launch arguments, e.g. "--lowvram"')
    g.add_argument("--sampler")
    g.add_argument("--scheduler")
    g.add_argument("--workflow", help="comfyui: custom API-format workflow JSON with {{placeholders}}")
    g.add_argument("--image-model", help="HF repo id / local path (diffusers) or model name (servers)")
    g.add_argument("--image-base-url", help="Server URL for the openai/dashscope backends")
    g.add_argument("--steps", type=int)
    g.add_argument("--megapixels", type=float, help="Panel resolution in megapixels (default 1.0)")
    g.add_argument("--cfg", type=float)
    g.add_argument("--use-references", action=argparse.BooleanOptionalAction,
                   help="Feed character sheets to an edit-capable model as image references")
    g = p.add_argument_group("planner (LLM that storyboards the chapter)")
    g.add_argument("--planner", choices=["anthropic", "llamacpp", "openai", "mock"])
    g.add_argument("--llm-model", help="llamacpp: the storyboard LLM's .gguf file")
    g.add_argument("--llama-server", help="llamacpp: path to the llama-server binary (default: found on PATH)")
    g.add_argument("--llm-context", type=int, help="llamacpp: context size (default 32768)")
    g.add_argument("--llm-args", help='llamacpp: extra llama-server arguments (default "--reasoning off")')
    g.add_argument("--planner-model")
    g.add_argument("--planner-base-url", help="For --planner openai: e.g. http://localhost:11434/v1")
    g.add_argument("--segment-words", type=int, help="Long chapters are storyboarded in parts of ~N words (default 1200)")
    g.add_argument("--density", type=float, help="Panels per 1000 words of text (default 10)")
    g.add_argument("--retries", type=int, help="Re-plans of a part when text was skipped (default 2)")
    g = p.add_argument_group("look")
    g.add_argument("--style", help="Art-style prompt prepended to every panel")
    g.add_argument("--negative", help="Negative prompt")
    g.add_argument("--width", type=int, help="Strip width in px (default 800)")
    g.add_argument("--font", help="Path to a .ttf used for lettering")
    g.add_argument("--lettering", choices=["app", "model"],
                   help="Who draws the text: 'app' (on top, exact) or 'model' (Qwen-Image draws it into the art)")
    g.add_argument("--model-text-max-words", type=int,
                   help="With --lettering model: longer texts are still added by the app (default 30)")


def _apply_settings(project, args) -> None:
    mapping = {
        "image_backend": (project.image, "backend"), "image_model": (project.image, "model"),
        "image_base_url": (project.image, "base_url"), "steps": (project.image, "steps"),
        "cfg": (project.image, "cfg"), "megapixels": (project.image, "megapixels"), "use_references": (project.image, "use_references"),
        "planner": (project.planner, "provider"), "planner_model": (project.planner, "model"),
        "planner_base_url": (project.planner, "base_url"), "segment_words": (project.planner, "segment_words"), "density": (project.planner, "panels_per_1000_words"),
        "retries": (project.planner, "max_retries"),
        "llm_model": (project.planner, "llm_model"), "llama_server": (project.planner, "llama_server"),
        "llm_context": (project.planner, "llm_context"), "llm_args": (project.planner, "llm_args"),
        "style": (project, "style"), "negative": (project, "negative_prompt"), "width": (project, "width"),
        "font": (project, "font"), "lettering": (project, "lettering"),
        "model_text_max_words": (project, "model_text_max_words"),
        "diffusion_model": (project.image, "diffusion_model"), "text_encoder": (project.image, "text_encoder"),
        "vae": (project.image, "vae"), "comfy_dir": (project.image, "comfy_dir"),
        "comfy_args": (project.image, "comfy_args"),
        "sampler": (project.image, "sampler"), "scheduler": (project.image, "scheduler"),
        "workflow": (project.image, "workflow_file"),
    }
    if getattr(args, "model_dir", None):
        from .comfyui import detect_model_files

        for kind, path in detect_model_files(Path(args.model_dir)).items():
            if path is not None and getattr(args, kind, None) is None:
                setattr(project.image, kind, str(path.resolve()))
    for arg, (obj, field) in mapping.items():
        value = getattr(args, arg, None)
        if value is not None:
            setattr(obj, field, value)
    if project.planner.provider == "llamacpp" and project.planner.llm_model:
        project.planner.model = Path(project.planner.llm_model).name


def _open(args, create: bool):
    pdir = ProjectDir(Path(args.project))
    if pdir.exists():
        project = pdir.load()
    elif create:
        project = pdir.create(getattr(args, "title", None) or pdir.root.name.replace("_", " ").title())
        print(f"Created project {pdir.root}")
    else:
        sys.exit(f"No project at {pdir.root} (run `init` or `chapter` first)")
    _apply_settings(project, args)
    pdir.save(project)
    return pdir, project


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="webtoon", description="Turn story chapters into webtoon strips with Qwen-Image.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="Create a project (or update its settings)")
    p.add_argument("project")
    p.add_argument("--title")
    _add_settings_args(p)

    p = sub.add_parser("chapter", help="Storyboard + render the next chapter(s)")
    p.add_argument("project")
    p.add_argument("chapter_file", type=Path, nargs="+",
                   help="One or more chapter files: all are storyboarded first (one LLM session), then all are drawn")
    p.add_argument("--number", type=int, help="Chapter number (default: next one). Reuse a number to redo it.")
    p.add_argument("--plan-only", action="store_true", help="Stop after writing plan.json so you can edit it")
    p.add_argument("--plan-file", type=Path, help="Use an existing plan JSON instead of calling the planner")
    p.add_argument("--title")
    _add_settings_args(p)

    p = sub.add_parser("render", help="(Re)render a chapter from its plan.json")
    p.add_argument("project")
    p.add_argument("number", type=int)
    p.add_argument("--only", help="Comma-separated panel numbers to (re)draw, e.g. 3,7")
    p.add_argument("--force", action="store_true", help="Redraw panels even if art already exists")
    p.add_argument("--letter-only", action="store_true", help="Only redo lettering/strip, no image generation")
    p.add_argument("--changed", action="store_true",
                   help="Also redraw panels whose prompt changed (e.g. after switching --lettering)")
    _add_settings_args(p)

    p = sub.add_parser("characters", help="Show the character bible")
    p.add_argument("project")

    p = sub.add_parser("app", help="Interactive terminal app (default when run without arguments)")
    p.add_argument("project", nargs="?")

    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        argv = ["app"]
    args = parser.parse_args(argv)

    if args.cmd == "app":
        from .tui import run

        run(args.project)
        return

    if args.cmd == "init":
        pdir, project = _open(args, create=True)
        print(f"Project '{project.title}' ready at {pdir.root}")

    elif args.cmd == "chapter":
        pdir, project = _open(args, create=True)
        if len(args.chapter_file) > 1 and (args.number or args.plan_file):
            sys.exit("--number and --plan-file work with a single chapter file")
        first = args.number or (max((c.number for c in project.chapters), default=0) + 1)
        numbers = list(range(first, first + len(args.chapter_file)))
        plan = ChapterPlan.model_validate_json(args.plan_file.read_text(encoding="utf-8")) if args.plan_file else None
        # Phase 1: storyboard everything while the LLM is loaded; phase 2: draw with the image model.
        with PlannerSession(project) as session:
            for number, chapter_file in zip(numbers, args.chapter_file):
                plan_step(pdir, project, chapter_file, number, plan=plan, session=session)
        if args.plan_only:
            for number in numbers:
                print(f"Plan written to {pdir.chapter_dir(number) / 'plan.json'}. Edit it, then run: "
                      f"python -m webtoon render {pdir.root} {number}")
            return
        for number in numbers:
            render_step(pdir, project, number, force=args.number is not None)

    elif args.cmd == "render":
        pdir, project = _open(args, create=False)
        if args.letter_only:
            assemble_step(pdir, project, args.number)
            return
        only = {int(x) for x in args.only.split(",")} if args.only else None
        render_step(pdir, project, args.number, only=only, force=args.force or only is not None,
                    redraw_changed=args.changed)

    elif args.cmd == "characters":
        pdir, project = _open(args, create=False)
        for c in project.characters:
            print(f"{c.name} (from ch.{c.first_chapter}){' aka ' + ', '.join(c.aliases) if c.aliases else ''}")
            print(f"   role:       {c.role}\n   appearance: {c.appearance}\n   outfit:     {c.outfit}")
            print(f"   sheet:      {c.reference_image or '-'}")


if __name__ == "__main__":
    main()

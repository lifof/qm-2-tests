"""Command-line interface.

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
from .pipeline import assemble_step, plan_step, render_step
from .project import ProjectDir


def _add_settings_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("image model (Qwen-Image)")
    g.add_argument("--image-backend", choices=["diffusers", "openai", "dashscope", "mock"])
    g.add_argument("--image-model", help="HF repo id / local path (diffusers) or model name (servers)")
    g.add_argument("--image-base-url", help="Server URL for the openai/dashscope backends")
    g.add_argument("--steps", type=int)
    g.add_argument("--cfg", type=float)
    g.add_argument("--use-references", action=argparse.BooleanOptionalAction,
                   help="Feed character sheets to an edit-capable model as image references")
    g = p.add_argument_group("planner (LLM that storyboards the chapter)")
    g.add_argument("--planner", choices=["anthropic", "openai"])
    g.add_argument("--planner-model")
    g.add_argument("--planner-base-url", help="For --planner openai: e.g. http://localhost:11434/v1")
    g.add_argument("--panels", dest="target_panels", help='Target panel count per chapter, e.g. "12-24"')
    g = p.add_argument_group("look")
    g.add_argument("--style", help="Art-style prompt prepended to every panel")
    g.add_argument("--negative", help="Negative prompt")
    g.add_argument("--width", type=int, help="Strip width in px (default 800)")
    g.add_argument("--font", help="Path to a .ttf used for lettering")


def _apply_settings(project, args) -> None:
    mapping = {
        "image_backend": (project.image, "backend"), "image_model": (project.image, "model"),
        "image_base_url": (project.image, "base_url"), "steps": (project.image, "steps"),
        "cfg": (project.image, "cfg"), "use_references": (project.image, "use_references"),
        "planner": (project.planner, "provider"), "planner_model": (project.planner, "model"),
        "planner_base_url": (project.planner, "base_url"), "target_panels": (project.planner, "target_panels"),
        "style": (project, "style"), "negative": (project, "negative_prompt"), "width": (project, "width"),
        "font": (project, "font"),
    }
    for arg, (obj, field) in mapping.items():
        value = getattr(args, arg, None)
        if value is not None:
            setattr(obj, field, value)


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

    p = sub.add_parser("chapter", help="Storyboard + render the next chapter")
    p.add_argument("project")
    p.add_argument("chapter_file", type=Path)
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
    _add_settings_args(p)

    p = sub.add_parser("characters", help="Show the character bible")
    p.add_argument("project")

    args = parser.parse_args(argv)

    if args.cmd == "init":
        pdir, project = _open(args, create=True)
        print(f"Project '{project.title}' ready at {pdir.root}")

    elif args.cmd == "chapter":
        pdir, project = _open(args, create=True)
        number = args.number or (max((c.number for c in project.chapters), default=0) + 1)
        plan = ChapterPlan.model_validate_json(args.plan_file.read_text(encoding="utf-8")) if args.plan_file else None
        plan_step(pdir, project, args.chapter_file, number, plan=plan)
        if args.plan_only:
            print(f"Plan written to {pdir.chapter_dir(number) / 'plan.json'}. Edit it, then run: "
                  f"python -m webtoon render {pdir.root} {number}")
            return
        render_step(pdir, project, number, force=args.number is not None)

    elif args.cmd == "render":
        pdir, project = _open(args, create=False)
        if args.letter_only:
            assemble_step(pdir, project, args.number)
            return
        only = {int(x) for x in args.only.split(",")} if args.only else None
        render_step(pdir, project, args.number, only=only, force=args.force or only is not None)

    elif args.cmd == "characters":
        pdir, project = _open(args, create=False)
        for c in project.characters:
            print(f"{c.name} (from ch.{c.first_chapter}){' aka ' + ', '.join(c.aliases) if c.aliases else ''}")
            print(f"   role:       {c.role}\n   appearance: {c.appearance}\n   outfit:     {c.outfit}")
            print(f"   sheet:      {c.reference_image or '-'}")


if __name__ == "__main__":
    main()

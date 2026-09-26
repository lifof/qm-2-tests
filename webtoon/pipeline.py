"""High-level steps: plan a chapter, render its panels, assemble the strip."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Callable, Iterable, Optional

from PIL import Image

from . import compose
from .image_backends import PORTRAIT_SIZE, ImageBackend, make_backend, size_for_shot
from .models import ChapterPlan, ChapterRecord, Project
from .planner import plan_chapter
from .project import ProjectDir, apply_plan, stable_seed
from .prompts import panel_prompt, reference_sheet_prompt

Log = Callable[[str], None]


def plan_step(pdir: ProjectDir, project: Project, chapter_file: Path, number: int, log: Log = print,
              plan: Optional[ChapterPlan] = None) -> ChapterPlan:
    if plan is None:
        log(f"Planning chapter {number} with {project.planner.provider}:{project.planner.model} ...")
        plan = plan_chapter(project, chapter_file.read_text(encoding="utf-8"), number)
    plan_path = pdir.save_plan(number, plan)
    added = apply_plan(project, plan, number)

    record = ChapterRecord(number=number, title=plan.title, summary=plan.summary, source_file=str(chapter_file),
                           plan_file=str(plan_path.relative_to(pdir.root)))
    project.chapters = sorted([c for c in project.chapters if c.number != number] + [record], key=lambda c: c.number)
    pdir.save(project)

    log(f"  '{plan.title}': {len(plan.panels)} panels")
    for c in added:
        log(f"  + new character: {c.name} - {c.appearance}")
    return plan


def ensure_reference_sheets(pdir: ProjectDir, project: Project, names: Iterable[str], backend: ImageBackend,
                            log: Log = print) -> None:
    wanted = set(names)
    for c in project.characters:
        if c.name not in wanted:
            continue
        if c.reference_image and (pdir.root / c.reference_image).exists():
            continue
        log(f"  drawing character sheet for {c.name} ...")
        w, h = PORTRAIT_SIZE
        img = backend.generate(reference_sheet_prompt(project, c), project.negative_prompt, w, h, c.seed)
        pdir.characters_dir.mkdir(parents=True, exist_ok=True)
        path = pdir.characters_dir / f"{_slug(c.name)}.png"
        img.save(path)
        c.reference_image = str(path.relative_to(pdir.root))
        pdir.save(project)


def render_step(pdir: ProjectDir, project: Project, number: int, only: Optional[set[int]] = None,
                force: bool = False, backend: Optional[ImageBackend] = None, log: Log = print) -> Path:
    plan = pdir.load_plan(number)
    apply_plan(project, plan, number)  # idempotent; picks up hand edits to plan.json
    pdir.save(project)

    cdir = pdir.chapter_dir(number)
    art_dir, lettered_dir = cdir / "art", cdir / "panels"
    art_dir.mkdir(parents=True, exist_ok=True)
    lettered_dir.mkdir(parents=True, exist_ok=True)

    todo = [
        i for i in range(len(plan.panels))
        if (only is None or i + 1 in only) and (force or not (art_dir / f"panel_{i + 1:03d}.png").exists())
    ]
    if todo and backend is None:
        log(f"Loading image backend {project.image.backend}:{project.image.model} ...")
        backend = make_backend(project.image)

    if todo:
        names = {n for i in todo for n in plan.panels[i].characters}
        ensure_reference_sheets(pdir, project, {c.name for c in project.characters if c.name in names or
                                                any(a in names for a in c.aliases)}, backend, log)

    use_refs = project.image.use_references and backend is not None and backend.supports_references
    for i in todo:
        panel = plan.panels[i]
        prompt, chars = panel_prompt(project, panel)
        refs = [pdir.root / c.reference_image for c in chars if c.reference_image] if use_refs else []
        w, h = size_for_shot(panel.shot)
        log(f"  panel {i + 1}/{len(plan.panels)} [{panel.shot}] {', '.join(c.name for c in chars) or '-'}")
        img = backend.generate(prompt, project.negative_prompt, w, h,
                               stable_seed(f"{project.title}:{number}:{i}"), refs[:3])
        img.save(art_dir / f"panel_{i + 1:03d}.png")
        (art_dir / f"panel_{i + 1:03d}.prompt.txt").write_text(prompt, encoding="utf-8")

    return assemble_step(pdir, project, number, plan, log)


def assemble_step(pdir: ProjectDir, project: Project, number: int, plan: Optional[ChapterPlan] = None,
                  log: Log = print) -> Path:
    plan = plan or pdir.load_plan(number)
    cdir = pdir.chapter_dir(number)
    lettered, plans = [], []
    for i, panel in enumerate(plan.panels):
        art_path = cdir / "art" / f"panel_{i + 1:03d}.png"
        if not art_path.exists():
            log(f"  (panel {i + 1} has no art yet, skipping)")
            continue
        img = compose.letter_panel(Image.open(art_path), panel, project.width, project.font)
        img.save(cdir / "panels" / f"panel_{i + 1:03d}.png")
        lettered.append(img)
        plans.append(panel)
    if not lettered:
        raise RuntimeError(f"Chapter {number} has no rendered panels")

    strip = compose.build_strip(lettered, plans, project.width)
    strip_dir = cdir / "strip"
    for old in strip_dir.glob("*.jpg"):
        old.unlink()
    slices = compose.slice_strip(strip, strip_dir, f"ch{number:02d}")
    if strip.height < 65000:
        strip.save(cdir / f"chapter_{number:02d}_full.png")
    reader = cdir / "reader.html"
    compose.write_reader(reader, f"{project.title} - Chapter {number}: {plan.title}", slices, project.width)

    for rec in project.chapters:
        if rec.number == number:
            rec.panel_images = [str(p.relative_to(pdir.root)) for p in sorted((cdir / "panels").glob("*.png"))]
            rec.strip_images = [str(p.relative_to(pdir.root)) for p in slices]
    pdir.save(project)
    write_index(pdir, project)
    log(f"Chapter {number} done -> {reader}")
    return reader


def write_index(pdir: ProjectDir, project: Project) -> None:
    items = "\n".join(
        f'<li><a href="chapter_{c.number:02d}/reader.html">Chapter {c.number}: {html.escape(c.title)}</a></li>'
        for c in project.chapters
    )
    chars = "\n".join(
        f'<figure><img src="{html.escape(c.reference_image)}" alt=""><figcaption><b>{html.escape(c.name)}</b>'
        f"<br>{html.escape(c.role)}</figcaption></figure>"
        for c in project.characters if c.reference_image
    )
    (pdir.root / "index.html").write_text(
        f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(project.title)}</title>
<style>body{{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 16px}}
.cast{{display:flex;flex-wrap:wrap;gap:12px}} figure{{width:160px;margin:0}} figure img{{width:100%;border-radius:8px}}</style>
</head><body><h1>{html.escape(project.title)}</h1><h2>Chapters</h2><ul>{items}</ul>
<h2>Cast</h2><div class="cast">{chars}</div></body></html>
""",
        encoding="utf-8",
    )


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_") or "character"

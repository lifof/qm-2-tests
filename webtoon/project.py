"""Loading, saving and updating the on-disk project (story.json + images)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional

from .models import Character, ChapterPlan, Project

STATE_FILE = "story.json"

DEFAULT_STYLE = (
    "Korean webtoon style digital illustration, clean confident line art, cel shading with soft gradients, "
    "vibrant but cohesive colour palette, expressive anime-influenced faces, detailed backgrounds, "
    "cinematic lighting, high quality"
)
DEFAULT_NEGATIVE = (
    "text, letters, words, watermark, signature, speech bubble, caption, panel border, frame, collage, "
    "multiple panels, split screen, blurry, lowres, deformed hands, extra fingers, extra limbs, bad anatomy, "
    "photorealistic, 3d render"
)


class ProjectDir:
    def __init__(self, root: Path):
        self.root = Path(root)

    # -- paths ---------------------------------------------------------------
    @property
    def state_path(self) -> Path:
        return self.root / STATE_FILE

    def chapter_dir(self, number: int) -> Path:
        return self.root / f"chapter_{number:02d}"

    @property
    def characters_dir(self) -> Path:
        return self.root / "characters"

    # -- persistence ---------------------------------------------------------
    def exists(self) -> bool:
        return self.state_path.exists()

    def load(self) -> Project:
        return Project.model_validate_json(self.state_path.read_text(encoding="utf-8"))

    def save(self, project: Project) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(project.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self.state_path)

    def create(self, title: str, style: Optional[str] = None, negative: Optional[str] = None) -> Project:
        project = Project(title=title, style=style or DEFAULT_STYLE, negative_prompt=negative or DEFAULT_NEGATIVE)
        self.save(project)
        return project

    def save_plan(self, number: int, plan: ChapterPlan) -> Path:
        path = self.chapter_dir(number) / "plan.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load_plan(self, number: int) -> ChapterPlan:
        return ChapterPlan.model_validate_json((self.chapter_dir(number) / "plan.json").read_text(encoding="utf-8"))


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def character_index(project: Project) -> Dict[str, Character]:
    """Map every lower-cased name and alias to its character."""
    index: Dict[str, Character] = {}
    for c in project.characters:
        index[c.name.lower()] = c
        for alias in c.aliases:
            index.setdefault(alias.lower(), c)
    return index


def apply_plan(project: Project, plan: ChapterPlan, chapter_number: int) -> list[Character]:
    """Merge new characters / updates from a plan into the character bible.

    Existing characters keep their locked appearance and seed so they look the same
    across chapters; only outfits and explicit permanent changes are applied.
    Returns the list of characters that were newly added.
    """
    index = character_index(project)
    added: list[Character] = []
    for spec in plan.new_characters:
        existing = index.get(spec.name.lower())
        if existing is not None and existing.first_chapter == chapter_number:
            # Re-applying this chapter's own plan (e.g. after hand-editing plan.json):
            # take the edited design, and drop a reference sheet that no longer matches.
            if (existing.appearance, existing.outfit) != (spec.appearance, spec.outfit):
                existing.reference_image = None
            existing.aliases, existing.role = spec.aliases, spec.role
            existing.appearance, existing.outfit = spec.appearance, spec.outfit
            existing.age = spec.age or existing.age
            continue
        if existing is not None:
            # Planner re-introduced someone from an earlier chapter: keep the locked look.
            for alias in spec.aliases:
                if alias not in existing.aliases and alias.lower() != existing.name.lower():
                    existing.aliases.append(alias)
            if not existing.age and spec.age.strip():
                existing.age = spec.age.strip()  # older bibles had no ages
            continue
        char = Character(
            name=spec.name,
            aliases=spec.aliases,
            role=spec.role,
            age=spec.age,
            appearance=spec.appearance,
            outfit=spec.outfit,
            seed=stable_seed(f"{project.title}:{spec.name}"),
            first_chapter=chapter_number,
        )
        project.characters.append(char)
        added.append(char)
        index = character_index(project)

    for upd in plan.character_updates:
        char = index.get(upd.name.lower())
        if char is None:
            continue
        if upd.outfit.strip():
            char.outfit = upd.outfit.strip()
        if upd.age.strip():
            char.age = upd.age.strip()
        change = upd.appearance_change.strip()
        if change and change not in char.appearance:
            char.appearance = f"{char.appearance}; {change}"
            char.reference_image = None
    return added


def bible_for_prompt(project: Project) -> str:
    if not project.characters:
        return "(empty - this is the first chapter)"
    return json.dumps(
        [
            {"name": c.name, "aliases": c.aliases, "role": c.role, "age": c.age or "(unknown - set it via character_updates)",
             "appearance": c.appearance, "current_outfit": c.outfit}
            for c in project.characters
        ],
        indent=2,
        ensure_ascii=False,
    )

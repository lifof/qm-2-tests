"""Builds image prompts from panel plans + the character bible."""

from __future__ import annotations

from typing import List, Tuple

from .models import Character, PanelPlan, Project
from .project import character_index

SHOT_PHRASES = {
    "establishing": "establishing shot, wide view of the location, characters small in frame",
    "wide": "wide shot, full bodies visible, environment clearly shown",
    "medium": "medium shot, framed from the waist up",
    "close-up": "close-up shot of the face and shoulders, detailed expression",
    "extreme-close-up": "extreme close-up, tightly framed on the eyes or a key detail",
    "over-the-shoulder": "over-the-shoulder shot",
    "action": "dynamic action shot, dramatic angle, motion lines, strong perspective",
}


def describe_character(c: Character) -> str:
    desc = f"{c.name}: {c.appearance}"
    if c.outfit:
        desc += f", wearing {c.outfit}"
    return desc


def panel_characters(project: Project, panel: PanelPlan) -> List[Character]:
    index = character_index(project)
    seen, result = set(), []
    for name in panel.characters:
        c = index.get(name.lower())
        if c is not None and c.name not in seen:
            seen.add(c.name)
            result.append(c)
    return result


def panel_prompt(project: Project, panel: PanelPlan) -> Tuple[str, List[Character]]:
    chars = panel_characters(project, panel)
    parts = [
        project.style,
        SHOT_PHRASES.get(panel.shot, panel.shot),
        f"Setting: {panel.location}, {panel.time_of_day}",
    ]
    if chars:
        count = {1: "one person", 2: "two people", 3: "three people"}.get(len(chars), f"{len(chars)} people")
        parts.append(f"The image shows exactly {count}. " + " | ".join(describe_character(c) for c in chars))
    parts.append(f"Scene: {panel.action}")
    parts.append(f"Mood and lighting: {panel.mood}")
    parts.append("A single full-bleed illustration with no text, no speech bubbles, no panel borders")
    return ". ".join(p.strip().rstrip(".") for p in parts if p.strip()) + ".", chars


def reference_sheet_prompt(project: Project, c: Character) -> str:
    return (
        f"{project.style}. Character reference sheet of a single character, full body, standing, front view, "
        f"neutral pose, neutral expression, plain light grey background, even studio lighting. "
        f"{describe_character(c)}. No text."
    )

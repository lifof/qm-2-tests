"""Builds image prompts from panel plans + the character bible."""

from __future__ import annotations

from typing import List, Optional, Tuple

from .models import Character, PanelPlan, Project
from .project import character_index
from .safety import PanelSafety, age_phrase, sheet_outfit

SHOT_PHRASES = {
    "establishing": "establishing shot, wide view of the location, characters small in frame",
    "wide": "wide shot, full bodies visible, environment clearly shown",
    "medium": "medium shot, framed from the waist up",
    "close-up": "close-up shot of the face and shoulders, detailed expression",
    "extreme-close-up": "extreme close-up, tightly framed on the eyes or a key detail",
    "over-the-shoulder": "over-the-shoulder shot",
    "action": "dynamic action shot, dramatic angle, motion lines, strong perspective",
}


def describe_character(c: Character, outfit: Optional[str] = None) -> str:
    outfit = c.outfit if outfit is None else outfit
    age = age_phrase(c)
    desc = f"{c.name}: {age + ', ' if age else ''}{c.appearance}"
    if outfit:
        desc += ", " + (outfit if outfit.startswith("wearing") else f"wearing {outfit}")
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


# Present in a prompt when the image model was asked to letter the panel itself; assemble_step
# uses it to know which texts are already in the art.
MODEL_TEXT_MARKER = "spelled exactly as quoted"
TEXT_NEGATIVE_TERMS = {"text", "letters", "words", "speech bubble", "caption", "captions", "speech bubbles"}


def _words(text: str) -> int:
    return len(text.split())


def split_lettering(project: Project, panel: PanelPlan) -> Tuple[PanelPlan, PanelPlan]:
    """(texts for the image model to draw, texts for the app to letter on top).

    With lettering="app" everything is lettered by the app. With "model", short texts go
    into the image prompt; long ones (and anything past the per-panel budget) stay with
    the app, because image models get unreliable with long passages.
    """
    empty = {"narration": "", "dialogue": [], "sfx": ""}
    if project.lettering != "model":
        return panel.model_copy(update=empty), panel
    limit = project.model_text_max_words
    budget = limit * 2
    to_model = {"narration": "", "dialogue": [], "sfx": panel.sfx}
    to_app = {"narration": "", "dialogue": [], "sfx": ""}

    def fits(text: str, extra_ok: bool = True) -> bool:
        nonlocal budget
        n = _words(text)
        if n <= limit and n <= budget and extra_ok:
            budget -= n
            return True
        return False

    narration = panel.narration.strip()
    if narration:
        (to_model if fits(narration) else to_app)["narration"] = narration
    for line in panel.dialogue:
        ok = fits(line.text, line.kind != "system" or len(line.text.strip().splitlines()) <= 8)
        (to_model if ok else to_app)["dialogue"].append(line)
    return panel.model_copy(update=to_model), panel.model_copy(update=to_app)


def _q(text: str) -> str:
    """Quote text for the prompt (Qwen-Image renders what's inside double quotes)."""
    return '"' + " ".join(text.replace('"', "'").split()) + '"'


def text_phrases(panel: PanelPlan, visible: List[str]) -> List[str]:
    phrases = []
    if panel.narration.strip():
        phrases.append(f"a rectangular comic narration caption box in the top-left corner with the text {_q(panel.narration)}")
    for line in panel.dialogue:
        who = (f"with its tail pointing to {line.speaker}" if line.speaker in visible
               else "coming from off-panel")
        if line.kind == "system":
            lines = [l.strip() for l in line.text.splitlines() if l.strip()]
            phrases.append("a glowing translucent blue holographic game interface window floating in the air, "
                           "displaying the lines " + ", ".join(_q(l) for l in lines))
        elif line.kind == "thought":
            phrases.append(f"a cloud-shaped thought bubble {who} with the text {_q(line.text)}")
        elif line.kind == "shout":
            phrases.append(f"a spiky jagged shout bubble {who} with the bold capital-letter text {_q(line.text.upper())}")
        elif line.kind == "whisper":
            phrases.append(f"a dashed-outline whisper bubble {who} with small text {_q(line.text)}")
        else:
            phrases.append(f"a white comic speech bubble {who} with the text {_q(line.text)}")
    if panel.sfx.strip():
        phrases.append(f"large bold stylised comic sound-effect lettering {_q(panel.sfx.upper())} drawn into the scene")
    return phrases


def negative_prompt_for(project: Project, prompt: str) -> str:
    """Drop 'no text' terms from the negative prompt when the model is asked to draw text."""
    if MODEL_TEXT_MARKER not in prompt:
        return project.negative_prompt
    terms = [t.strip() for t in project.negative_prompt.split(",")]
    return ", ".join(t for t in terms if t and t.lower() not in TEXT_NEGATIVE_TERMS)


def panel_prompt(project: Project, panel: PanelPlan,
                 model_text: Optional[PanelPlan] = None) -> Tuple[str, List[Character]]:
    chars = panel_characters(project, panel)
    split = split_lettering(project, panel)
    if model_text is None:
        model_text = split[0]
    app_text = split[1]
    parts = [
        project.style,
        SHOT_PHRASES.get(panel.shot, panel.shot),
        f"Setting: {panel.location}, {panel.time_of_day}",
    ]
    safety = PanelSafety(panel.action, chars)
    if chars:
        count = {1: "one person", 2: "two people", 3: "three people"}.get(len(chars), f"{len(chars)} people")
        parts.append(f"The image shows exactly {count}. " +
                     " | ".join(describe_character(c, safety.outfit(c, project.nudity_cover)) for c in chars))
    parts.append(f"Scene: {safety.action(panel.action)}")
    if safety.cover:
        parts.append("Everyone in the image is fully and modestly clothed; tasteful, non-explicit, all-ages image")
    parts.append(f"Mood and lighting: {panel.mood}")
    phrases = text_phrases(model_text, [c.name for c in chars])
    if phrases:
        parts.append("Comic lettering in the panel: " + "; ".join(phrases))
        parts.append(f"All lettering is clean, legible English comic lettering, {MODEL_TEXT_MARKER}, "
                     "and there is no other text. A single full-bleed illustration with no panel borders")
    else:
        parts.append("A single full-bleed illustration with no text, no speech bubbles, no panel borders")
    if app_text.narration.strip() or any(d.kind != "system" for d in app_text.dialogue):
        parts.append("The composition leaves calm, uncluttered background space near the top of the frame")
    if any(d.kind == "system" for d in app_text.dialogue):
        parts.append("Any floating holographic screen in the scene is completely blank, with no writing or symbols")
    return ". ".join(p.strip().rstrip(".") for p in parts if p.strip()) + ".", chars


def reference_sheet_prompt(project: Project, c: Character) -> str:
    return (
        f"{project.style}. Character reference sheet of a single character, full body, standing, front view, "
        f"neutral pose, neutral expression, plain light grey background, even studio lighting. "
        f"{describe_character(c, sheet_outfit(c, project.nudity_cover))}. Fully and modestly clothed. No text."
    )

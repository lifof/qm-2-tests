"""Turns chapter text into a panel-by-panel plan using an LLM.

The planner is given the persistent character bible and the story-so-far, so it
reuses the same canonical characters in every chapter instead of inventing new ones.
"""

from __future__ import annotations

import json
import os

from pydantic import ValidationError

from .models import ChapterPlan, PlannerSettings, Project
from .project import bible_for_prompt

SYSTEM_PROMPT = """You are a webtoon storyboard artist adapting prose chapters into vertical-scroll webtoon panels.

You receive: the CHARACTER BIBLE (characters already established in earlier chapters), the STORY SO FAR, and the text of the NEW CHAPTER.

Rules:
- Every character already in the bible MUST be referred to by their exact canonical `name`, even if the chapter uses a nickname or pronoun. Never re-describe or redesign them; put them in `new_characters` only if they are genuinely absent from the bible.
- For a new character, write `appearance` as a precise, permanent, purely visual description an illustrator could reproduce identically every time (sex, apparent age, build, skin tone, face, eyes, hair colour/length/style, distinguishing marks). If the text gives few details, invent specific, distinctive ones that fit the story, and make each character visually distinct from the others.
- Only list a character in `character_updates` when their outfit or permanent look actually changes in this chapter.
- Adapt the whole chapter in order. Pace it like a webtoon: establishing shots when the location changes, close-ups for emotional beats, action shots for movement. Vary shots.
- Each panel is one single image: at most 3 characters visible, one moment in time. `action` must be concrete and visual (poses, expressions, where people are, key props) because it becomes an image prompt.
- Keep dialogue short and punchy (webtoon bubbles, max ~20 words each, max 3 bubbles per panel). Use narration boxes sparingly. Dialogue speakers must be canonical names; a speaker does not need to be visible in the panel.
- Keep locations consistent: describe a recurring location the same way each time.
"""


def _user_prompt(project: Project, chapter_text: str, chapter_number: int) -> str:
    story_so_far = "\n".join(f"Chapter {c.number} - {c.title}: {c.summary}" for c in project.chapters) or "(none)"
    return (
        f"<character_bible>\n{bible_for_prompt(project)}\n</character_bible>\n\n"
        f"<story_so_far>\n{story_so_far}\n</story_so_far>\n\n"
        f"<new_chapter number=\"{chapter_number}\">\n{chapter_text}\n</new_chapter>\n\n"
        f"Storyboard chapter {chapter_number} as roughly {project.planner.target_panels} panels."
    )


def plan_chapter(project: Project, chapter_text: str, chapter_number: int) -> ChapterPlan:
    settings = project.planner
    user = _user_prompt(project, chapter_text, chapter_number)
    if settings.provider == "anthropic":
        raw = _plan_with_claude(settings, user)
    elif settings.provider == "openai":
        raw = _plan_with_openai_compatible(settings, user)
    else:
        raise ValueError(f"Unknown planner provider: {settings.provider}")
    try:
        return ChapterPlan.model_validate_json(raw)
    except ValidationError as exc:
        raise RuntimeError(f"Planner returned JSON that doesn't match the plan schema:\n{exc}\n\n{raw[:2000]}") from exc


def _plan_with_claude(settings: PlannerSettings, user: str) -> str:
    import anthropic

    client = anthropic.Anthropic(base_url=settings.base_url) if settings.base_url else anthropic.Anthropic()
    # Streaming: long chapters can produce long plans. Server-side fallbacks keep a
    # false-positive safety decline from failing the whole run.
    with client.beta.messages.stream(
        model=settings.model,
        max_tokens=64000,
        thinking={"type": "adaptive"},
        output_config={
            "effort": "high",
            "format": {"type": "json_schema", "schema": ChapterPlan.model_json_schema()},
        },
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        raise RuntimeError(f"The planner model declined this chapter: {message.stop_details}")
    if message.stop_reason == "max_tokens":
        raise RuntimeError("Planner output was cut off (max_tokens). Try splitting the chapter or lowering --panels.")
    return "".join(block.text for block in message.content if block.type == "text")


def _plan_with_openai_compatible(settings: PlannerSettings, user: str) -> str:
    """For a self-hosted LLM (e.g. a Qwen model behind vLLM / Ollama / LM Studio)."""
    from openai import OpenAI

    client = OpenAI(base_url=settings.base_url, api_key=os.environ.get("OPENAI_API_KEY", "not-needed"))
    schema = ChapterPlan.model_json_schema()
    response = client.chat.completions.create(
        model=settings.model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT + "\nRespond with a single JSON object matching this schema:\n" + json.dumps(schema),
            },
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_schema", "json_schema": {"name": "chapter_plan", "schema": schema}},
    )
    return response.choices[0].message.content or ""

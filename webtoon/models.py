"""Data models for a webtoon project: the character bible, panel plans, and saved state."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Shot = Literal["establishing", "wide", "medium", "close-up", "extreme-close-up", "over-the-shoulder", "action"]
BubbleKind = Literal["speech", "thought", "shout", "whisper"]


class _Strict(BaseModel):
    # extra="forbid" makes the JSON schema emit additionalProperties: false,
    # which structured outputs require.
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# What the planner LLM returns for one chapter
# ---------------------------------------------------------------------------


class CharacterSpec(_Strict):
    name: str = Field(description="Canonical name, reused verbatim in every later chapter.")
    aliases: List[str] = Field(description="Other names/nicknames the text uses for this character.")
    role: str = Field(description="One short line: who they are in the story.")
    appearance: str = Field(
        description=(
            "Permanent, purely visual description used as the image prompt: sex, apparent age, "
            "build, height, skin tone, face shape, eye colour/shape, hair colour/length/style, "
            "distinguishing marks. No clothing, no personality."
        )
    )
    outfit: str = Field(description="What they are wearing right now, visually precise.")


class CharacterUpdate(_Strict):
    name: str = Field(description="Canonical name of an existing character.")
    outfit: str = Field(description="New outfit, or empty string if unchanged.")
    appearance_change: str = Field(
        description="Permanent visual change that happened in this chapter (new scar, haircut...), or empty string."
    )


class Dialogue(_Strict):
    speaker: str = Field(description="Canonical character name.")
    text: str
    kind: BubbleKind


class PanelPlan(_Strict):
    shot: Shot
    location: str = Field(description="Where the panel takes place, visually described.")
    time_of_day: str
    characters: List[str] = Field(description="Canonical names of characters visible in the panel (max 3).")
    action: str = Field(
        description="What is visibly happening: poses, expressions, positions, key props. Visual only, no dialogue."
    )
    mood: str = Field(description="Lighting / emotional tone in a few words.")
    narration: str = Field(description="Caption box text, or empty string.")
    dialogue: List[Dialogue]
    sfx: str = Field(description="Sound effect lettering, or empty string.")


class ChapterPlan(_Strict):
    title: str
    new_characters: List[CharacterSpec] = Field(
        description="Characters appearing for the first time (not already in the character bible)."
    )
    character_updates: List[CharacterUpdate]
    panels: List[PanelPlan]
    summary: str = Field(description="3-6 sentence summary of this chapter for continuity in later chapters.")


# ---------------------------------------------------------------------------
# Persistent project state (story.json)
# ---------------------------------------------------------------------------


class Character(BaseModel):
    name: str
    aliases: List[str] = []
    role: str = ""
    appearance: str
    outfit: str = ""
    seed: int = 0
    reference_image: Optional[str] = None  # path relative to project dir
    first_chapter: int = 1


class ChapterRecord(BaseModel):
    number: int
    title: str
    summary: str
    source_file: str
    plan_file: str
    panel_images: List[str] = []
    strip_images: List[str] = []


class ImageSettings(BaseModel):
    backend: str = "diffusers"  # diffusers | openai | dashscope | mock
    model: str = "Qwen/Qwen-Image-2.1"
    base_url: Optional[str] = None
    steps: int = 40
    cfg: float = 4.0
    use_references: bool = False  # pass character sheets as image conditioning (edit-capable backends only)


class PlannerSettings(BaseModel):
    provider: str = "anthropic"  # anthropic | openai
    model: str = "claude-opus-5"
    base_url: Optional[str] = None
    target_panels: str = "12-24"


class Project(BaseModel):
    title: str
    style: str
    negative_prompt: str
    width: int = 800  # final strip width in pixels
    font: Optional[str] = None  # path to a .ttf for lettering; auto-detected if unset
    image: ImageSettings = ImageSettings()
    planner: PlannerSettings = PlannerSettings()
    characters: List[Character] = []
    chapters: List[ChapterRecord] = []

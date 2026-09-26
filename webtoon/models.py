"""Data models for a webtoon project: the character bible, panel plans, and saved state."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Shot = Literal["establishing", "wide", "medium", "close-up", "extreme-close-up", "over-the-shoulder", "action"]
BubbleKind = Literal["speech", "thought", "shout", "whisper", "system"]


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
    text: str = Field(description="The line exactly as written in the text (split long speeches over several bubbles).")
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
    source_paragraphs: List[int] = Field(
        description="Numbers of the [n] paragraphs of the input text that this panel adapts."
    )


class ChapterPlan(_Strict):
    title: str = Field(description="Chapter title (the given one if any, otherwise a short invented one).")
    new_characters: List[CharacterSpec] = Field(
        description="Characters appearing for the first time (not already in the character bible)."
    )
    character_updates: List[CharacterUpdate]
    panels: List[PanelPlan]
    summary: str = Field(description="2-5 sentence summary of the events of this text, for continuity.")


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
    backend: str = "comfyui"  # comfyui | diffusers | openai | dashscope | mock
    model: str = "qwen-image-2.1"  # diffusers folder / repo id, or model name on a server
    base_url: Optional[str] = None  # server URL (comfyui default: http://127.0.0.1:8188)
    steps: int = 25
    cfg: float = 1.0
    use_references: bool = True  # pass character sheets as image conditioning (edit-capable backends only)
    megapixels: float = 1.0  # panel resolution; panels are shrunk to the strip width afterwards

    # --- comfyui backend (Qwen-Image 2.1 split checkpoints: .gguf / .safetensors) ---
    diffusion_model: Optional[str] = None  # e.g. .../qwen-image-2.1-UC-BF16.gguf
    text_encoder: Optional[str] = None  # e.g. .../qwen3vl_8b_bf16.safetensors
    vae: Optional[str] = None  # e.g. .../qwen_image_2.1_vae_bf16.safetensors
    comfy_dir: Optional[str] = None  # local ComfyUI install: model files get linked in, server auto-started
    auto_start: bool = True
    comfy_args: str = ""  # extra ComfyUI launch arguments, e.g. "--lowvram"
    sampler: str = "euler"
    scheduler: str = "simple"
    workflow_file: Optional[str] = None  # optional custom API-format workflow with {{placeholders}}


class PlannerSettings(BaseModel):
    provider: str = "anthropic"  # anthropic | llamacpp | openai | mock
    model: str = "claude-opus-5"
    base_url: Optional[str] = None
    # --- llamacpp: a local GGUF served by llama-server, started only while storyboarding ---
    llm_model: Optional[str] = None  # path to the .gguf
    llama_server: str = "llama-server"  # binary name on PATH, or its full path
    llm_context: int = 32768
    llm_args: str = "--reasoning off"  # extra llama-server arguments
    segment_words: int = 1200  # long chapters are storyboarded in segments of about this many words
    panels_per_1000_words: float = 10.0  # pacing; more panels = more faithful, slower to render
    max_retries: int = 2  # re-plans of a segment when the coverage check finds skipped text


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

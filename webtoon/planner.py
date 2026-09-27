"""Turns chapter text into a panel-by-panel plan using an LLM.

Long chapters are storyboarded in segments (see segment.py). Each segment gets the
persistent character bible, the story so far and the end of the previous segment,
so the characters and the scene stay continuous. After every segment a coverage check
verifies that every paragraph and every line of dialogue made it into the plan; gaps
are sent back to the planner, and anything still missing after the retries is
inserted as captions so no content is ever dropped.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Callable, List, Optional

from pydantic import ValidationError

from .llm_server import DEFAULT_LLAMA_URL, PlannerSession
from .models import ChapterPlan, Dialogue, PanelPlan, PlannerSettings, Project
from .project import apply_plan, bible_for_prompt, character_index
from .segment import Segment, check_coverage, extract_dialogue, patch_gaps, split_segments, target_panels

Log = Callable[[str], None]

SYSTEM_PROMPT = """You are a webtoon storyboard artist adapting prose chapters into vertical-scroll webtoon panels.

You receive: the CHARACTER BIBLE (characters already established), the STORY SO FAR, what happened earlier in THIS CHAPTER, and the next part of the chapter with numbered paragraphs.

Faithfulness - the most important rule:
- Adapt ALL of the given text, in order. Never skip, merge away or summarise out an event, an action, a meaningful description, an inner thought or a line of dialogue. If there is a lot of content, use more panels.
- Every paragraph number [n] must appear in the `source_paragraphs` of at least one panel (a panel may adapt several paragraphs; a long paragraph may span several panels).
- Put every line of dialogue in a bubble (a `dialogue` entry) exactly as written (verbatim, without the quotation marks) - never inside `narration`. Split a long speech across several bubbles or panels, at most ~25 words per bubble and 3 bubbles per panel. Inner thoughts use kind "thought".
- Show, don't narrate: what can be drawn goes into `action`, not into captions. `narration` is for what can't be shown (time skips, backstory, inner feelings) and must be SHORT - one or two sentences, at most ~25 words, condensed from the prose rather than copied. Most panels have no caption. Paragraphs that are pure description are covered by the pictures, not by captions.
- Game / system windows (quests, character sheets, stat blocks, item descriptions, "You have defeated..." notifications - common in LitRPG) are dialogue entries with kind "system" and speaker "System", text verbatim with its line breaks (a very long sheet may be split over consecutive panels). The window is added during lettering: in `action`, only show the character looking at something floating in front of them - do not describe the window or any writing.
- `sfx` only for a distinct sound or impact that actually happens in this moment (a thud, a crash, a squeak); leave it empty in most panels - at most one panel in four has one, and never for emotions ("OUCH" for confusion is wrong).

Characters:
- Every character already in the bible MUST be referred to by their exact canonical `name`, even if the text uses a nickname or pronoun. Never re-describe or redesign them; list a character in `new_characters` only if they are genuinely absent from the bible.
- For a new character, write `appearance` as a precise, permanent, purely visual description an illustrator could reproduce identically every time (sex, build, skin tone, face, eyes, hair colour/length/style, distinguishing marks). If the text gives few details, invent specific, distinctive ones that fit the story, and make each character visually distinct from the others.
- `age`: the age the story states or implies. Adults are adults - a grown man who plays video games and has adult parents is in his 20s or 30s, never a child. When unsure whether someone is an adult, make them clearly adult unless the text says they are a child.
- Recurring creatures, monsters and important objects also go in `new_characters` (role: what it is, age: "n/a"), with a precise appearance including SIZE relative to a person, so they look the same in every panel (e.g. a monstrous hamster "the size of a man's head"). Use their canonical name in `characters` when they are visible.
- List a character in `character_updates` when their outfit or permanent look changes, or to fill in an `age` the bible doesn't have yet.

Panels:
- Pace it like a webtoon: establishing shots when the location changes, close-ups for emotional beats, action shots for movement. Vary shots.
- Each panel is one single image: at most 3 characters visible, one moment in time. `action` must be concrete and visual (poses, expressions, where people are, key props) because it becomes an image prompt. Dialogue speakers must be canonical names; a speaker does not need to be visible.
- Keep locations consistent: describe a recurring location with the same words each time.
- `action` describes only what is visible in this one frame. Never put backstory, memories, thoughts or explanations there (those go in captions); if a caption talks about something not present (e.g. a character's father), the image shows the present scene, or a clearly separate flashback panel.
- Keep every image non-explicit and all-ages. Never describe nudity in `action` or `outfit`: a character who is naked in the prose is drawn in simple modest clothing, and the caption or dialogue can mention it. Graphic violence is shown through reactions, motion and aftermath rather than gore.
"""


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def _panel_brief(p: PanelPlan) -> str:
    lines = [f"- [{p.shot}] {p.location}; {', '.join(p.characters) or 'no characters'}: {p.action}"]
    lines += [f"    {d.speaker}: \"{d.text}\"" for d in p.dialogue]
    return "\n".join(lines)


def _segment_prompt(project: Project, segment: Segment, chapter_number: int, chapter_so_far: List[str],
                    previous_panels: List[PanelPlan], chapter_title: str) -> str:
    story_so_far = "\n".join(f"Chapter {c.number} - {c.title}: {c.summary}"
                             for c in project.chapters if c.number < chapter_number) or "(none)"
    earlier = "\n".join(chapter_so_far) or "(this is the start of the chapter)"
    last_panels = "\n".join(_panel_brief(p) for p in previous_panels[-3:]) or "(none)"
    n = target_panels(segment.words, project.planner.panels_per_1000_words)
    part = f"part {segment.index + 1} of {segment.total}" if segment.total > 1 else "the whole chapter"
    title_hint = f'The chapter title is "{chapter_title}"; reuse it as `title`.' if chapter_title else ""
    return (
        f"<character_bible>\n{bible_for_prompt(project)}\n</character_bible>\n\n"
        f"<story_so_far>\n{story_so_far}\n</story_so_far>\n\n"
        f"<earlier_in_this_chapter>\n{earlier}\n</earlier_in_this_chapter>\n\n"
        f"<last_panels_drawn>\n{last_panels}\n</last_panels_drawn>\n\n"
        f"<text chapter=\"{chapter_number}\" part=\"{part}\">\n{segment.numbered()}\n</text>\n\n"
        f"Storyboard this text ({segment.words} words, paragraphs [1]-[{len(segment.units)}]) as about {n} panels "
        f"- more if needed so that nothing is left out. Continue seamlessly from the last panels drawn. {title_hint}"
    )


def _retry_prompt(base: str, draft: ChapterPlan, problems: str) -> str:
    return (
        f"{base}\n\n<previous_draft>\n{draft.model_dump_json()}\n</previous_draft>\n\n"
        f"<coverage_problems>\n{problems}\n</coverage_problems>\n\n"
        "Your previous draft skipped content. Return the complete revised storyboard for this text: keep what was "
        "good, and add or extend panels, in story order, so that every problem above is fixed."
    )


# ---------------------------------------------------------------------------
# Chapter orchestration
# ---------------------------------------------------------------------------


def plan_chapter(project: Project, chapter_text: str, chapter_number: int, cache_dir: Optional[Path] = None,
                 log: Log = print, session: Optional[PlannerSession] = None) -> tuple[ChapterPlan, dict]:
    """Plan a whole chapter segment by segment. Returns (plan, coverage report).

    Mutates `project` (character bible) as segments introduce characters, so later
    segments reuse them. Segment results are cached in `cache_dir` for resuming.
    A local LLM is started on the first real planner call and stopped when the
    session ends (pass a session to keep it loaded across several chapters).
    """
    if session is None:
        with PlannerSession(project, log) as own:
            return plan_chapter(project, chapter_text, chapter_number, cache_dir, log, own)
    settings = project.planner
    segments = split_segments(chapter_text, settings.segment_words)
    heading = chapter_text.strip().splitlines()[0].strip() if chapter_text.strip() else ""
    chapter_title = re.sub(r"^(chapter|ch\.?)\s*\d+\s*[-:–—.]?\s*", "", heading, flags=re.I) if (
        len(heading) < 80 and re.match(r"^(chapter|ch\.?)\s*\d+", heading, re.I)) else ""
    log(f"Chapter {chapter_number}: {sum(s.words for s in segments)} words -> {len(segments)} segment(s)")

    plans: List[ChapterPlan] = []
    report: dict = {"segments": []}
    all_panels: List[PanelPlan] = []
    offset = 0
    for seg in segments:
        # the instructions are part of the key, so improved storyboarding rules re-plan old segments
        key = hashlib.sha256(f"{settings.provider}:{settings.model}:{settings.panels_per_1000_words}:"
                             f"{SYSTEM_PROMPT}:{seg.text}".encode()).hexdigest()[:16]
        cache = cache_dir / f"segment_{seg.index + 1:03d}.json" if cache_dir else None
        plan = None
        entry = {"segment": seg.index + 1, "paragraphs": len(seg.units), "words": seg.words}
        if cache and cache.exists():
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("key") == key:
                plan = ChapterPlan.model_validate(cached["plan"])
                entry.update(cached.get("report", {}))
                log(f"  segment {seg.index + 1}/{len(segments)}: reusing saved plan")

        if plan is None:
            base = _segment_prompt(project, seg, chapter_number, [p.summary for p in plans], all_panels,
                                   chapter_title or (plans[0].title if plans else ""))
            log(f"  segment {seg.index + 1}/{len(segments)} ({seg.words} words) ...")
            if settings.provider != "mock":
                session.ensure_ready()
            plan = _call_planner(settings, base, seg, project)
            coverage = check_coverage(seg, plan)
            attempts = 1
            while not coverage.ok and attempts <= settings.max_retries:
                log(f"    coverage check: {len(coverage.missing_units)} paragraph(s) and "
                    f"{len(coverage.missing_dialogue)} dialogue line(s) missing, {len(coverage.long_captions)} "
                    "caption(s) too long - asking for a revision")
                revised = _call_planner(settings, _retry_prompt(base, plan, coverage.feedback(seg)), seg, project)
                attempts += 1
                new_cov = check_coverage(seg, revised)
                # missing story content matters most, then everything else
                rank = lambda c: (len(c.missing_units) + len(c.missing_dialogue), c.problems)  # noqa: E731
                if rank(new_cov) <= rank(coverage):
                    plan, coverage = revised, new_cov
            if coverage.long_captions:
                log(f"    note: {len(coverage.long_captions)} caption(s) are still long")
            if coverage.content_missing:
                log(f"    still missing after {attempts} attempt(s); inserting the text as captions")
                plan = patch_gaps(seg, plan, coverage)
            entry.update({
                "attempts": attempts,
                "patched_paragraphs": [seg.units[n - 1] for n in coverage.missing_units],
                "patched_dialogue": [line for _, line in coverage.missing_dialogue],
            })
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps({"key": key, "plan": plan.model_dump(), "report": entry}, indent=2,
                                            ensure_ascii=False), encoding="utf-8")

        apply_plan(project, plan, chapter_number)  # later segments see characters introduced here
        for p in plan.panels:  # segment-local paragraph numbers -> chapter-wide numbers
            p.source_paragraphs = [n + offset for n in p.source_paragraphs]
        offset += len(seg.units)
        plans.append(plan)
        all_panels.extend(plan.panels)
        report["segments"].append(entry)
        log(f"    {len(plan.panels)} panels")

    merged = ChapterPlan(
        title=chapter_title or plans[0].title,
        new_characters=[c for p in plans for c in p.new_characters],
        character_updates=[u for p in plans for u in p.character_updates],
        panels=all_panels,
        summary=" ".join(p.summary for p in plans),
    )
    report["total_panels"] = len(all_panels)
    report["patched"] = sum(len(e.get("patched_paragraphs", [])) + len(e.get("patched_dialogue", []))
                            for e in report["segments"])
    return merged, report


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


def _call_planner(settings: PlannerSettings, user: str, segment: Segment, project: Project) -> ChapterPlan:
    if settings.provider == "mock":
        return mock_plan(segment, project)
    if settings.provider == "anthropic":
        raw = _plan_with_claude(settings, user)
    elif settings.provider in ("openai", "llamacpp"):
        raw = _plan_with_openai_compatible(settings, user)
    else:
        raise ValueError(f"Unknown planner provider: {settings.provider}")
    try:
        return ChapterPlan.model_validate_json(_strip_fences(raw))
    except ValidationError as exc:
        raise RuntimeError(f"Planner returned JSON that doesn't match the plan schema:\n{exc}\n\n{raw[:2000]}") from exc


def _strip_fences(raw: str) -> str:
    """Extract the JSON object from an LLM reply (local models may add <think> blocks, fences or chatter)."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.S)
    if m:
        return m.group(1)
    start, end = raw.find("{"), raw.rfind("}")
    return raw[start:end + 1] if 0 <= start < end else raw


def _plan_with_claude(settings: PlannerSettings, user: str) -> str:
    import anthropic

    client = anthropic.Anthropic(base_url=settings.base_url) if settings.base_url else anthropic.Anthropic()
    # Streaming: plans can be long. Server-side fallbacks keep a false-positive
    # safety decline (e.g. on a violent chapter) from failing the whole run.
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
        raise RuntimeError(f"The planner model declined this text: {message.stop_details}")
    if message.stop_reason == "max_tokens":
        raise RuntimeError("Planner output was cut off (max_tokens). Lower the segment size in the settings.")
    return "".join(block.text for block in message.content if block.type == "text")


def _plan_with_openai_compatible(settings: PlannerSettings, user: str) -> str:
    """For a self-hosted LLM (e.g. a Qwen model behind vLLM / Ollama / LM Studio)."""
    from openai import BadRequestError, OpenAI

    base_url = settings.base_url or (DEFAULT_LLAMA_URL if settings.provider == "llamacpp" else None)
    # local models can take many minutes for a long storyboard
    client = OpenAI(base_url=base_url, api_key=os.environ.get("OPENAI_API_KEY", "not-needed"), timeout=3600)
    schema = ChapterPlan.model_json_schema()
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + "\nRespond with a single JSON object matching this schema:\n" + json.dumps(schema),
        },
        {"role": "user", "content": user},
    ]
    # Prefer schema-constrained output; fall back for servers that only know JSON mode (or neither).
    formats = [{"type": "json_schema", "json_schema": {"name": "chapter_plan", "schema": schema}},
               {"type": "json_object"}, None]
    for i, fmt in enumerate(formats):
        try:
            extra = {"response_format": fmt} if fmt else {}
            response = client.chat.completions.create(model=settings.model, messages=messages, **extra)
            break
        except BadRequestError:
            if i == len(formats) - 1:
                raise
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise RuntimeError("Planner output was cut off. Lower the segment size in the settings.")
    return choice.message.content or ""


def mock_plan(segment: Segment, project: Project) -> ChapterPlan:
    """Offline planner for trying the pipeline: one panel per paragraph (short caption + its quoted lines)."""
    index = character_index(project)
    panels = []
    for n, unit in enumerate(segment.units, start=1):
        lowered = unit.lower()
        present = list(dict.fromkeys(c.name for alias, c in index.items() if re.search(rf"\b{re.escape(alias)}\b", lowered)))
        lines = [line for _, line in extract_dialogue([unit])]
        prose = " ".join(re.sub(r"[“\"«「『][^”\"»」』]*[”\"»」』]", " ", unit).split())
        caption = " ".join(prose.split()[:40]) + ("..." if len(prose.split()) > 40 else "")
        panels.append(PanelPlan(shot="medium", location="the scene", time_of_day="day", characters=present[:3],
                                action=unit[:300], mood="neutral", narration=caption,
                                dialogue=[Dialogue(speaker="", text=line, kind="speech") for line in lines], sfx="",
                                source_paragraphs=[n]))
    return ChapterPlan(title="Untitled", new_characters=[], character_updates=[], panels=panels,
                       summary=segment.units[0][:200])

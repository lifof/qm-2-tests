"""Splitting long chapters into plannable segments, and checking that a plan covers all of the text.

A chapter is cut into numbered *units* (paragraphs; very long paragraphs are split
at sentence boundaries). Units are grouped into segments of roughly
`segment_words` words that are storyboarded one after another. Every panel cites
the units it adapts, which lets us verify mechanically that nothing was skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import List, Sequence

from .models import ChapterPlan, Dialogue, PanelPlan

MAX_UNIT_WORDS = 160

_SENTENCE_END = re.compile(r"(?<=[.!?…。！？])\s+|(?<=[.!?…。！？][\"'”’»」』)])\s+")
_DIALOGUE_PATTERNS = [
    re.compile(r"“([^”]+)”"),
    re.compile(r"\"([^\"]+)\""),
    re.compile(r"«\s*([^»]+?)\s*»"),
    re.compile(r"「([^」]+)」"),
    re.compile(r"『([^』]+)』"),
]
# Single-quoted dialogue, only used when a text has no double-quoted dialogue at all
# (avoids mistaking apostrophes for quotes).
_SINGLE_QUOTE = re.compile(r"(?:(?<=^)|(?<=[\s(\[—–-]))[‘']([^'’]{2,}?[.,!?…—-])[’'](?=\s|$|[,.;:!?)])", re.M)


def word_count(text: str) -> int:
    return len(text.split())


@dataclass
class Segment:
    units: List[str]
    index: int = 0  # 0-based position in the chapter
    total: int = 1

    @property
    def text(self) -> str:
        return "\n\n".join(self.units)

    @property
    def words(self) -> int:
        return sum(word_count(u) for u in self.units)

    def numbered(self) -> str:
        return "\n\n".join(f"[{i}] {u}" for i, u in enumerate(self.units, start=1))


def split_units(text: str) -> List[str]:
    text = text.replace("\r\n", "\n").strip()
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) <= 1 and text.count("\n") > 3:
        # no blank lines between paragraphs: treat every line as a paragraph
        blocks = [b.strip() for b in text.split("\n") if b.strip()]
    units: List[str] = []
    for block in blocks:
        block = re.sub(r"\s*\n\s*", " ", block)
        if word_count(block) <= MAX_UNIT_WORDS:
            units.append(block)
            continue
        current: List[str] = []
        for sentence in _SENTENCE_END.split(block):
            if current and word_count(" ".join(current + [sentence])) > MAX_UNIT_WORDS:
                units.append(" ".join(current))
                current = []
            current.append(sentence)
        if current:
            units.append(" ".join(current))
    return units


def split_segments(text: str, segment_words: int) -> List[Segment]:
    segments: List[Segment] = []
    current: List[str] = []
    words = 0
    for unit in split_units(text):
        w = word_count(unit)
        if current and words + w > segment_words:
            segments.append(Segment(current))
            current, words = [], 0
        current.append(unit)
        words += w
    if current:
        segments.append(Segment(current))
    # Avoid a tiny trailing segment with no context: fold it into the previous one.
    if len(segments) > 1 and segments[-1].words < segment_words * 0.25:
        segments[-2].units.extend(segments.pop().units)
    for i, s in enumerate(segments):
        s.index, s.total = i, len(segments)
    return segments


def target_panels(words: int, per_1000_words: float) -> int:
    return max(3, round(words * per_1000_words / 1000))


# ---------------------------------------------------------------------------
# Coverage checking
# ---------------------------------------------------------------------------


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())


def extract_dialogue(units: Sequence[str]) -> List[tuple[int, str]]:
    """(1-based unit number, quoted line) for every quoted line in the units."""
    found: List[tuple[int, str]] = []
    for i, unit in enumerate(units, start=1):
        for pattern in _DIALOGUE_PATTERNS:
            found.extend((i, m.strip()) for m in pattern.findall(unit) if _norm(m))
    if not found:
        for i, unit in enumerate(units, start=1):
            found.extend((i, m.strip()) for m in _SINGLE_QUOTE.findall(unit) if _norm(m))
    return found


MAX_CAPTION_WORDS = 45


def _bubble_text(panels: Sequence[PanelPlan]) -> str:
    """Dialogue has to be in bubbles; a caption quoting it doesn't count."""
    return _norm(" ".join(d.text for p in panels for d in p.dialogue))


def line_is_covered(line: str, haystack: str) -> bool:
    needle = _norm(line)
    if not needle or needle in haystack:
        return True
    match = SequenceMatcher(None, haystack, needle, autojunk=False).find_longest_match(0, len(haystack), 0, len(needle))
    return match.size >= 0.8 * len(needle)


@dataclass
class Coverage:
    missing_units: List[int] = field(default_factory=list)
    missing_dialogue: List[tuple[int, str]] = field(default_factory=list)
    long_captions: List[tuple[int, str]] = field(default_factory=list)  # (panel number, caption)

    @property
    def problems(self) -> int:
        return len(self.missing_units) + len(self.missing_dialogue) + len(self.long_captions)

    @property
    def ok(self) -> bool:
        return self.problems == 0

    @property
    def content_missing(self) -> bool:
        return bool(self.missing_units or self.missing_dialogue)

    def feedback(self, segment: Segment) -> str:
        lines = []
        if self.missing_units:
            lines.append("These paragraphs are not cited by any panel's source_paragraphs, so their content was skipped:")
            lines.extend(f"[{n}] {segment.units[n - 1]}" for n in self.missing_units)
        if self.missing_dialogue:
            lines.append("These lines of dialogue are missing from the speech bubbles (they must appear verbatim in "
                         "`dialogue` entries, not in captions):")
            lines.extend(f"[{n}] \"{text}\"" for n, text in self.missing_dialogue)
        if self.long_captions:
            lines.append("These captions are too long (webtoon captions are 1-2 short sentences, max ~25 words). "
                         "Condense them, move what can be drawn into `action`, and put any dialogue in bubbles:")
            lines.extend(f"panel {n}: {text}" for n, text in self.long_captions)
        return "\n".join(lines)


def check_coverage(segment: Segment, plan: ChapterPlan) -> Coverage:
    cited = {n for p in plan.panels for n in p.source_paragraphs}
    missing_units = [n for n in range(1, len(segment.units) + 1) if n not in cited]
    haystack = _bubble_text(plan.panels)
    missing_dialogue = [(n, line) for n, line in extract_dialogue(segment.units) if not line_is_covered(line, haystack)]
    long_captions = [(i, p.narration) for i, p in enumerate(plan.panels, 1) if word_count(p.narration) > MAX_CAPTION_WORDS]
    return Coverage(missing_units, missing_dialogue, long_captions)


def patch_gaps(segment: Segment, plan: ChapterPlan, coverage: Coverage) -> ChapterPlan:
    """Last resort after retries: put any still-missing text into the plan so nothing is lost.

    Missing paragraphs become caption-only panels inserted in story order; missing
    dialogue lines become speech bubbles in the panel that adapts their paragraph.
    """
    panels = [p.model_copy(deep=True) for p in plan.panels]
    missing_units = set(coverage.missing_units)
    for n, line in coverage.missing_dialogue:
        if n in missing_units:
            continue  # the whole paragraph gets its own caption panel below
        host = next((p for p in panels if n in p.source_paragraphs), panels[-1] if panels else None)
        if host is not None:
            host.dialogue.append(Dialogue(speaker="", text=line, kind="speech"))

    for n in sorted(missing_units):
        # insert after the last panel adapting an earlier paragraph
        pos = 0
        for i, p in enumerate(panels):
            if p.source_paragraphs and min(p.source_paragraphs) < n:
                pos = i + 1
        ref = panels[pos - 1] if pos else (panels[0] if panels else None)
        panels.insert(pos, PanelPlan(
            shot="wide",
            location=ref.location if ref else "the scene",
            time_of_day=ref.time_of_day if ref else "",
            characters=ref.characters[:2] if ref else [],
            action=ref.action if ref else "the scene continues",
            mood=ref.mood if ref else "",
            narration=segment.units[n - 1],
            dialogue=[],
            sfx="",
            source_paragraphs=[n],
        ))
    return plan.model_copy(update={"panels": panels})

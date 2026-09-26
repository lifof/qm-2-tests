"""Long chapters: segmentation, coverage check, retry and last-resort patching."""

import json

import pytest

from webtoon import planner
from webtoon.models import ChapterPlan, Dialogue, PanelPlan
from webtoon.pipeline import plan_step
from webtoon.project import ProjectDir
from webtoon.segment import check_coverage, extract_dialogue, split_segments, split_units


def long_chapter(paragraphs: int = 60) -> str:
    parts = ["Chapter 7 - The Long Night"]
    for i in range(paragraphs):
        parts.append(
            f"Paragraph {i}: Mira walked further along the flooded tunnel, counting the lanterns on the wall "
            f"and listening to the water drip from the ceiling somewhere in the dark ahead of her. "
            f"\"Line {i}, are you still there?\" she called."
        )
    return "\n\n".join(parts)


def panel(units, dialogue=(), narration=""):
    return PanelPlan(shot="medium", location="tunnel", time_of_day="night", characters=[], action="walking",
                     mood="dark", narration=narration, sfx="", source_paragraphs=list(units),
                     dialogue=[Dialogue(speaker="Mira Han", text=t, kind="speech") for t in dialogue])


def faithful_plan(segment, skip=()):
    panels = []
    for n, unit in enumerate(segment.units, start=1):
        if n in skip:
            continue
        lines = [line for _, line in extract_dialogue([unit])]
        panels.append(panel([n], lines))
    return ChapterPlan(title="The Long Night", new_characters=[], character_updates=[], panels=panels,
                       summary=f"part {segment.index + 1}")


def test_split_keeps_every_word_and_respects_segment_size():
    text = long_chapter()
    segments = split_segments(text, segment_words=500)
    assert len(segments) >= 5
    assert all(s.words <= 500 + 160 for s in segments)
    rejoined = " ".join(" ".join(s.units) for s in segments).split()
    assert rejoined == text.split()


def test_huge_paragraph_is_split_at_sentences():
    text = " ".join(f"Sentence number {i} is here." for i in range(400))
    units = split_units(text)
    assert len(units) > 1
    assert " ".join(units).split() == text.split()


def test_dialogue_extraction_styles():
    units = ["“Curly quotes,” she said.", "\"Straight quotes!\"", "«Guillemets»", "「Japanese」"]
    assert [line for _, line in extract_dialogue(units)] == ["Curly quotes,", "Straight quotes!", "Guillemets", "Japanese"]
    assert [line for _, line in extract_dialogue(["'Single quotes,' he said. It's fine."])] == ["Single quotes,"]


def test_coverage_detects_skipped_paragraph_and_dialogue():
    seg = split_segments(long_chapter(5), 10_000)[0]
    plan = faithful_plan(seg, skip={3})
    plan.panels[1].dialogue = []  # unit 1 is the heading; drop unit 2's dialogue but keep citing it
    cov = check_coverage(seg, plan)
    assert cov.missing_units == [3]
    assert [n for n, _ in cov.missing_dialogue] == [2, 3]


@pytest.fixture
def project(tmp_path):
    pdir = ProjectDir(tmp_path / "story")
    proj = pdir.create("Story")
    proj.planner.segment_words = 400
    pdir.save(proj)
    return pdir, proj


def test_long_chapter_retries_until_complete(project, tmp_path, monkeypatch):
    pdir, proj = project
    calls = []

    def fake(settings, user, segment, project_):
        calls.append(segment.index)
        retry = "<coverage_problems>" in user
        return faithful_plan(segment, skip=() if retry else {2})

    monkeypatch.setattr(planner, "_call_planner", fake)
    chapter = tmp_path / "ch.txt"
    chapter.write_text(long_chapter(), encoding="utf-8")
    plan = plan_step(pdir, proj, chapter, 1, log=lambda m: None)

    segments = split_segments(long_chapter(), 400)
    assert len(calls) == 2 * len(segments)  # one retry per segment
    total_units = sum(len(s.units) for s in segments)
    cited = sorted({n for p in plan.panels for n in p.source_paragraphs})
    assert cited == list(range(1, total_units + 1))  # chapter-wide numbering, nothing skipped
    assert plan.title == "The Long Night"
    report = json.loads((pdir.chapter_dir(1) / "coverage.json").read_text())
    assert report["patched"] == 0

    # Re-planning the same chapter reuses the saved segment plans instead of calling the LLM again.
    calls.clear()
    plan_step(pdir, proj, chapter, 1, log=lambda m: None)
    assert calls == []


def test_stubborn_gaps_are_inserted_as_captions(project, tmp_path, monkeypatch):
    pdir, proj = project
    monkeypatch.setattr(planner, "_call_planner", lambda s, u, seg, p: faithful_plan(seg, skip={2}))
    chapter = tmp_path / "ch.txt"
    chapter.write_text(long_chapter(20), encoding="utf-8")
    plan = plan_step(pdir, proj, chapter, 1, log=lambda m: None)

    captions = [p.narration for p in plan.panels]
    bubbles = " ".join(d.text for p in plan.panels for d in p.dialogue)
    for seg in split_segments(long_chapter(20), 400):
        assert seg.units[1] in captions  # the paragraph the planner always skipped is kept verbatim
    for _, line in extract_dialogue(split_units(long_chapter(20))):
        assert line in bubbles or any(line in c for c in captions)
    report = json.loads((pdir.chapter_dir(1) / "coverage.json").read_text())
    assert report["patched"] > 0
    # the inserted caption panels sit in story order
    order = [min(p.source_paragraphs) for p in plan.panels]
    assert order == sorted(order)


def test_planner_json_extraction_from_local_models():
    from webtoon.planner import _strip_fences

    body = '{"title": "x"}'
    assert _strip_fences(body) == body
    assert _strip_fences(f"<think>let me plan {{panels}}</think>\n{body}") == body
    assert _strip_fences(f"```json\n{body}\n```") == body
    assert _strip_fences(f"Sure! Here is the storyboard:\n{body}\nHope it helps.") == body

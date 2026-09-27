"""End-to-end tests with canned plans and the mock image backend (no GPU / API key needed)."""

from pathlib import Path

from webtoon.cli import main
from webtoon.models import ChapterPlan
from webtoon.project import ProjectDir, bible_for_prompt
from webtoon.prompts import panel_prompt

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def run_chapter(project: Path, n: int) -> None:
    main([
        "chapter", str(project), str(EXAMPLES / f"chapter{n}.txt"),
        "--plan-file", str(EXAMPLES / f"chapter{n}.plan.json"),
        "--image-backend", "mock",
    ])


def test_two_chapters_keep_same_characters(tmp_path):
    project = tmp_path / "story"
    run_chapter(project, 1)
    state = ProjectDir(project).load()
    first = {c.name: c for c in state.characters}
    assert set(first) == {"Mira Han", "Joon Seo"}

    run_chapter(project, 2)
    state = ProjectDir(project).load()
    after = {c.name: c for c in state.characters}

    # Chapter 2 re-lists Mira with a different design; the locked look must win.
    assert after["Mira Han"].appearance == first["Mira Han"].appearance
    assert after["Mira Han"].seed == first["Mira Han"].seed
    assert after["Mira Han"].reference_image == first["Mira Han"].reference_image
    # Outfit update applied, new character added, alias merged.
    assert "raincoat" in after["Mira Han"].outfit
    assert "Captain Vale" in after
    assert "Mimi" in after["Mira Han"].aliases

    # The bible given to the planner for chapter 3 contains every character.
    bible = bible_for_prompt(state)
    assert all(name in bible for name in after)

    for n in (1, 2):
        cdir = project / f"chapter_{n:02d}"
        assert (cdir / "reader.html").exists()
        assert list((cdir / "strip").glob("*.jpg"))
    assert (project / "index.html").exists()
    assert [c.number for c in state.chapters] == [1, 2]


def test_panel_prompt_uses_bible_description_for_aliases(tmp_path):
    project = tmp_path / "story"
    run_chapter(project, 1)
    run_chapter(project, 2)  # introduces the "Mimi" alias
    state = ProjectDir(project).load()
    plan = ChapterPlan.model_validate_json((EXAMPLES / "chapter2.plan.json").read_text())
    alias_panel = next(p for p in plan.panels if "Mimi" in p.characters)
    prompt, chars = panel_prompt(state, alias_panel)
    assert [c.name for c in chars] == ["Mira Han"]
    assert state.characters[0].appearance in prompt


def test_rerender_single_panel_and_letter_only(tmp_path):
    project = tmp_path / "story"
    run_chapter(project, 1)
    art = project / "chapter_01" / "art" / "panel_002.png"
    before = art.stat().st_mtime_ns
    main(["render", str(project), "1", "--only", "2"])
    assert art.stat().st_mtime_ns != before
    main(["render", str(project), "1", "--letter-only"])


def test_system_windows_are_lettered_and_overflow_below_the_art():
    from PIL import Image

    from webtoon.compose import letter_panel
    from webtoon.models import Dialogue, PanelPlan

    sheet = "\n".join(["Jason Asano", "Race: Outworlder.", "Current rank: normal."] + [f"[Attribute {i}]: normal." for i in range(30)])
    panel = PanelPlan(shot="medium", location="maze", time_of_day="day", characters=[], action="a", mood="m",
                      narration="", sfx="", source_paragraphs=[1],
                      dialogue=[Dialogue(speaker="System", kind="system", text="New Quest: [Stranger]\nReward: pants."),
                                Dialogue(speaker="System", kind="system", text=sheet),
                                Dialogue(speaker="Jason", kind="speech", text="Is this a character sheet?")])
    art = Image.new("RGB", (1024, 1024), (40, 160, 60))
    out = letter_panel(art, panel, 800)
    assert out.width == 800 and out.height > 800  # the long sheet (and the bubble after it) continue below the art
    # the first window is drawn on the art: its dark-blue fill replaces the green there
    r, g, b = out.getpixel((400, 60))
    assert b > g and b > r


def test_warns_when_storyboard_has_no_text(tmp_path):
    from webtoon.models import ChapterPlan
    from webtoon.pipeline import plan_step, render_step

    plan = ChapterPlan.model_validate_json((EXAMPLES / "chapter1.plan.json").read_text())
    for p in plan.panels:
        p.narration, p.dialogue, p.sfx = "", [], ""
    pdir = ProjectDir(tmp_path / "story")
    project = pdir.create("Story")
    project.image.backend = "mock"
    messages = []
    plan_step(pdir, project, EXAMPLES / "chapter1.txt", 1, plan=plan, log=messages.append)
    render_step(pdir, project, 1, log=messages.append)
    assert any("(0 with text)" in m for m in messages)
    assert any("no dialogue, captions or sound effects" in m for m in messages)


def test_reference_sheet_is_redrawn_when_its_description_changes(tmp_path):
    project = tmp_path / "story"
    run_chapter(project, 1)
    pdir = ProjectDir(project)
    state = pdir.load()
    sheet = project / state.characters[0].reference_image
    before = sheet.stat().st_mtime_ns
    plan = pdir.load_plan(1)
    plan.new_characters[0].outfit = "nothing (naked)"  # e.g. an older storyboard left a character unclothed
    pdir.save_plan(1, plan)
    main(["render", str(project), "1", "--changed"])
    after = pdir.load().characters[0]
    assert sheet.stat().st_mtime_ns != before
    assert "modest clothing" in after.reference_prompt and "naked" not in after.reference_prompt

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

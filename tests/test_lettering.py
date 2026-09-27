"""lettering='model': Qwen-Image draws the text into the art; the app only adds what's too long."""

from pathlib import Path

from PIL import Image, ImageChops

from webtoon.cli import main
from webtoon.models import Character, Dialogue, PanelPlan, Project
from webtoon.project import ProjectDir
from webtoon.prompts import MODEL_TEXT_MARKER, negative_prompt_for, panel_prompt, split_lettering

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
SHEET = "\n".join(["Jason Asano", "Race: Outworlder.", "Current rank: normal."] + [f"[Attr {i}]: normal." for i in range(12)])


def project(lettering="model") -> Project:
    p = Project(title="T", style="webtoon style", negative_prompt="text, letters, watermark, speech bubble, blurry",
                lettering=lettering)
    p.characters.append(Character(name="Jason Asano", aliases=["Jason"], appearance="completely bald young man"))
    return p


def panel(**kw) -> PanelPlan:
    base = dict(shot="medium", location="hedge maze", time_of_day="day", characters=["Jason Asano"],
                action="Jason looks up", mood="bright", narration="", dialogue=[], sfx="", source_paragraphs=[1])
    return PanelPlan(**{**base, **kw})


def test_app_mode_keeps_everything_for_the_app():
    p = panel(dialogue=[Dialogue(speaker="Jason Asano", text="Huh.", kind="speech")], sfx="thud")
    to_model, to_app = split_lettering(project("app"), p)
    assert not to_model.dialogue and not to_model.sfx and to_app == p
    prompt, _ = panel_prompt(project("app"), p)
    assert "no text" in prompt and MODEL_TEXT_MARKER not in prompt


def test_short_text_goes_to_the_model_long_text_to_the_app():
    long_caption = " ".join(["word"] * 45)
    p = panel(narration=long_caption, sfx="thud", dialogue=[
        Dialogue(speaker="Jason Asano", text="What the bloody hell is going on?", kind="speech"),
        Dialogue(speaker="System", text="New Quest: [Stranger in a Strange Land]\nReward: Simple pants.", kind="system"),
        Dialogue(speaker="System", text=SHEET, kind="system"),
        Dialogue(speaker="Farrah", text="Over here!", kind="shout"),
    ])
    to_model, to_app = split_lettering(project(), p)
    assert [d.text for d in to_model.dialogue] == ["What the bloody hell is going on?",
                                                   "New Quest: [Stranger in a Strange Land]\nReward: Simple pants.",
                                                   "Over here!"]
    assert to_model.sfx == "thud" and to_model.narration == ""
    assert to_app.narration == long_caption and [d.text for d in to_app.dialogue] == [SHEET]

    prompt, _ = panel_prompt(project(), p)
    assert 'speech bubble with its tail pointing to Jason Asano with the text "What the bloody hell is going on?"' in prompt
    assert 'displaying the lines "New Quest: [Stranger in a Strange Land]", "Reward: Simple pants."' in prompt
    assert 'shout bubble coming from off-panel with the bold capital-letter text "OVER HERE!"' in prompt
    assert '"THUD"' in prompt and MODEL_TEXT_MARKER in prompt and "no text" not in prompt
    assert negative_prompt_for(project(), prompt) == "watermark, blurry"


def test_per_panel_word_budget():
    lines = [Dialogue(speaker="Jason Asano", text=" ".join(["hi"] * 25), kind="speech") for _ in range(3)]
    to_model, to_app = split_lettering(project(), panel(dialogue=lines))
    assert len(to_model.dialogue) == 2 and len(to_app.dialogue) == 1  # 2 x 25 fits the 60-word budget, not 3


def test_switching_redraws_only_panels_with_text(tmp_path):
    proj = tmp_path / "story"
    main(["chapter", str(proj), str(EXAMPLES / "chapter1.txt"), "--plan-file", str(EXAMPLES / "chapter1.plan.json"),
          "--image-backend", "mock"])
    art = proj / "chapter_01" / "art"
    before = {f.name: f.stat().st_mtime_ns for f in art.glob("*.png")}
    plan = ProjectDir(proj).load_plan(1)

    main(["init", str(proj), "--lettering", "model"])
    main(["render", str(proj), "1"])  # without --changed nothing is redrawn
    assert {f.name: f.stat().st_mtime_ns for f in art.glob("*.png")} == before
    main(["render", str(proj), "1", "--changed"])

    for i, p in enumerate(plan.panels, 1):
        name = f"panel_{i:03d}.png"
        has_text = bool(p.narration or p.dialogue or p.sfx)
        assert (art / name).stat().st_mtime_ns != before[name] if has_text else (art / name).stat().st_mtime_ns == before[name]
        assert (MODEL_TEXT_MARKER in (art / name).with_suffix(".prompt.txt").read_text()) == has_text

    # A panel whose text all went to the model gets no app lettering: panels/ == resized art.
    i = next(i for i, p in enumerate(plan.panels, 1) if p.dialogue and not p.narration)
    raw = Image.open(art / f"panel_{i:03d}.png").convert("RGB")
    raw = raw.resize((800, round(raw.height * 800 / raw.width)), Image.LANCZOS)
    lettered = Image.open(proj / "chapter_01" / "panels" / f"panel_{i:03d}.png").convert("RGB")
    assert ImageChops.difference(raw, lettered).getbbox() is None

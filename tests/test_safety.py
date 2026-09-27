"""Content safeguards on image prompts, stricter storyboard checks, and SFX placement."""

import re

from PIL import Image

from webtoon.compose import _letter_on, letter_panel, load_font
from webtoon.models import ChapterPlan, Character, Dialogue, PanelPlan, Project
from webtoon.prompts import panel_prompt, reference_sheet_prompt
from webtoon.safety import is_confirmed_adult, is_minor, parse_age
from webtoon.segment import check_coverage, patch_gaps, split_segments

NUDE_WORDS = re.compile(r"\b(naked|nude|privates|genitals|topless|shirtless)\b", re.I)


def make_project(*chars: Character) -> Project:
    p = Project(title="T", style="webtoon style", negative_prompt="blurry")
    p.characters.extend(chars)
    return p


def panel(action: str, *names: str, **kw) -> PanelPlan:
    base = dict(shot="wide", location="hedge maze", time_of_day="day", characters=list(names), action=action,
                mood="tense", narration="", dialogue=[], sfx="", source_paragraphs=[1])
    return PanelPlan(**{**base, **kw})


JASON = dict(name="Jason Asano", aliases=["Jason"], appearance="completely bald man, no eyebrows")


def test_ages():
    assert parse_age("32") == 32 and parse_age("early 30s") == 30 and parse_age("adult") == 18
    assert parse_age("") is None and parse_age("n/a") is None
    assert is_confirmed_adult(Character(**JASON, age="32"))
    assert not is_confirmed_adult(Character(**JASON))  # unknown age is not assumed adult
    assert is_minor(Character(name="Kid", age="12", appearance="small boy"))
    assert is_minor(Character(name="X", age="", appearance="a teenage girl with a ponytail"))


def test_nudity_never_reaches_the_image_model():
    for age in ("32", ""):
        p = make_project(Character(**JASON, age=age, outfit="nothing (naked)"))
        prompt, _ = panel_prompt(p, panel("Jason, naked, stands between the hedges, hands covering his privates, "
                                          "looking around nervously", "Jason Asano"))
        assert not NUDE_WORDS.search(prompt), prompt
        assert "modest clothing" in prompt and "fully and modestly clothed" in prompt
        assert "looking around nervously" in prompt  # the rest of the action survives
    assert "an adult, 32 years old" in panel_prompt(make_project(Character(**JASON, age="32")),
                                                    panel("Jason waves", "Jason Asano"))[0]


def test_revealing_clothes_only_for_confirmed_adults():
    adult = make_project(Character(**JASON, age="32", outfit="swim trunks"))
    prompt, _ = panel_prompt(adult, panel("Jason, shirtless, dives into the lake", "Jason Asano"))
    assert "shirtless" in prompt and "swim trunks" in prompt
    for c in (Character(name="Tom", age="12", appearance="small boy", outfit="swim trunks"),
              Character(name="Tom", age="", appearance="young man", outfit="swim trunks")):
        prompt, _ = panel_prompt(make_project(c), panel("Tom, shirtless, dives into the lake", "Tom"))
        assert not NUDE_WORDS.search(prompt) and "swim trunks" not in prompt and "modest clothing" in prompt


def test_reference_sheets_are_always_clothed():
    p = make_project(Character(**JASON, age="32", outfit="nothing"))
    sheet = reference_sheet_prompt(p, p.characters[0])
    assert "nothing" not in sheet and "modest clothing" in sheet and "Fully and modestly clothed" in sheet


def test_dialogue_in_a_caption_or_a_wall_of_text_needs_a_revision():
    seg = split_segments('There wasn\'t any response.\n\n"Maybe it\'s not morning. Guten tag?"', 1000)[0]
    lazy = ChapterPlan(title="t", new_characters=[], character_updates=[], summary="s", panels=[
        panel("Jason waits", narration="There wasn't any response.", source_paragraphs=[1]),
        panel("Jason shrugs", narration="Maybe it's not morning. Guten tag? " + "word " * 50, source_paragraphs=[2])])
    cov = check_coverage(seg, lazy)
    assert [n for n, _ in cov.missing_dialogue] == [2] and [n for n, _ in cov.long_captions] == [2]
    assert "not in captions" in cov.feedback(seg) and "too long" in cov.feedback(seg)
    patched = patch_gaps(seg, lazy, cov)
    assert patched.panels[1].dialogue[-1].text == "Maybe it's not morning. Guten tag?"


def test_sfx_avoids_bubbles_and_captions():
    p = panel("x", narration="He dropped to his knees.", sfx="squeak! splat!",
              dialogue=[Dialogue(speaker="Jason Asano", text="AAARGH!", kind="shout"),
                        Dialogue(speaker="Jason Asano", text="What in the merry hell is happening?", kind="speech")])
    art = Image.new("RGB", (1024, 1024), (90, 90, 90))
    out = letter_panel(art, p, 800)
    occupied = []
    _letter_on(art.resize((800, 800)), p, 800, load_font(800 // 30), occupied)
    yellow = [(x, y) for x in range(0, 800, 4) for y in range(0, 800, 4)
              if (lambda r, g, b: r > 230 and 190 < g < 235 and b < 60)(*out.getpixel((x, y)))]
    assert yellow, "SFX was drawn"
    inside = [pt for pt in yellow if any(o[0] <= pt[0] <= o[2] and o[1] <= pt[1] <= o[3] for o in occupied)]
    assert not inside

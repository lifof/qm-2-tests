"""Deterministic content safeguards applied to every image prompt.

These don't rely on the storyboard LLM following instructions:

- Nudity is never sent to the image model. Clauses describing it are removed and the
  characters are drawn in modest clothing (captions and dialogue keep the story's words).
- Revealing clothing (shirtless, underwear, swimwear...) is only drawn when every
  character in the panel has an explicitly adult age; otherwise they are covered too.
- Characters are described with their age, so adults are drawn as adults.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

from .models import Character

NUDITY = re.compile(
    r"\b(naked|nude|nudity|unclothed|undressed|in the buff|au naturel|birthday suit|"
    r"no (?:clothes|clothing|pants|trousers|underwear)|without (?:any )?(?:clothes|clothing|pants)|"
    r"bare[- ](?:body|bodied|skin|bottom|butt|buttocks|chest|breasts?|backside)|"
    r"privates|private parts|genitals?|groin|crotch|breasts?|nipples?|buttocks|topless|bottomless|"
    r"skinny[- ]dipp\w*|exposed (?:body|skin|chest))\b", re.I)
REVEALING = re.compile(
    r"\b(shirtless|bare[- ]chested|underwear|boxers|briefs|panties|lingerie|bra|swimsuit|swimwear|bikini|"
    r"trunks|bath(?:ing)?|shower(?:ing)?|towel|loincloth|scantily|revealing)\b", re.I)
MINOR_WORDS = re.compile(
    r"\b(child|children|kid|kids|boy|girl|teen|teens|teenager|teenaged|adolescent|toddler|baby|infant|minor|"
    r"preteen|pre-teen|schoolboy|schoolgirl|underage|little (?:boy|girl))\b", re.I)
NO_OUTFIT = re.compile(r"^\s*(none|nothing|no clothes|no clothing|naked|nude)\b", re.I)

DEFAULT_COVER = "wearing plain, simple, modest clothing that fully covers the torso and hips"


def parse_age(age: str) -> Optional[int]:
    """'32' -> 32, 'early 30s' -> 30, '25-30' -> 25, 'adult' -> 18, '' -> None."""
    numbers = [int(n) for n in re.findall(r"\d+", age or "")]
    if numbers:
        return min(numbers)
    if re.search(r"\badult\b|\bmiddle[- ]aged\b|\belderly\b|\bold (?:man|woman)\b", age or "", re.I):
        return 18
    return None


def is_minor(c: Character) -> bool:
    age = parse_age(c.age)
    return (age is not None and age < 18) or bool(MINOR_WORDS.search(f"{c.age} {c.appearance}"))


def is_confirmed_adult(c: Character) -> bool:
    age = parse_age(c.age)
    return age is not None and age >= 18 and not is_minor(c)


def outfit_is_nude(outfit: str) -> bool:
    return bool(NO_OUTFIT.match(outfit or "")) or bool(NUDITY.search(outfit or ""))


def remove_clauses(text: str, *patterns: re.Pattern) -> str:
    """Drop the clauses of `text` that match any pattern, keep the rest readable."""
    parts = re.split(r"(?<=[.;!?])\s+|,\s*|\s+(?:and|while|with)\s+(?=\w)", text or "")
    kept = [p.strip(" ,.;") for p in parts if p.strip(" ,.;") and not any(pat.search(p) for pat in patterns)]
    return ", ".join(kept)


def age_phrase(c: Character) -> str:
    age = parse_age(c.age)
    if age is None:
        return ""
    if age >= 18 and not is_minor(c):
        return f"an adult, {c.age.strip()} years old, with adult facial features and adult body proportions" \
            if c.age.strip().isdigit() else f"an adult ({c.age.strip()}), with adult facial features and adult body proportions"
    return f"{c.age.strip()} years old" if c.age.strip().isdigit() else c.age.strip()


class PanelSafety:
    """Decides how one panel's characters and action are allowed to appear."""

    def __init__(self, action: str, characters: Iterable[Character]):
        self.characters: List[Character] = list(characters)
        outfits = " ".join(c.outfit for c in self.characters)
        self.nudity = bool(NUDITY.search(action or "")) or any(outfit_is_nude(c.outfit) for c in self.characters)
        all_adults = all(is_confirmed_adult(c) for c in self.characters)
        self.revealing = not all_adults and bool(REVEALING.search(f"{action} {outfits}"))
        self.cover = self.nudity or self.revealing

    def action(self, action: str) -> str:
        if not self.cover:
            return action
        cleaned = remove_clauses(action, NUDITY, REVEALING)
        return cleaned or "the character stands in the scene"

    def outfit(self, c: Character, cover_text: str) -> str:
        if self.cover and (outfit_is_nude(c.outfit) or NUDITY.search(c.outfit) or REVEALING.search(c.outfit)
                           or not c.outfit.strip()):
            return cover_text or DEFAULT_COVER
        return c.outfit


def sheet_outfit(c: Character, cover_text: str) -> str:
    """Reference sheets are always fully clothed: they condition every panel the character is in."""
    if not c.outfit.strip() or outfit_is_nude(c.outfit) or NUDITY.search(c.outfit) or \
            (REVEALING.search(c.outfit) and not is_confirmed_adult(c)):
        return cover_text or DEFAULT_COVER
    return c.outfit

"""Reading chapter files whatever their text encoding.

Chapters often come from TextEdit, Word or old downloads saved as Mac OS Roman or
Windows-1252 rather than UTF-8. Both are 8-bit encodings that decode any byte
sequence without error, so the right one is picked by how plausible the decoded
text looks (curly quotes and dashes are frequent in fiction; stray symbols are not).
"""

from __future__ import annotations

import codecs
import sys
from pathlib import Path

ENCODING_NAMES = {"utf-8": "UTF-8", "utf-8-sig": "UTF-8 (with BOM)", "utf-16": "UTF-16", "utf-32": "UTF-32",
                  "utf-16-le": "UTF-16", "utf-16-be": "UTF-16", "mac_roman": "Mac OS Roman", "cp1252": "Windows-1252"}

_TYPOGRAPHIC = set("‘’“”—–…«»¡¿")
_COMMON_ACCENTED = set("éèàâçêëîïôûùüöäñáíóúßãõœæÉÀÇ")


class ChapterFileError(ValueError):
    pass


def _score(text: str) -> int:
    score = 0
    for ch in text:
        if ch.isascii():
            if ord(ch) < 32 and ch not in "\n\r\t":
                score -= 5
        elif ch in _TYPOGRAPHIC:
            score += 3
        elif ch in _COMMON_ACCENTED:
            score += 1
        else:
            score -= 2
    return score


def decode_chapter(data: bytes, name: str = "the file") -> tuple[str, str]:
    """Return (text, encoding) for the raw bytes of a chapter file."""
    if data.startswith(b"{\\rtf"):
        raise ChapterFileError(f"{name} is an RTF document (TextEdit's rich-text format), not plain text. In TextEdit "
                               "choose Format > Make Plain Text, then save it (encoding: Unicode UTF-8).")
    if data.startswith(b"PK\x03\x04"):
        raise ChapterFileError(f"{name} is a Word / Pages document, not plain text. Export or 'Save As' plain text "
                               "(.txt, UTF-8).")
    if data.startswith(b"%PDF-"):
        raise ChapterFileError(f"{name} is a PDF. Copy the chapter text into a plain .txt file.")

    for bom, encoding in ((codecs.BOM_UTF32_LE, "utf-32"), (codecs.BOM_UTF32_BE, "utf-32"),
                          (codecs.BOM_UTF8, "utf-8-sig"), (codecs.BOM_UTF16_LE, "utf-16"),
                          (codecs.BOM_UTF16_BE, "utf-16")):
        if data.startswith(bom):
            return data.decode(encoding), encoding
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    sample = data[:4000]
    if sample and sample[1::2].count(0) > len(sample) // 4:  # UTF-16 without a BOM (mostly-ASCII text)
        for encoding in ("utf-16-le", "utf-16-be"):
            try:
                return data.decode(encoding), encoding
            except UnicodeDecodeError:
                pass

    candidates = []
    for encoding in ("mac_roman", "cp1252"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:  # cp1252 leaves a few bytes undefined
            continue
        candidates.append((_score(text), encoding == ("mac_roman" if sys.platform == "darwin" else "cp1252"),
                           text, encoding))
    best = max(candidates, key=lambda c: (c[0], c[1]))
    return best[2].replace("\r\n", "\n").replace("\r", "\n"), best[3]


def read_chapter(path: Path) -> tuple[str, str]:
    path = Path(path)
    return decode_chapter(path.read_bytes(), path.name)


def encoding_label(encoding: str) -> str:
    return ENCODING_NAMES.get(encoding, encoding)

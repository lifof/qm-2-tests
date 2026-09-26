"""Chapter files in the encodings people actually have."""

import pytest

from webtoon.textio import ChapterFileError, decode_chapter

TEXT = ("Chapter 1 — The Arrival\n\n“Where am I?” Jason asked. It wasn’t a dream… "
        "The sky was wrong—two suns, one blue.\n\n‘Stay calm,’ he told himself. Café, naïve, señor.")


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "mac_roman", "cp1252"])
def test_round_trip(encoding):
    text, detected = decode_chapter(TEXT.encode(encoding))
    assert text == TEXT
    assert detected == encoding or (encoding == "utf-8-sig" and detected == "utf-8-sig")


def test_mac_roman_em_dash_at_position_8():
    # the reported error: byte 0xd1 at position 8 (an em dash in Mac OS Roman)
    data = "Chapter1— Jason\n\nHe woke up.".encode("mac_roman")
    assert data[8] == 0xD1
    text, encoding = decode_chapter(data)
    assert encoding == "mac_roman" and text.startswith("Chapter1—")


def test_classic_mac_line_endings():
    text, _ = decode_chapter("One—\rTwo’s\r\rThree".encode("mac_roman"))
    assert text == "One—\nTwo’s\n\nThree"


def test_utf16_without_bom():
    text, encoding = decode_chapter(TEXT.encode("utf-16-le"))
    assert text == TEXT and encoding == "utf-16-le"


@pytest.mark.parametrize("data, hint", [
    (b"{\\rtf1\\ansi\\ansicpg1252\\cocoartf2761 Chapter", "Make Plain Text"),
    (b"PK\x03\x04\x14\x00\x06\x00", "Word / Pages"),
    (b"%PDF-1.7\n", "PDF"),
])
def test_rich_documents_get_a_clear_message(data, hint):
    with pytest.raises(ChapterFileError, match=hint):
        decode_chapter(data, "Chapter1.txt")

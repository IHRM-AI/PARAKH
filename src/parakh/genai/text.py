from __future__ import annotations

import re

_HEADER = re.compile(r"^#{1,6}[ \t]*", re.MULTILINE)
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_BULLET = re.compile(r"^[ \t]*[*-][ \t]+", re.MULTILINE)
_BLANKS = re.compile(r"\n{3,}")


def plain_text(text: str) -> str:
    """Strip Markdown emphasis, headers and bullet markers from model output."""
    text = _HEADER.sub("", text)
    text = _BOLD.sub(r"\1", text)
    text = _BULLET.sub("- ", text)
    text = text.replace("*", "").replace("`", "")
    return _BLANKS.sub("\n\n", text).strip()

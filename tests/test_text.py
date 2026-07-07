from parakh.genai.text import plain_text


def test_strips_markdown():
    out = plain_text("**Bold**\n## Heading\n* one\n- two\n`code`")
    assert "*" not in out
    assert "#" not in out
    assert "`" not in out
    assert "Bold" in out and "Heading" in out


def test_collapses_blank_lines():
    assert "\n\n\n" not in plain_text("a\n\n\n\nb")

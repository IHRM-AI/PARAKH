from pathlib import Path

import pytest

from parakh.genai.ocr import OcrClient


def test_offline_client_is_unavailable():
    assert OcrClient(base_url="").available is False


def test_extract_requires_configured_endpoint():
    with pytest.raises(RuntimeError):
        OcrClient(base_url="").extract(Path("does_not_matter.pdf"))

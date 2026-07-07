from __future__ import annotations

from pathlib import Path

import httpx

from parakh.config import settings


class OcrClient:
    """Client for the self-hosted OCR service (multi-language and handwritten).

    Extracts text from scanned statements, GST returns and financial documents
    so unstructured filings become model features.
    """

    def __init__(self, base_url: str | None = None, timeout: int = 300):
        self._base = (base_url if base_url is not None else settings.ocr_service_url).rstrip("/")
        self._timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self._base)

    def extract(self, document: Path) -> str:
        if not self.available:
            raise RuntimeError("OCR endpoint is not configured (set OCR_SERVICE_URL).")
        with document.open("rb") as handle:
            response = httpx.post(
                f"{self._base}/ocr",
                files={"file": (document.name, handle)},
                timeout=self._timeout,
            )
        response.raise_for_status()
        return response.json().get("text", "")

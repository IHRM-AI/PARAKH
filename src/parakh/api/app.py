from __future__ import annotations

from dataclasses import asdict

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from parakh.config import settings
from parakh.consent.artefact import create_consent
from parakh.genai.memo import LenderMemoService
from parakh.genai.ocr import OcrClient
from parakh.monitoring.trajectory import simulate
from parakh.scoring.card import CardService

app = FastAPI(title="PARAKH", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

_ARTIFACT = settings.artifacts_dir / "health_model.joblib"
_service: CardService | None = None


class ConsentRequest(BaseModel):
    purpose_code: str


class ScoreRequest(BaseModel):
    features: dict[str, float]


class WhatIfRequest(BaseModel):
    features: dict[str, float]
    adjustments: dict[str, float]


class MemoRequest(BaseModel):
    borrower: str
    features: dict[str, float]


_memo = LenderMemoService()


@app.on_event("startup")
def _load() -> None:
    global _service
    if _ARTIFACT.exists():
        _service = CardService.from_artifacts(_ARTIFACT)


def _require_service() -> CardService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Model artifact not loaded. Train the model first.")
    return _service


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model_loaded": _service is not None}


@app.post("/consent")
def consent(request: ConsentRequest) -> dict[str, object]:
    try:
        return create_consent(request.purpose_code).model_dump()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/score")
def score(request: ScoreRequest) -> dict[str, object]:
    card = _require_service().build(request.features)
    payload = asdict(card)
    payload["reason_codes"] = [asdict(code) for code in card.reason_codes]
    return payload


@app.post("/whatif")
def whatif(request: WhatIfRequest) -> dict[str, object]:
    return _require_service().what_if(request.features, request.adjustments)


@app.post("/monitor")
def monitor(request: ScoreRequest) -> dict[str, object]:
    service = _require_service()
    return asdict(simulate(service.model, request.features))


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict[str, object]:
    client = OcrClient()
    if not client.available:
        raise HTTPException(
            status_code=503, detail="OCR service not configured. Set OCR_SERVICE_URL."
        )
    suffix = Path(file.filename or "document").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(await file.read())
        path = Path(handle.name)
    try:
        text = client.extract(path)
    finally:
        path.unlink(missing_ok=True)
    return {"filename": file.filename, "characters": len(text), "text": text}


@app.post("/memo")
def memo(request: MemoRequest) -> dict[str, object]:
    card = _require_service().build(request.features)
    return asdict(_memo.draft(request.borrower, card))

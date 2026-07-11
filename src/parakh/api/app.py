from __future__ import annotations

import logging
import math
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict

import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from parakh.api.extraction import demo_extraction, is_demo_request, parse_document_text
from parakh.config import settings, PACKAGE_ROOT
from parakh.consent.artefact import create_consent
from parakh.genai.memo import LenderMemoService
from parakh.genai.ocr import OcrClient
from parakh.monitoring.trajectory import simulate
from parakh.scoring.card import CardService

logger = logging.getLogger("parakh.api")

_ARTIFACT = settings.artifacts_dir / "health_model.joblib"
_service: CardService | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global _service
    if _ARTIFACT.exists():
        _service = CardService.from_artifacts(_ARTIFACT)
        logger.info("model artifact loaded from %s", _ARTIFACT)
    else:
        logger.warning("model artifact not found at %s; scoring endpoints disabled", _ARTIFACT)
    yield


app = FastAPI(title="PARAKH", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request id=%s method=%s path=%s status=%s duration_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["x-request-id"] = request_id
    return response


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Optional API-key gate.

    When ``API_KEY`` is unset the dependency is a no-op, so the live demo keeps
    working without a key. When set, callers must present a matching
    ``x-api-key`` header or receive HTTP 401.
    """
    expected = settings.api_key
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


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


def _require_service() -> CardService:
    if _service is None:
        raise HTTPException(status_code=503, detail="Model artifact not loaded. Train the model first.")
    return _service


def _validate_features(features: Mapping[str, float], expected: list[str]) -> None:
    """Reject unknown, missing or non-finite feature inputs with HTTP 422.

    Without this guard an incomplete or malformed payload reindexes to NaN and
    the model silently returns a NaN probability. Validating up front turns that
    into an explicit, debuggable client error.
    """
    provided = set(features)
    known = set(expected)
    unknown = sorted(provided - known)
    missing = sorted(known - provided)
    non_finite = sorted(
        name
        for name in provided & known
        if not math.isfinite(float(features[name]))
    )
    problems: list[str] = []
    if missing:
        problems.append(f"missing features: {missing}")
    if unknown:
        problems.append(f"unknown features: {unknown}")
    if non_finite:
        problems.append(f"non-finite values: {non_finite}")
    if problems:
        raise HTTPException(status_code=422, detail="; ".join(problems))


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model_loaded": _service is not None}


@app.post("/consent")
def consent(request: ConsentRequest, _: None = Depends(require_api_key)) -> dict[str, object]:
    try:
        return create_consent(request.purpose_code).model_dump()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/score")
def score(request: ScoreRequest, _: None = Depends(require_api_key)) -> dict[str, object]:
    service = _require_service()
    _validate_features(request.features, service.model.feature_names)
    card = service.build(request.features)
    payload = asdict(card)
    payload["reason_codes"] = [asdict(code) for code in card.reason_codes]
    return payload


@app.post("/whatif")
def whatif(request: WhatIfRequest, _: None = Depends(require_api_key)) -> dict[str, object]:
    service = _require_service()
    _validate_features(request.features, service.model.feature_names)
    adjusted = {**request.features, **request.adjustments}
    _validate_features(adjusted, service.model.feature_names)
    return service.what_if(request.features, request.adjustments)


@app.post("/monitor")
def monitor(request: ScoreRequest, _: None = Depends(require_api_key)) -> dict[str, object]:
    service = _require_service()
    _validate_features(request.features, service.model.feature_names)
    return asdict(simulate(service.model, request.features))


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...), _: None = Depends(require_api_key)
) -> dict[str, object]:
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


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    demo: bool = Query(default=False),
    _: None = Depends(require_api_key),
) -> dict[str, object]:
    """Auto-fill the onboarding form from an uploaded document.

    Runs OCR to get the raw text, then parses it into the NewMsmeForm fields and
    returns only the values it is confident about, for the officer to review
    before computing the score. Degrades gracefully: when the OCR service is
    unset or unreachable, a recognised sample filename or ``demo=true`` returns a
    clearly labelled canned fixture instead of failing.
    """
    filename = file.filename

    # Demo requests are an explicit "give me the canned firm" signal, so serve
    # the fixture before touching OCR. This keeps the demo instant even when
    # OCR_SERVICE_URL points at a stopped GPU that would otherwise hang.
    if is_demo_request(filename, demo):
        result = demo_extraction()
        return {"filename": filename, "fields": result.fields, "source": result.source}

    client = OcrClient()
    if not client.available:
        return {
            "filename": filename,
            "fields": {},
            "source": "unavailable",
            "message": "OCR service is offline. Set OCR_SERVICE_URL, or upload a sample document to preview the auto-fill.",
        }

    suffix = Path(filename or "document").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(await file.read())
        path = Path(handle.name)
    try:
        text = client.extract(path)
    except Exception:
        logger.exception("OCR extraction failed for %s", filename)
        return {
            "filename": filename,
            "fields": {},
            "source": "unavailable",
            "message": "OCR service could not be reached. Upload a sample document to preview the auto-fill.",
        }
    finally:
        path.unlink(missing_ok=True)

    return {"filename": filename, "fields": parse_document_text(text), "source": "ocr"}


@app.post("/memo")
def memo(request: MemoRequest, _: None = Depends(require_api_key)) -> dict[str, object]:
    service = _require_service()
    _validate_features(request.features, service.model.feature_names)
    card = service.build(request.features)
    return asdict(_memo.draft(request.borrower, card))


_FRONTEND_DIST = PACKAGE_ROOT / "frontend" / "dist"
if _FRONTEND_DIST.is_dir():
    # Serve the built single-page app from the same origin as the API so a judge
    # can run the whole demo from one process. Mounted last so API routes win;
    # skipped entirely when the frontend has not been built (API-only deploys).
    app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")
